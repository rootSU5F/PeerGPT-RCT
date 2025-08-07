import re
from unittest import result
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np
import statistics



class AgeGeneralizabilityCode(Enum):
    """Age generalizability classification codes"""
    GOOD_GENERALIZABILITY = 0      # Mean/median age ≥70 years
    LIMITED_GENERALIZABILITY = 1   # Mean/median age <70 years (MINOR ISSUE)
    AGE_NOT_REPORTED = 2          # No age information provided
    AGE_UNCLEAR = 3               # Age information present but unclear


@dataclass
class AgeFeatures:
    """Container for age-related features"""
    # Extracted age data
    mean_age: Optional[float]              # Extracted mean age
    median_age: Optional[float]            # Extracted median age
    primary_age_measure: Optional[float]   # Best age measure found (mean or median)
    age_measure_type: Optional[str]        # "mean" or "median"
    
    # Age range information
    age_range_min: Optional[float]         # Minimum age if range provided
    age_range_max: Optional[float]         # Maximum age if range provided
    age_inclusion_criteria: List[str]      # Age-related inclusion criteria
    
    # Multiple age mentions
    all_ages_found: List[Tuple[float, str]]  # All age values with their context
    age_statistics: List[str]              # Raw age statistics found
    
    # Pattern scores
    age_reported_score: int                # Evidence of age reporting
    elderly_population_score: int          # Evidence of elderly participants
    young_population_score: int            # Evidence of younger participants
    
    # Text characteristics
    demographic_detail_level: float        # How detailed demographic reporting is
    age_reporting_quality: str             # Quality of age reporting
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_age_info: List[str]


@dataclass
class AgeGeneralizabilityResult:
    code: AgeGeneralizabilityCode
    confidence: float
    message: str
    features: AgeFeatures
    reasoning: List[str]
    clinical_implications: str
    generalizability_assessment: str


class AgeGeneralizabilityClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_age_patterns()
        self._setup_clinical_implications()
    
    def _setup_age_patterns(self):
        """Define patterns for age extraction and classification"""
        
        # Age extraction patterns - prioritized by reliability
        self.age_extraction_patterns = [
            # Mean age patterns (highest priority)
            (r'\bmean\s+age\s*(?:was\s*|of\s*|:?\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\.?)?', 'mean', 3),
            (r'\baverage\s+age\s*(?:was\s*|of\s*|:?\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\.?)?', 'mean', 3),
            (r'\bage\s*(?::?\s*|was\s*)(\d+(?:\.\d+)?)\s*(?:±|±\s*|±\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', 'mean', 3),
            (r'\bage\s*(?:=|:)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', 'mean', 2),
            
            # Median age patterns (highest priority) - ENHANCED for your example
            (r'\bmedian\s+age\s*(?:was\s*|of\s*|:?\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\.?)?', 'median', 3),
            (r'\bmedian\s+age\s+of\s+(?:patients?|participants?|subjects?)\s+was\s+(\d+(?:\.\d+)?)', 'median', 3),
            (r'\bthe\s+median\s+age\s+(?:was\s+|of\s+patients?\s+was\s+)?(\d+(?:\.\d+)?)', 'median', 3),
            (r'\bage\s*\(median\)\s*(?:was\s*|:?\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', 'median', 3),
            
            # Age with IQR patterns (for your specific example)
            (r'\bmedian\s+age\s+(?:of\s+patients?\s+)?was\s+(\d+(?:\.\d+)?)\s*\(interquartile\s+range\s+(\d+)[-–](\d+)\)', 'median', 3),
            (r'\bmedian\s+age\s*(?:was\s*)?(\d+(?:\.\d+)?)\s*\(IQR\s+(\d+)[-–](\d+)\)', 'median', 3),
            (r'\bthe\s+median\s+age\s+(?:of\s+patients?\s+)?was\s+(\d+(?:\.\d+)?)\s*\(interquartile\s+range\s+(\d+)[-–](\d+)\)', 'median', 3),
            
            # Age with range patterns
            (r'\bmean\s+age\s*(?:was\s*)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*\((?:range\s*)?(\d+)[-–](\d+)\)', 'mean', 3),
            (r'\bmedian\s+age\s*(?:was\s*)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*\((?:range\s*)?(\d+)[-–](\d+)\)', 'median', 3),
            
            # More specific age reporting patterns - ENHANCED
            (r'\bpatients?\s+(?:had\s+)?(?:a\s+)?median\s+age\s+of\s+(\d+(?:\.\d+)?)', 'median', 3),
            (r'\bparticipants?\s+(?:had\s+)?(?:a\s+)?median\s+age\s+of\s+(\d+(?:\.\d+)?)', 'median', 3),
            (r'\bpatients?\s+(?:had\s+)?(?:a\s+)?mean\s+age\s+of\s+(\d+(?:\.\d+)?)', 'mean', 3),
            (r'\bparticipants?\s+(?:had\s+)?(?:a\s+)?mean\s+age\s+of\s+(\d+(?:\.\d+)?)', 'mean', 3),
            
            # Age range patterns
            (r'\bage(?:d|s)?\s*(?:ranged\s*)?(?:from\s*)?(\d+)[-–](\d+)\s*(?:years?|yrs?)', 'range', 2),
            (r'\bages?\s*(\d+)\s*(?:to|-)?\s*(\d+)\s*(?:years?|yrs?)', 'range', 2),
            (r'\bparticipants?\s+aged\s+(\d+)[-–](\d+)', 'range', 2),
            
            # Generic age mentions (lower priority)
            (r'\bage\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', 'generic', 1),
            (r'\b(\d+)[-–]year[-\s]old', 'generic', 1),
            (r'\b(\d+)\s*(?:years?|yrs?)\s*old', 'generic', 1),
            
            # Demographic table patterns - ENHANCED
            (r'\bAge\s*(?:\(years?\))?:?\s*(\d+(?:\.\d+)?)', 'table', 2),
            (r'\bMean\s*\(SD\)\s*age:?\s*(\d+(?:\.\d+)?)', 'mean', 3),
            (r'\bMedian\s*\(IQR\)\s*age:?\s*(\d+(?:\.\d+)?)', 'median', 3),
            (r'\bAge,\s*(?:years?|yrs?),?\s*(?:mean|median)?\s*(\d+(?:\.\d+)?)', 'table', 2),
            
            # Additional comprehensive patterns for edge cases
            (r'\bbaseline\s+(?:characteristics|demographics).*?(?:median\s+)?age\s*(?:was\s*)?(\d+(?:\.\d+)?)', 'median', 2),
            (r'\bbaseline\s+(?:characteristics|demographics).*?(?:mean\s+)?age\s*(?:was\s*)?(\d+(?:\.\d+)?)', 'mean', 2),
            (r'\bdemographics.*?age\s*(?:was\s*)?(\d+(?:\.\d+)?)', 'generic', 1),
            
            # Age in parentheses patterns
            (r'\bpatients?\s+\(age\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\)', 'generic', 2),
            (r'\bparticipants?\s+\(age\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\)', 'generic', 2),
            (r'\bsubjects?\s+\(age\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\)', 'generic', 2)
        ]
        
        
        # Age inclusion/exclusion criteria patterns
        self.age_criteria_patterns = [
            (r'\b(?:aged|age)\s*(?:≥|>=|>)\s*(\d+)', 'min_age'),
            (r'\b(?:aged|age)\s*(?:≤|<=|<)\s*(\d+)', 'max_age'),
            (r'\b(?:aged|age)\s*(\d+)[-–](\d+)', 'age_range'),
            (r'\b(?:inclusion|eligibility).*?(?:aged|age)\s*(\d+)[-–](\d+)', 'inclusion_range'),
            (r'\b(?:adults?|patients?)\s*(?:aged|age)\s*(?:≥|>=|>)\s*(\d+)', 'adult_min'),
            (r'\bolder\s+adults?\s*\((?:aged\s*)?(?:≥|>=|>)\s*(\d+)', 'elderly_min')
        ]
        
        # Elderly population indicators
        self.elderly_population_patterns = [
            (r'\belderly\s+(?:adults?|patients?|participants?|subjects?)\b', 3),
            (r'\bolder\s+(?:adults?|patients?|participants?|subjects?)\b', 3),
            (r'\bgeriatric\s+(?:population|patients?|participants?)\b', 3),
            (r'\baged?\s+(?:≥|>=|>)\s*(?:70|75|80|85)', 2),
            (r'\bnursing\s+home\s+(?:residents?|patients?)\b', 2),
            (r'\bretirement\s+home\s+(?:residents?|patients?)\b', 2),
            (r'\bseniors?\b', 1),
            (r'\bage\s*≥\s*65', 1)
        ]
        
        # Young population indicators
        self.young_population_patterns = [
            (r'\byoung\s+(?:adults?|patients?|participants?|subjects?)\b', 3),
            (r'\badolescents?\b', 3),
            (r'\bpediatric\s+(?:population|patients?|participants?)\b', 3),
            (r'\bchildren\b', 3),
            (r'\baged?\s+(?:≤|<=|<)\s*(?:50|60|65)', 2),
            (r'\bmiddle[-\s]aged\s+(?:adults?|patients?)\b', 1),
            (r'\bworking[-\s]age\s+(?:adults?|population)\b', 1)
        ]
        
        # Age reporting quality indicators
        self.age_reporting_patterns = [
            (r'\bmean\s+age.*?(?:±|standard\s+deviation|SD)', 'detailed'),
            (r'\bmedian\s+age.*?(?:IQR|interquartile\s+range|range)', 'detailed'),
            (r'\bage.*?(?:\(.*?\))', 'moderate'),
            (r'\bage\s*:?\s*\d+', 'basic')
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            AgeGeneralizabilityCode.GOOD_GENERALIZABILITY: 
                "Mean/median age ≥70 years indicates good representation of older adults. "
                "Findings are more generalizable to elderly populations who often have "
                "different disease patterns, comorbidities, and treatment responses than younger adults.",
                
            AgeGeneralizabilityCode.LIMITED_GENERALIZABILITY:
                "The mean or median age of participants is under 70 years, which may limit the "
                "generalizability of findings to older adults. Elderly patients often have "
                "different disease progression, comorbidity burden, and treatment tolerability "
                "compared to younger populations.",
                
            AgeGeneralizabilityCode.AGE_NOT_REPORTED:
                "Age demographics not reported. Without age information, it's impossible to "
                "assess the generalizability of findings across different age groups. This is "
                "particularly concerning for conditions that vary significantly with age.",
                
            AgeGeneralizabilityCode.AGE_UNCLEAR:
                "Age information provided but unclear or incomplete. Ambiguous age reporting "
                "limits the ability to assess generalizability to specific age populations."
        }
    
    def check_age_generalizability(self, text: str) -> AgeGeneralizabilityResult:
        """
        Analyze age demographics and generalizability to older adults
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            AgeGeneralizabilityResult with generalizability assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                AgeGeneralizabilityCode.AGE_NOT_REPORTED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess generalizability"
            )
        
        # Extract comprehensive age features
        features = self._extract_age_features(text, reasoning)
        
        # Calculate pattern scores
        age_reported_score = self._calculate_age_reporting_score(text, features)
        elderly_score = self._score_patterns(text, self.elderly_population_patterns)
        young_score = self._score_patterns(text, self.young_population_patterns)
        
        features.age_reported_score = age_reported_score
        features.elderly_population_score = elderly_score
        features.young_population_score = young_score
        
        reasoning.append(f"Pattern scores - Age reported: {age_reported_score}, "
                        f"Elderly indicators: {elderly_score}, Young indicators: {young_score}")
        
        # Assess generalizability
        generalizability_assessment = self._assess_generalizability(features, reasoning)
        
        # Make age-based classification
        return self._make_age_classification(features, reasoning, generalizability_assessment)
    
    def _extract_age_features(self, text: str, reasoning: List[str]) -> AgeFeatures:
        """Extract comprehensive age-related features"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Extract all age mentions
        all_ages_found = []
        age_statistics = []
        
        for pattern, age_type, priority in self.age_extraction_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if age_type in ['mean', 'median', 'generic', 'table']:
                        age_value = float(match.group(1))
                        if 18 <= age_value <= 120:  # Reasonable age range
                            all_ages_found.append((age_value, age_type, priority))
                            age_statistics.append(f"{age_type}: {age_value} years")
                    elif age_type == 'range':
                        # Handle both 2-group and 3-group matches for ranges
                        groups = match.groups()
                        if len(groups) >= 2:
                            age1 = float(groups[0])
                            age2 = float(groups[1]) if groups[1] else age1
                            if 18 <= age1 <= 120 and 18 <= age2 <= 120:
                                # Use midpoint of range as representative age
                                midpoint = (age1 + age2) / 2
                                all_ages_found.append((midpoint, 'range_midpoint', priority))
                                age_statistics.append(f"range: {age1}-{age2} years (midpoint: {midpoint})")
                        elif len(groups) >= 3 and groups[2]:
                            # For patterns with IQR: median age + range
                            median_age = float(groups[0])
                            range_min = float(groups[1])
                            range_max = float(groups[2])
                            if 18 <= median_age <= 120:
                                all_ages_found.append((median_age, 'median', priority))
                                age_statistics.append(f"median: {median_age} years (IQR: {range_min}-{range_max})")
                except (ValueError, IndexError, AttributeError):
                    continue
        
        # Determine primary age measures
        mean_age = None
        median_age = None
        primary_age_measure = None
        age_measure_type = None
        
        if all_ages_found:
            # Sort by priority (higher priority first)
            all_ages_found.sort(key=lambda x: x[2], reverse=True)
            
            # Extract mean and median ages
            mean_ages = [age for age, type_, _ in all_ages_found if type_ == 'mean']
            median_ages = [age for age, type_, _ in all_ages_found if type_ == 'median']
            
            if mean_ages:
                mean_age = mean_ages[0]  # Take first (highest priority) mean
            if median_ages:
                median_age = median_ages[0]  # Take first (highest priority) median
            
            # Determine primary age measure (prefer median > mean > others)
            if median_age is not None:
                primary_age_measure = median_age
                age_measure_type = 'median'
            elif mean_age is not None:
                primary_age_measure = mean_age
                age_measure_type = 'mean'
            else:
                # Use highest priority age found
                primary_age_measure = all_ages_found[0][0]
                age_measure_type = all_ages_found[0][1]
        
        # Extract age ranges
        age_range_min = None
        age_range_max = None
        age_inclusion_criteria = []
        
        for pattern, criteria_type in self.age_criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if criteria_type == 'min_age':
                        age_range_min = float(match)
                        age_inclusion_criteria.append(f"minimum age: {match}")
                    elif criteria_type == 'max_age':
                        age_range_max = float(match)
                        age_inclusion_criteria.append(f"maximum age: {match}")
                    elif criteria_type in ['age_range', 'inclusion_range']:
                        if isinstance(match, tuple) and len(match) >= 2:
                            age_range_min = float(match[0])
                            age_range_max = float(match[1])
                            age_inclusion_criteria.append(f"age range: {match[0]}-{match[1]}")
                except (ValueError, TypeError):
                    continue
        
        # Assess demographic detail level
        demographic_terms = len(re.findall(
            r'\b(?:age|demographics|characteristics|baseline|participants|patients|subjects)\b', 
            clean_text
        ))
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        demographic_detail_level = demographic_terms / total_words if total_words > 0 else 0
        
        # Assess age reporting quality
        age_reporting_quality = self._assess_age_reporting_quality(text)
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)
        
        # Create extracted age info
        extracted_age_info = []
        if primary_age_measure:
            extracted_age_info.append(f"Primary age: {primary_age_measure} years ({age_measure_type})")
        if age_range_min or age_range_max:
            range_str = f"Range: {age_range_min or '?'}-{age_range_max or '?'}"
            extracted_age_info.append(range_str)
        
        if all_ages_found:
            reasoning.append(f"Ages extracted: {age_statistics}")
            if primary_age_measure:
                reasoning.append(f"Primary age measure: {primary_age_measure} years ({age_measure_type})")
        else:
            reasoning.append("No explicit age values found")
        
        return AgeFeatures(
            mean_age=mean_age,
            median_age=median_age,
            primary_age_measure=primary_age_measure,
            age_measure_type=age_measure_type,
            age_range_min=age_range_min,
            age_range_max=age_range_max,
            age_inclusion_criteria=age_inclusion_criteria,
            all_ages_found=[(age, type_) for age, type_, _ in all_ages_found],  # Remove priority for storage
            age_statistics=age_statistics,
            age_reported_score=0,              # Will be set later
            elderly_population_score=0,
            young_population_score=0,
            demographic_detail_level=demographic_detail_level,
            age_reporting_quality=age_reporting_quality,
            detected_patterns=detected_patterns,
            extracted_age_info=extracted_age_info
        )
    
    def _calculate_age_reporting_score(self, text: str, features: AgeFeatures) -> int:
        """Calculate how well age is reported"""
        score = 0
        
        # Points for having explicit age measures
        if features.mean_age is not None:
            score += 3
        if features.median_age is not None:
            score += 3
        if features.primary_age_measure is not None:
            score += 2
        
        # Points for age ranges/criteria
        if features.age_range_min or features.age_range_max:
            score += 2
        if features.age_inclusion_criteria:
            score += 1
        
        # Points for detailed reporting
        if re.search(r'\bmean\s+age.*?±', text, re.IGNORECASE):
            score += 2
        if re.search(r'\bmedian\s+age.*?(?:IQR|range)', text, re.IGNORECASE):
            score += 2
        
        return score
    
    def _assess_age_reporting_quality(self, text: str) -> str:
        """Assess the quality of age reporting"""
        
        detailed_patterns = [
            r'\bmean\s+age.*?(?:±|standard\s+deviation|SD)',
            r'\bmedian\s+age.*?(?:IQR|interquartile\s+range)',
            r'\bage.*?(?:range|min|max)'
        ]
        
        moderate_patterns = [
            r'\bmean\s+age\s*:?\s*\d+',
            r'\bmedian\s+age\s*:?\s*\d+',
            r'\bage\s*\(.*?\)'
        ]
        
        basic_patterns = [
            r'\bage\s*:?\s*\d+',
            r'\baged\s+\d+',
            r'\d+\s*years?\s*old'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in detailed_patterns):
            return 'detailed'
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in moderate_patterns):
            return 'moderate'
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in basic_patterns):
            return 'basic'
        else:
            return 'none'
    
    def _assess_generalizability(self, features: AgeFeatures, reasoning: List[str]) -> str:
        """Assess generalizability based on age demographics"""
        
        primary_age = features.primary_age_measure
        elderly_score = features.elderly_population_score
        young_score = features.young_population_score
        
        if primary_age is not None:
            if primary_age >= 70:
                assessment = f"GOOD generalizability to older adults (age: {primary_age} ≥ 70 years)"
            elif primary_age >= 65:
                assessment = f"MODERATE generalizability to older adults (age: {primary_age} years, close to elderly threshold)"
            else:
                assessment = f"LIMITED generalizability to older adults (age: {primary_age} < 70 years)"
        elif elderly_score >= 3:
            assessment = "LIKELY GOOD generalizability (strong elderly population indicators)"
        elif young_score >= 3:
            assessment = "LIKELY LIMITED generalizability (strong young population indicators)"
        else:
            assessment = "UNKNOWN generalizability (insufficient age information)"
        
        reasoning.append(f"Generalizability assessment: {assessment}")
        return assessment
    
    def _score_patterns(self, text: str, patterns: List[Tuple[str, int]]) -> int:
        """Score text against a list of patterns"""
        total_score = 0
        for pattern, points in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                pattern_score = min(matches * points, points * 2)  # Cap per pattern
                total_score += pattern_score
        return total_score
    
    def _get_detected_patterns(self, text: str) -> List[str]:
        """Get list of detected patterns for transparency"""
        detected = []
        
        all_pattern_sets = [
            ('age_extraction', [(p[0], p[2]) for p in self.age_extraction_patterns]),
            ('elderly_pop', self.elderly_population_patterns),
            ('young_pop', self.young_population_patterns)
        ]
        
        for category, patterns in all_pattern_sets:
            for pattern, score in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(f"{category}: {pattern[:40]}...")
        
        return detected[:6]  # Limit for readability
    
    def _make_age_classification(self, features: AgeFeatures, 
                               reasoning: List[str], generalizability_assessment: str) -> AgeGeneralizabilityResult:
        """Make age-based generalizability classification"""
        
        primary_age = features.primary_age_measure
        age_reported_score = features.age_reported_score
        elderly_score = features.elderly_population_score
        young_score = features.young_population_score
        
        # AGE-BASED GENERALIZABILITY LOGIC
        
        # 1. Clear numerical age available (highest confidence)
        if primary_age is not None:
            reasoning.append(f"Using primary age measure: {primary_age} years ({features.age_measure_type})")
            
            if primary_age >= 70:
                # Good generalizability to older adults
                reasoning.append(f"Age ≥ 70 years: GOOD generalizability to older adults")
                return self._create_result(
                    AgeGeneralizabilityCode.GOOD_GENERALIZABILITY,
                    0.95,
                    f"Good generalizability: {features.age_measure_type} age {primary_age} years (≥70)",
                    features,
                    reasoning,
                    generalizability_assessment
                )
            else:
                # Limited generalizability - MINOR ISSUE
                reasoning.append(f"Age < 70 years: LIMITED generalizability to older adults")
                return self._create_result(
                    AgeGeneralizabilityCode.LIMITED_GENERALIZABILITY,
                    0.95,
                    f"Limited generalizability: {features.age_measure_type} age {primary_age} years (<70)",
                    features,
                    reasoning,
                    generalizability_assessment
                )
        
        # 2. Strong elderly population indicators
        elif elderly_score >= 3:
            reasoning.append(f"Strong elderly population indicators (score: {elderly_score})")
            return self._create_result(
                AgeGeneralizabilityCode.GOOD_GENERALIZABILITY,
                0.8,
                f"Likely good generalizability: elderly population indicators (score: {elderly_score})",
                features,
                reasoning,
                generalizability_assessment
            )
        
        # 3. Strong young population indicators
        elif young_score >= 3:
            reasoning.append(f"Strong young population indicators (score: {young_score})")
            return self._create_result(
                AgeGeneralizabilityCode.LIMITED_GENERALIZABILITY,
                0.8,
                f"Likely limited generalizability: young population indicators (score: {young_score})",
                features,
                reasoning,
                generalizability_assessment
            )
        
        # 4. Some age reporting but unclear
        elif age_reported_score >= 2 or features.age_reporting_quality in ['basic', 'moderate']:
            reasoning.append(f"Some age information but primary measure unclear")
            return self._create_result(
                AgeGeneralizabilityCode.AGE_UNCLEAR,
                0.7,
                f"Age information present but unclear (reporting quality: {features.age_reporting_quality})",
                features,
                reasoning,
                generalizability_assessment
            )
        
        # 5. Default: No age information
        else:
            reasoning.append("No clear age information found")
            return self._create_result(
                AgeGeneralizabilityCode.AGE_NOT_REPORTED,
                0.9,
                f"Age demographics not reported",
                features,
                reasoning,
                generalizability_assessment
            )
    
    def _create_result(self, code: AgeGeneralizabilityCode, confidence: float, 
                      message: str, features: AgeFeatures, 
                      reasoning: List[str], generalizability_assessment: str) -> AgeGeneralizabilityResult:
        """Create an AgeGeneralizabilityResult object"""
        return AgeGeneralizabilityResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code],
            generalizability_assessment=generalizability_assessment
        )
    
    def _create_empty_features(self) -> AgeFeatures:
        """Create empty features object for error cases"""
        return AgeFeatures(
            mean_age=None, median_age=None, primary_age_measure=None, age_measure_type=None,
            age_range_min=None, age_range_max=None, age_inclusion_criteria=[],
            all_ages_found=[], age_statistics=[], age_reported_score=0,
            elderly_population_score=0, young_population_score=0,
            demographic_detail_level=0, age_reporting_quality='none',
            detected_patterns=[], extracted_age_info=[]
        )



def run_check(abstract : str):# just a wrapper method
    classifier = AgeGeneralizabilityClassifier()
    result = classifier.check_age_generalizability(abstract)
    return result
