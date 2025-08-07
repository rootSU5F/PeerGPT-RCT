import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np



class SexDistributionCode(Enum):
    """Simple sex distribution classification codes"""
    ADEQUATE_FEMALE_REPRESENTATION = 0    # ≥50% female - NO ISSUE
    LIMITED_FEMALE_REPRESENTATION = 1     # <50% female - MINOR ISSUE
    SEX_NOT_REPORTED = 2                 # No sex information provided


@dataclass
class SexFeatures:
    """Container for sex distribution features"""
    # Extracted sex data
    female_percentage: Optional[float]         # % female participants
    male_percentage: Optional[float]           # % male participants
    female_count: Optional[int]                # Number of females
    male_count: Optional[int]                  # Number of males
    total_participants: Optional[int]          # Total number of participants
    
    # Primary measure
    primary_female_percentage: Optional[float] # Best female % measure found
    
    # Pattern scores
    sex_reported_score: int                    # Evidence of sex reporting
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_sex_info: List[str]


@dataclass
class SexDistributionResult:
    code: SexDistributionCode
    confidence: float
    message: str
    features: SexFeatures
    reasoning: List[str]
    clinical_implications: str


class SimpleSexDistributionClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_sex_patterns()
        self._setup_clinical_implications()
    
    def _setup_sex_patterns(self):
        """Define patterns for sex extraction - HIGHEST ACCURACY"""
        
        # Female percentage patterns (comprehensive coverage)
        self.female_percentage_patterns = [
            # Direct percentage patterns - HIGHEST PRIORITY
            (r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+(?:participants?|patients?|subjects?)\s+)?(?:were\s+)?female', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?women', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*female\s*(?:participants?|patients?|subjects?)?', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*women', 3),
            
            # Reverse patterns
            (r'female\s*[:\s]*(\d+(?:\.\d+)?)\s*%', 3),
            (r'women\s*[:\s]*(\d+(?:\.\d+)?)\s*%', 3),
            (r'female\s*(?:participants?|patients?|subjects?)\s*[:\s]*(\d+(?:\.\d+)?)\s*%', 3),
            
            # Table/demographic patterns
            (r'Female\s*[,\s]*n\s*\(%\)\s*[:\s]*\d+\s*\((\d+(?:\.\d+)?)\)', 3),
            (r'Female\s*[,\s]*No\.\s*\(%\)\s*[:\s]*\d+\s*\((\d+(?:\.\d+)?)\)', 3),
            (r'Sex.*?Female.*?(\d+(?:\.\d+)?)\s*%', 2),
            (r'Gender.*?Female.*?(\d+(?:\.\d+)?)\s*%', 2),
            
            # Baseline characteristics patterns
            (r'baseline\s+characteristics.*?(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?female', 2),
            (r'demographics.*?(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?female', 2),
            (r'patient\s+characteristics.*?(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?female', 2),
            
            # Complex patterns with context
            (r'(\d+(?:\.\d+)?)\s*%\s*of\s+the\s+(?:study\s+)?(?:participants?|patients?|subjects?)\s+were\s+female', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*of\s+(?:participants?|patients?|subjects?)\s+were\s+women', 3),
            
            # In parentheses patterns
            (r'female\s*[:\s]*(\d+(?:\.\d+)?)', 2),
            (r'women\s*[:\s]*(\d+(?:\.\d+)?)', 2)
        ]
        
        # Male percentage patterns (for calculation if female not found)
        self.male_percentage_patterns = [
            (r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+(?:participants?|patients?|subjects?)\s+)?(?:were\s+)?male', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?men', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*male\s*(?:participants?|patients?|subjects?)?', 3),
            (r'(\d+(?:\.\d+)?)\s*%\s*men', 3),
            (r'male\s*[:\s]*(\d+(?:\.\d+)?)\s*%', 3),
            (r'men\s*[:\s]*(\d+(?:\.\d+)?)\s*%', 3),
            (r'Male\s*[,\s]*n\s*\(%\)\s*[:\s]*\d+\s*\((\d+(?:\.\d+)?)\)', 3)
        ]
        
        # Count-based patterns (number of females/males with percentages)
        self.female_count_patterns = [
            # Count with percentage in parentheses
            (r'(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*(?:were\s+)?female', 3),
            (r'(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*(?:were\s+)?women', 3),
            (r'(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*female\s*(?:participants?|patients?|subjects?)', 3),
            
            # Complex demographic reporting
            (r'(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*(?:participants?|patients?|subjects?)\s+were\s+female', 3),
            (r'of\s+the\s+\d+\s+(?:participants?|patients?|subjects?),?\s*(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*were\s+female', 3),
            
            # Simple count patterns
            (r'(\d+)\s*female\s*(?:participants?|patients?|subjects?)', 2),
            (r'(\d+)\s*women', 2),
            (r'female\s*[:\s]*(\d+)', 1),
            (r'women\s*[:\s]*(\d+)', 1)
        ]
        
        # Male count patterns (for calculation)
        self.male_count_patterns = [
            (r'(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*(?:were\s+)?male', 3),
            (r'(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*(?:were\s+)?men', 3),
            (r'(\d+)\s*male\s*(?:participants?|patients?|subjects?)', 2),
            (r'(\d+)\s*men', 2),
            (r'male\s*[:\s]*(\d+)', 1)
        ]
        
        # Total participant patterns (for calculating percentages from counts)
        self.total_participants_patterns = [
            (r'(\d+)\s*(?:participants?|patients?|subjects?|individuals?)\s+were\s+(?:enrolled|randomized|included|recruited)', 3),
            (r'total\s+of\s+(\d+)\s*(?:participants?|patients?|subjects?)', 3),
            (r'(\d+)\s*(?:participants?|patients?|subjects?)\s+(?:completed|analyzed|included)', 2),
            (r'sample\s+size\s*[:\s]*(\d+)', 2),
            (r'n\s*=\s*(\d+)', 2),
            (r'(\d+)\s*(?:participants?|patients?|subjects?)', 1)
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            SexDistributionCode.ADEQUATE_FEMALE_REPRESENTATION: 
                "Adequate female representation (≥50%) supports generalizability across sexes. "
                "Results are more likely to be applicable to both male and female populations.",
                
            SexDistributionCode.LIMITED_FEMALE_REPRESENTATION:
                "This study population includes fewer than 50% females, limiting the applicability of "
                "results across sexes. Findings may not generalize well to female populations.",
                
            SexDistributionCode.SEX_NOT_REPORTED:
                "Sex/gender distribution not reported. Without sex demographics, generalizability "
                "across male and female populations cannot be assessed."
        }
    
    def check_sex_distribution(self, text: str) -> SexDistributionResult:
        """
        Simple check: Is female representation ≥50% or <50%?
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            SexDistributionResult with simple classification
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                SexDistributionCode.SEX_NOT_REPORTED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"]
            )
        
        # Extract sex features
        features = self._extract_sex_features(text, reasoning)
        
        # Calculate sex reporting score
        sex_reported_score = self._calculate_sex_reporting_score(features)
        features.sex_reported_score = sex_reported_score
        
        reasoning.append(f"Sex reporting score: {sex_reported_score}")
        
        # Make simple classification
        return self._make_sex_classification(features, reasoning)
    
    def _extract_sex_features(self, text: str, reasoning: List[str]) -> SexFeatures:
        """Extract sex distribution with HIGHEST ACCURACY focus"""
        
        # Initialize variables
        female_percentage = None
        male_percentage = None
        female_count = None
        male_count = None
        total_participants = None
        all_extractions = []
        
        # Extract female percentages (HIGHEST PRIORITY)
        for pattern, priority in self.female_percentage_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    percentage = float(match.group(1))
                    if 0 <= percentage <= 100:
                        all_extractions.append(f"female {percentage}% (pattern priority {priority})")
                        if female_percentage is None or priority >= 3:  # Higher priority overwrites
                            female_percentage = percentage
                except (ValueError, IndexError):
                    continue
        
        # Extract male percentages
        for pattern, priority in self.male_percentage_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    percentage = float(match.group(1))
                    if 0 <= percentage <= 100:
                        all_extractions.append(f"male {percentage}% (pattern priority {priority})")
                        if male_percentage is None or priority >= 3:
                            male_percentage = percentage
                except (ValueError, IndexError):
                    continue
        
        # Extract female counts (with percentages if available)
        for pattern, priority in self.female_count_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    count = int(match.group(1))
                    if count > 0:
                        all_extractions.append(f"female count {count}")
                        if female_count is None or priority >= 3:
                            female_count = count
                        
                        # Check if percentage is also in the match
                        if len(match.groups()) > 1 and match.group(2):
                            percentage = float(match.group(2))
                            if 0 <= percentage <= 100:
                                all_extractions.append(f"female {percentage}% (from count pattern)")
                                if female_percentage is None or priority >= 3:
                                    female_percentage = percentage
                except (ValueError, IndexError):
                    continue
        
        # Extract male counts
        for pattern, priority in self.male_count_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    count = int(match.group(1))
                    if count > 0:
                        if male_count is None or priority >= 3:
                            male_count = count
                            
                        # Check if percentage is also in the match
                        if len(match.groups()) > 1 and match.group(2):
                            percentage = float(match.group(2))
                            if 0 <= percentage <= 100:
                                if male_percentage is None or priority >= 3:
                                    male_percentage = percentage
                except (ValueError, IndexError):
                    continue
        
        # Extract total participants
        for pattern, priority in self.total_participants_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    total = int(match.group(1))
                    if total > 0:
                        if total_participants is None or priority >= 3:
                            total_participants = total
                            all_extractions.append(f"total participants {total}")
                except (ValueError, IndexError):
                    continue
        
        # CALCULATE MISSING PERCENTAGES (HIGHEST ACCURACY)
        
        # If we have female count and total, calculate female percentage
        if female_percentage is None and female_count is not None and total_participants is not None:
            female_percentage = (female_count / total_participants) * 100
            all_extractions.append(f"calculated female {female_percentage:.1f}% ({female_count}/{total_participants})")
            reasoning.append(f"Calculated female percentage: {female_percentage:.1f}% from count ({female_count}/{total_participants})")
        
        # If we have male percentage but not female, calculate female
        if female_percentage is None and male_percentage is not None:
            female_percentage = 100 - male_percentage
            all_extractions.append(f"calculated female {female_percentage:.1f}% (100% - {male_percentage}% male)")
            reasoning.append(f"Calculated female percentage: {female_percentage:.1f}% (100% - {male_percentage}% male)")
        
        # If we have male count and total, and no female data, calculate
        if female_percentage is None and male_count is not None and total_participants is not None:
            calculated_female_count = total_participants - male_count
            female_percentage = (calculated_female_count / total_participants) * 100
            all_extractions.append(f"calculated female {female_percentage:.1f}% ({calculated_female_count}/{total_participants})")
            reasoning.append(f"Calculated female percentage: {female_percentage:.1f}% from male count")
        
        # Set primary female percentage
        primary_female_percentage = female_percentage
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(text)
        
        # Create extracted sex info
        extracted_sex_info = []
        if primary_female_percentage is not None:
            extracted_sex_info.append(f"Female: {primary_female_percentage:.1f}%")
        if male_percentage is not None:
            extracted_sex_info.append(f"Male: {male_percentage:.1f}%")
        if female_count is not None:
            extracted_sex_info.append(f"Female count: {female_count}")
        if total_participants is not None:
            extracted_sex_info.append(f"Total: {total_participants}")
        
        if all_extractions:
            reasoning.append(f"All sex extractions: {all_extractions}")
            if primary_female_percentage is not None:
                reasoning.append(f"Primary female percentage: {primary_female_percentage:.1f}%")
        else:
            reasoning.append("No sex distribution data found")
        
        return SexFeatures(
            female_percentage=female_percentage,
            male_percentage=male_percentage,
            female_count=female_count,
            male_count=male_count,
            total_participants=total_participants,
            primary_female_percentage=primary_female_percentage,
            sex_reported_score=0,  # Will be set later
            detected_patterns=detected_patterns,
            extracted_sex_info=extracted_sex_info
        )
    
    def _calculate_sex_reporting_score(self, features: SexFeatures) -> int:
        """Calculate quality of sex reporting"""
        score = 0
        
        if features.female_percentage is not None:
            score += 3
        if features.male_percentage is not None:
            score += 2
        if features.female_count is not None:
            score += 2
        if features.total_participants is not None:
            score += 1
        
        return score
    
    def _get_detected_patterns(self, text: str) -> List[str]:
        """Get detected patterns for transparency"""
        detected = []
        
        # Check for female percentage patterns
        for pattern, priority in self.female_percentage_patterns[:5]:  # Top 5 patterns
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"female_%: {pattern[:40]}...")
        
        # Check for male percentage patterns
        for pattern, priority in self.male_percentage_patterns[:3]:  # Top 3 patterns
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"male_%: {pattern[:40]}...")
        
        return detected[:5]  # Limit for readability
    
    def _make_sex_classification(self, features: SexFeatures, reasoning: List[str]) -> SexDistributionResult:
        """Make simple classification: ≥50% female vs <50% female"""
        
        female_percentage = features.primary_female_percentage
        sex_reported_score = features.sex_reported_score
        
        # SIMPLE BINARY LOGIC - EXACTLY WHAT YOU WANT
        
        # 1. Clear female percentage available (highest confidence)
        if female_percentage is not None:
            reasoning.append(f"Using female percentage: {female_percentage:.1f}%")
            
            if female_percentage >= 50:
                # Adequate female representation - NO ISSUE
                reasoning.append(f"Female ≥ 50%: ADEQUATE representation")
                return self._create_result(
                    SexDistributionCode.ADEQUATE_FEMALE_REPRESENTATION,
                    0.95,
                    f"Adequate female representation: {female_percentage:.1f}% (≥50%)",
                    features,
                    reasoning
                )
            else:
                # Limited female representation - MINOR ISSUE
                reasoning.append(f"Female < 50%: LIMITED representation - MINOR ISSUE")
                return self._create_result(
                    SexDistributionCode.LIMITED_FEMALE_REPRESENTATION,
                    0.95,
                    f"Limited female representation: {female_percentage:.1f}% (<50%)",
                    features,
                    reasoning
                )
        
        # 2. Some sex reporting but percentage unclear
        elif sex_reported_score >= 2:
            reasoning.append("Some sex data found but percentage unclear")
            return self._create_result(
                SexDistributionCode.SEX_NOT_REPORTED,
                0.7,
                "Sex data present but female percentage unclear",
                features,
                reasoning
            )
        
        # 3. Default: No sex information
        else:
            reasoning.append("No sex distribution information found")
            return self._create_result(
                SexDistributionCode.SEX_NOT_REPORTED,
                0.9,
                "Sex distribution not reported",
                features,
                reasoning
            )
    
    def _create_result(self, code: SexDistributionCode, confidence: float, 
                      message: str, features: SexFeatures, 
                      reasoning: List[str]) -> SexDistributionResult:
        """Create a SexDistributionResult object"""
        return SexDistributionResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code]
        )
    
    def _create_empty_features(self) -> SexFeatures:
        """Create empty features object for error cases"""
        return SexFeatures(
            female_percentage=None, male_percentage=None, female_count=None,
            male_count=None, total_participants=None, primary_female_percentage=None,
            sex_reported_score=0, detected_patterns=[], extracted_sex_info=[]
        )




def run_check(abstract : str):# just a wrapper method
    classifier = SimpleSexDistributionClassifier()
    result = classifier.check_sex_distribution(abstract)
    return result