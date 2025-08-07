import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np



class SampleSizeCode(Enum):
    """Sample size classification codes"""
    ADEQUATE_SAMPLE_SIZE = 0        # ≥1000 participants - Good
    MODERATE_SAMPLE_SIZE = 1        # 500-1000 participants - MAJOR ISSUE (limits effect detection)
    SMALL_SAMPLE_SIZE = 2          # <500 participants - MAJOR ISSUE (chance findings, confounding)
    SAMPLE_SIZE_NOT_REPORTED = 3    # No sample size information


@dataclass
class SampleSizeFeatures:
    """Container for sample size features"""
    # Extracted sample size data
    total_sample_size: Optional[int]           # Primary sample size found
    randomized_sample_size: Optional[int]      # Number randomized
    completed_sample_size: Optional[int]       # Number completed/analyzed
    enrolled_sample_size: Optional[int]        # Number enrolled
    
    # Multiple sample size mentions
    all_sample_sizes: List[Tuple[int, str]]    # All sample sizes with context
    sample_size_sources: List[str]             # Where sample sizes were found
    
    # Power analysis indicators
    power_calculation_mentioned: bool          # Power/sample size calculation mentioned
    power_analysis_details: List[str]         # Power analysis information
    
    # Study design context
    study_type_context: str                    # RCT, observational, pilot, etc.
    primary_outcome_context: str              # What primary outcome is measured
    effect_size_mentioned: bool               # Effect size considerations mentioned
    
    # Sample size quality indicators
    sample_size_reporting_score: int          # Quality of sample size reporting
    enrollment_completion_ratio: Optional[float] # Ratio of completed to enrolled
    
    # Text characteristics
    statistical_power_detail: float           # Density of power-related terms
    sample_size_confidence: float             # Confidence in extracted sample size
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_sample_info: List[str]


@dataclass
class SampleSizeResult:
    code: SampleSizeCode
    confidence: float
    message: str
    features: SampleSizeFeatures
    reasoning: List[str]
    clinical_implications: str
    power_assessment: str


class SampleSizeClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_sample_size_patterns()
        self._setup_clinical_implications()
    
    def _setup_sample_size_patterns(self):
        """Define patterns for sample size extraction - HIGHEST ACCURACY"""
        
        # Primary sample size patterns (highest priority)
        self.sample_size_patterns = [
            # Total participants patterns (priority 3)
            (r'(\d+)\s*(?:patients?|participants?|subjects?|individuals?)\s+were\s+(?:enrolled|recruited|included|randomized)', 3, 'enrolled'),
            (r'(?:a\s+)?total\s+of\s+(\d+)\s*(?:patients?|participants?|subjects?)', 3, 'total'),
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:were\s+)?(?:included|enrolled|recruited)', 3, 'included'),
            
            # Randomization patterns (priority 3)
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:were\s+)?randomized', 3, 'randomized'),
            (r'randomized\s+(\d+)\s*(?:patients?|participants?|subjects?)', 3, 'randomized'),
            (r'randomization\s+of\s+(\d+)\s*(?:patients?|participants?|subjects?)', 3, 'randomized'),
            
            # Sample size specification patterns (priority 3)
            (r'sample\s+size\s*(?:was\s*|of\s*|:?\s*)(\d+)', 3, 'sample_size'),
            (r'n\s*=\s*(\d+)', 3, 'n_equals'),
            (r'\(n\s*=\s*(\d+)\)', 3, 'n_parentheses'),
            
            # Study population patterns (priority 2)
            (r'(?:the\s+)?study\s+(?:population\s+)?(?:included|comprised|consisted\s+of)\s+(\d+)', 2, 'study_population'),
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:with|having)', 2, 'condition_specific'),
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:aged|age)', 1, 'demographic'),
            
            # Completion/analysis patterns (priority 2)
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:completed|analyzed|finished)', 2, 'completed'),
            (r'(?:analysis\s+(?:included|of)\s+)?(\d+)\s*(?:patients?|participants?|subjects?)', 2, 'analyzed'),
            (r'data\s+(?:from\s+)?(\d+)\s*(?:patients?|participants?|subjects?)', 2, 'data_from'),
            
            # Group allocation patterns (priority 1)
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:in\s+(?:the\s+)?(?:treatment|control|intervention|placebo)\s+group)', 1, 'group_allocation'),
            (r'(\d+)\s*(?:patients?|participants?|subjects?)\s+(?:received|assigned\s+to)', 1, 'treatment_received'),
            
            # Alternative formats (priority 1)
            (r'(?:of\s+the\s+)?(\d+)\s*(?:patients?|participants?|subjects?)', 1, 'of_the'),
            (r'among\s+(\d+)\s*(?:patients?|participants?|subjects?)', 1, 'among')
        ]
        
        # Power analysis and sample size calculation patterns
        self.power_analysis_patterns = [
            # Power calculation patterns (priority 3)
            (r'\bpower\s+(?:analysis|calculation|assessment)\b', 3),
            (r'\bsample\s+size\s+(?:calculation|determination|estimation)\b', 3),
            (r'\bstatistical\s+power\s+(?:analysis|calculation)\b', 3),
            (r'\bpower\s+(?:was\s+)?calculated\b', 3),
            
            # Effect size patterns (priority 2)
            (r'\beffect\s+size\b', 2),
            (r'\bminimum\s+(?:detectable\s+)?(?:difference|effect)\b', 2),
            (r'\bclinically\s+(?:meaningful|significant)\s+(?:difference|effect)\b', 2),
            (r'\bdetectable\s+(?:difference|effect)\b', 2),
            
            # Power parameters (priority 2)
            (r'\b(?:80|90)\s*%\s*power\b', 2),
            (r'\bpower\s+(?:of\s+)?(?:80|90)\s*%\b', 2),
            (r'\balpha\s*=\s*0\.05\b', 1),
            (r'\bbeta\s*=\s*0\.(?:1|2)\b', 1),
            (r'\btype\s+(?:I|II)\s+error\b', 1),
            
            # Sample size justification (priority 1)
            (r'\bsample\s+size\s+(?:was\s+)?(?:based\s+on|determined\s+by|calculated\s+to)\b', 2),
            (r'\b(?:adequate|sufficient)\s+(?:power|sample\s+size)\b', 1),
            (r'\bpowered\s+to\s+detect\b', 2)
        ]
        
        # Study type context patterns
        self.study_type_patterns = [
            # Randomized controlled trials
            (r'\brandomized\s+(?:controlled\s+)?trial\b', 'RCT'),
            (r'\bRCT\b', 'RCT'),
            (r'\bdouble[-\s]blind\b', 'RCT'),
            (r'\bplacebo[-\s]controlled\b', 'RCT'),
            
            # Observational studies
            (r'\bcohort\s+study\b', 'cohort'),
            (r'\bcase[-\s]control\s+study\b', 'case-control'),
            (r'\bcross[-\s]sectional\s+study\b', 'cross-sectional'),
            (r'\bobservational\s+study\b', 'observational'),
            
            # Special study types
            (r'\bpilot\s+study\b', 'pilot'),
            (r'\bfeasibility\s+study\b', 'feasibility'),
            (r'\bproof[-\s]of[-\s]concept\b', 'proof-of-concept'),
            (r'\bphase\s+(?:I|1)\b', 'phase-1'),
            (r'\bphase\s+(?:II|2)\b', 'phase-2'),
            (r'\bphase\s+(?:III|3)\b', 'phase-3')
        ]
        
        # Outcome context patterns (for power assessment)
        self.outcome_patterns = [
            # Hard endpoints (need larger samples)
            (r'\bmortality\b', 'mortality'),
            (r'\bdeath\b', 'mortality'),
            (r'\bsurvival\b', 'survival'),
            (r'\bmajor\s+(?:adverse\s+)?(?:cardiac\s+)?events?\b', 'MACE'),
            
            # Intermediate endpoints (moderate samples)
            (r'\bblood\s+pressure\b', 'BP'),
            (r'\bcholesterol\b', 'lipids'),
            (r'\bHbA1c\b', 'diabetes'),
            (r'\bglucose\b', 'glucose'),
            
            # Surrogate endpoints (smaller samples acceptable)
            (r'\bbiomarker\b', 'biomarker'),
            (r'\blaboratory\s+(?:values?|parameters?)\b', 'lab_values'),
            (r'\bimaging\s+(?:parameters?|measures?)\b', 'imaging')
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            SampleSizeCode.ADEQUATE_SAMPLE_SIZE: 
                "Adequate sample size (≥1000 participants) provides sufficient statistical power "
                "to detect clinically meaningful effects and reduces the risk of chance findings. "
                "Results are more likely to be reliable and generalizable.",
                
            SampleSizeCode.MODERATE_SAMPLE_SIZE:
                "This was a relatively small study, which limits the ability to detect a small effect "
                "size. Moderate sample size (500-1000 participants) may be adequate for large effect "
                "sizes but insufficient for detecting smaller but clinically meaningful differences.",
                
            SampleSizeCode.SMALL_SAMPLE_SIZE:
                "This was a small study, which increases the possibility that results are due to "
                "chance alone and that not all baseline confounders were balanced. Small sample size "
                "(<500 participants) significantly limits statistical power and generalizability.",
                
            SampleSizeCode.SAMPLE_SIZE_NOT_REPORTED:
                "Sample size not clearly reported. Without knowing the number of participants, "
                "it's impossible to assess statistical power and the reliability of findings. "
                "This limits the interpretation of study results."
        }
    
    def check_sample_size(self, text: str) -> SampleSizeResult:
        """
        Analyze sample size and assess statistical power adequacy
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            SampleSizeResult with sample size assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                SampleSizeCode.SAMPLE_SIZE_NOT_REPORTED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess sample size"
            )
        
        # Extract comprehensive sample size features
        features = self._extract_sample_size_features(text, reasoning)
        
        # Calculate pattern scores
        sample_size_score = self._calculate_sample_size_reporting_score(features)
        features.sample_size_reporting_score = sample_size_score
        
        # Analyze study context
        study_context = self._analyze_study_context(text, features, reasoning)
        features.study_type_context = study_context
        
        # Assess power implications
        power_assessment = self._assess_power_implications(features, reasoning)
        
        # Make sample size classification
        return self._make_sample_size_classification(features, reasoning, power_assessment)
    
    def _extract_sample_size_features(self, text: str, reasoning: List[str]) -> SampleSizeFeatures:
        """Extract comprehensive sample size features with HIGHEST ACCURACY"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Extract all sample sizes with context
        all_sample_sizes = []
        sample_size_sources = []
        
        for pattern, priority, context in self.sample_size_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    sample_size = int(match.group(1))
                    # Reasonable sample size range (avoid extracting dates, ages, etc.)
                    if 10 <= sample_size <= 100000:
                        all_sample_sizes.append((sample_size, context, priority))
                        sample_size_sources.append(f"{context}: {sample_size}")
                except (ValueError, IndexError):
                    continue
        
        # Determine primary sample sizes
        total_sample_size = None
        randomized_sample_size = None
        completed_sample_size = None
        enrolled_sample_size = None
        
        if all_sample_sizes:
            # Sort by priority (higher priority first)
            all_sample_sizes.sort(key=lambda x: x[2], reverse=True)
            
            # Extract specific types of sample sizes
            for size, context, priority in all_sample_sizes:
                if context in ['total', 'sample_size', 'n_equals'] and total_sample_size is None:
                    total_sample_size = size
                elif context in ['randomized'] and randomized_sample_size is None:
                    randomized_sample_size = size
                elif context in ['completed', 'analyzed'] and completed_sample_size is None:
                    completed_sample_size = size
                elif context in ['enrolled', 'included', 'recruited'] and enrolled_sample_size is None:
                    enrolled_sample_size = size
            
            # Determine primary sample size (use the most reliable measure)
            if randomized_sample_size:
                total_sample_size = randomized_sample_size
            elif total_sample_size is None and enrolled_sample_size:
                total_sample_size = enrolled_sample_size
            elif total_sample_size is None and completed_sample_size:
                total_sample_size = completed_sample_size
            elif total_sample_size is None:
                # Use the largest sample size found (likely the enrollment)
                total_sample_size = max(all_sample_sizes, key=lambda x: x[0])[0]
        
        # Check for power analysis mentions
        power_calculation_mentioned = False
        power_analysis_details = []
        
        for pattern, priority in self.power_analysis_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                power_calculation_mentioned = True
                matches = re.findall(pattern, text, re.IGNORECASE)
                power_analysis_details.extend(matches)
        
        # Check for effect size mentions
        effect_size_mentioned = bool(re.search(
            r'\beffect\s+size\b|\bminimum\s+(?:detectable\s+)?(?:difference|effect)\b', 
            text, re.IGNORECASE
        ))
        
        # Calculate enrollment completion ratio
        enrollment_completion_ratio = None
        if enrolled_sample_size and completed_sample_size:
            enrollment_completion_ratio = completed_sample_size / enrolled_sample_size
        
        # Calculate statistical power detail
        power_terms = len(re.findall(
            r'\b(?:power|sample\s+size|effect\s+size|statistical|significance|alpha|beta)\w*\b', 
            clean_text
        ))
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        statistical_power_detail = power_terms / total_words if total_words > 0 else 0
        
        # Assess sample size confidence
        sample_size_confidence = self._assess_sample_size_confidence(all_sample_sizes, power_calculation_mentioned)
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)
        
        # Create extracted sample info
        extracted_sample_info = []
        if total_sample_size:
            extracted_sample_info.append(f"Primary sample size: {total_sample_size}")
        if randomized_sample_size and randomized_sample_size != total_sample_size:
            extracted_sample_info.append(f"Randomized: {randomized_sample_size}")
        if completed_sample_size and completed_sample_size != total_sample_size:
            extracted_sample_info.append(f"Completed: {completed_sample_size}")
        if power_calculation_mentioned:
            extracted_sample_info.append("Power calculation mentioned")
        
        if all_sample_sizes:
            reasoning.append(f"Sample sizes extracted: {sample_size_sources}")
            if total_sample_size:
                reasoning.append(f"Primary sample size: {total_sample_size}")
        else:
            reasoning.append("No sample size information found")
        
        return SampleSizeFeatures(
            total_sample_size=total_sample_size,
            randomized_sample_size=randomized_sample_size,
            completed_sample_size=completed_sample_size,
            enrolled_sample_size=enrolled_sample_size,
            all_sample_sizes=[(size, context) for size, context, _ in all_sample_sizes],
            sample_size_sources=sample_size_sources,
            power_calculation_mentioned=power_calculation_mentioned,
            power_analysis_details=power_analysis_details,
            study_type_context="",              # Will be set later
            primary_outcome_context="",         # Will be set later
            effect_size_mentioned=effect_size_mentioned,
            sample_size_reporting_score=0,     # Will be set later
            enrollment_completion_ratio=enrollment_completion_ratio,
            statistical_power_detail=statistical_power_detail,
            sample_size_confidence=sample_size_confidence,
            detected_patterns=detected_patterns,
            extracted_sample_info=extracted_sample_info
        )
    
    def _assess_sample_size_confidence(self, all_sample_sizes: List[Tuple[int, str, int]], 
                                     power_mentioned: bool) -> float:
        """Assess confidence in extracted sample size"""
        
        if not all_sample_sizes:
            return 0.0
        
        # Base confidence on highest priority pattern found
        max_priority = max(priority for _, _, priority in all_sample_sizes)
        confidence = max_priority / 3.0  # Normalize to 0-1
        
        # Boost confidence if multiple consistent sample sizes
        primary_sizes = [size for size, _, priority in all_sample_sizes if priority >= 2]
        if len(set(primary_sizes)) == 1:  # All high-priority sizes are the same
            confidence += 0.2
        
        # Boost confidence if power analysis mentioned
        if power_mentioned:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_sample_size_reporting_score(self, features: SampleSizeFeatures) -> int:
        """Calculate quality of sample size reporting"""
        score = 0
        
        if features.total_sample_size is not None:
            score += 3
        if features.randomized_sample_size is not None:
            score += 2
        if features.completed_sample_size is not None:
            score += 1
        if features.power_calculation_mentioned:
            score += 3
        if features.effect_size_mentioned:
            score += 2
        if features.enrollment_completion_ratio is not None:
            score += 1
        
        return score
    
    def _analyze_study_context(self, text: str, features: SampleSizeFeatures, reasoning: List[str]) -> str:
        """Analyze study type context for sample size assessment"""
        
        # Determine study type
        study_type = "unclear"
        for pattern, study_type_name in self.study_type_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                study_type = study_type_name
                break
        
        # Determine primary outcome context
        outcome_type = "unclear"
        for pattern, outcome_name in self.outcome_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                outcome_type = outcome_name
                features.primary_outcome_context = outcome_type
                break
        
        # Assess context for sample size needs
        if study_type == "pilot" or study_type == "feasibility":
            context_assessment = "Pilot/feasibility study (smaller samples expected)"
        elif study_type == "phase-1":
            context_assessment = "Phase 1 study (small samples typical for safety)"
        elif outcome_type == "mortality":
            context_assessment = "Mortality outcome (large samples typically needed)"
        elif study_type == "RCT":
            context_assessment = "Randomized controlled trial (adequate power important)"
        else:
            context_assessment = f"Study type: {study_type}"
        
        reasoning.append(f"Study context: {context_assessment}")
        return context_assessment
    
    def _assess_power_implications(self, features: SampleSizeFeatures, reasoning: List[str]) -> str:
        """Assess power and statistical implications of sample size"""
        
        sample_size = features.total_sample_size
        study_context = features.study_type_context
        power_mentioned = features.power_calculation_mentioned
        
        if sample_size is None:
            assessment = "UNKNOWN power (sample size not reported)"
        elif sample_size >= 1000:
            if power_mentioned:
                assessment = f"ADEQUATE power (n={sample_size} with power calculation)"
            else:
                assessment = f"LIKELY ADEQUATE power (n={sample_size})"
        elif sample_size >= 500:
            if "Pilot" in study_context or "Phase 1" in study_context:
                assessment = f"REASONABLE for study type (n={sample_size})"
            else:
                assessment = f"MODERATE power (n={sample_size}) - may miss small effects"
        elif sample_size >= 100:
            if "Pilot" in study_context:
                assessment = f"REASONABLE for pilot study (n={sample_size})"
            else:
                assessment = f"LOW power (n={sample_size}) - risk of chance findings"
        else:
            assessment = f"VERY LOW power (n={sample_size}) - high risk of unreliable results"
        
        reasoning.append(f"Power assessment: {assessment}")
        return assessment
    
    def _get_detected_patterns(self, text: str) -> List[str]:
        """Get list of detected patterns for transparency"""
        detected = []
        
        # Check sample size patterns
        for pattern, priority, context in self.sample_size_patterns[:8]:  # Top patterns
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"sample_size_{context}: {pattern[:40]}...")
        
        # Check power analysis patterns
        for pattern, priority in self.power_analysis_patterns[:4]:  # Top patterns
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"power: {pattern[:40]}...")
        
        return detected[:6]  # Limit for readability
    
    def _make_sample_size_classification(self, features: SampleSizeFeatures, 
                                       reasoning: List[str], power_assessment: str) -> SampleSizeResult:
        """Make sample size classification based on clinical thresholds"""
        
        sample_size = features.total_sample_size
        study_context = features.study_type_context
        confidence = features.sample_size_confidence
        
        # SAMPLE SIZE CLASSIFICATION LOGIC - Based on your heuristic
        
        # 1. Clear sample size available
        if sample_size is not None:
            reasoning.append(f"Using sample size: {sample_size}")
            
            if sample_size >= 1000:
                # Adequate sample size - NO ISSUE
                reasoning.append(f"Sample size ≥ 1000: ADEQUATE statistical power")
                return self._create_result(
                    SampleSizeCode.ADEQUATE_SAMPLE_SIZE,
                    min(0.95, confidence + 0.1),
                    f"Adequate sample size: {sample_size} participants (≥1000)",
                    features,
                    reasoning,
                    power_assessment
                )
            elif sample_size >= 500:
                # Moderate sample size - MAJOR ISSUE (limits effect detection)
                reasoning.append(f"Sample size 500-1000: MAJOR ISSUE - limits ability to detect small effects")
                return self._create_result(
                    SampleSizeCode.MODERATE_SAMPLE_SIZE,
                    min(0.95, confidence + 0.1),
                    f"Moderate sample size: {sample_size} participants (500-1000) - limits effect detection",
                    features,
                    reasoning,
                    power_assessment
                )
            else:
                # Small sample size - MAJOR ISSUE (chance findings, confounding)
                reasoning.append(f"Sample size < 500: MAJOR ISSUE - risk of chance findings and confounding")
                return self._create_result(
                    SampleSizeCode.SMALL_SAMPLE_SIZE,
                    min(0.95, confidence + 0.1),
                    f"Small sample size: {sample_size} participants (<500) - high risk of unreliable results",
                    features,
                    reasoning,
                    power_assessment
                )
        
        # 2. Some sample size reporting but unclear
        elif features.sample_size_reporting_score >= 2:
            reasoning.append("Some sample size information found but primary size unclear")
            return self._create_result(
                SampleSizeCode.SAMPLE_SIZE_NOT_REPORTED,
                0.7,
                "Sample size information present but unclear",
                features,
                reasoning,
                power_assessment
            )
        
        # 3. Default: No sample size information
        else:
            reasoning.append("No sample size information found")
            return self._create_result(
                SampleSizeCode.SAMPLE_SIZE_NOT_REPORTED,
                0.9,
                "Sample size not reported",
                features,
                reasoning,
                power_assessment
            )
    
    def _create_result(self, code: SampleSizeCode, confidence: float, 
                      message: str, features: SampleSizeFeatures, 
                      reasoning: List[str], power_assessment: str) -> SampleSizeResult:
        """Create a SampleSizeResult object"""
        return SampleSizeResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code],
            power_assessment=power_assessment
        )
    
    def _create_empty_features(self) -> SampleSizeFeatures:
        """Create empty features object for error cases"""
        return SampleSizeFeatures(
            total_sample_size=None, randomized_sample_size=None, completed_sample_size=None,
            enrolled_sample_size=None, all_sample_sizes=[], sample_size_sources=[],
            power_calculation_mentioned=False, power_analysis_details=[],
            study_type_context="", primary_outcome_context="", effect_size_mentioned=False,
            sample_size_reporting_score=0, enrollment_completion_ratio=None,
            statistical_power_detail=0, sample_size_confidence=0,
            detected_patterns=[], extracted_sample_info=[]
        )





def run_check(abstract : str):# just a wrapper method
    classifier = SampleSizeClassifier()
    result = classifier.check_sample_size(abstract)
    return result