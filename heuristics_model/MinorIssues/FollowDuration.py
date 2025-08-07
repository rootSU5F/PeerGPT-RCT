import re
from unittest import result
import spacy
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np



class FollowUpCode(Enum):
    """Follow-up duration classification codes - Simple binary assessment"""
    ADEQUATE = 0              # >=1 year or clearly long-term
    LESS_THAN_ONE_YEAR = 1    # <1 year - Minor issue
    NOT_PROVIDED = 2          # No duration mentioned - Minor issue


@dataclass
class FollowUpFeatures:
    """Container for follow-up duration features"""
    # Duration metrics
    explicit_duration_months: Optional[float]
    max_duration_found: Optional[float]
    
    # Pattern scores
    short_duration_score: int      # <1 year indicators
    adequate_duration_score: int   # >=1 year indicators
    no_followup_score: int        # No mention of follow-up
    
    # Text characteristics
    temporal_density: float
    has_explicit_duration: bool
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_durations: List[str]


@dataclass
class FollowUpResult:
    code: FollowUpCode
    confidence: float
    message: str
    features: FollowUpFeatures
    reasoning: List[str]
    clinical_implications: str


class FollowUpDurationClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_duration_patterns()
        self._setup_clinical_implications()
    
    def _setup_duration_patterns(self):
        """Define patterns for duration classification - Simple binary approach"""
        
        # Short duration patterns (<1 year) - MINOR ISSUE
        self.short_duration_patterns = [
            # Explicit short durations (Strong indicators: 3 points)
            (r'\b(?:[1-9]|10|11)\s*(?:month|mo)s?\b', 3),          # 1-11 months
            (r'\b(?:[1-9]\d?)\s*(?:week|wk)s?\b', 3),              # weeks
            (r'\b(?:[1-9]\d?)\s*(?:day|d)s?\b', 2),                # days
            (r'\bshort-?term\s+follow-?up\b', 3),
            (r'\bacute\s+follow-?up\b', 2),
            
            # Common short timeframes (Strong indicators: 3 points)
            (r'\b(?:3|6|9)\s*month\s+(?:follow-?up|study)\b', 3),
            (r'\b(?:6|9)\s*(?:month|mo)s?\b', 3),
            
            # Follow-up phrases indicating short duration (Moderate: 2 points)
            (r'\bfollow(?:ed|ing)?\s+(?:for|up)\s+(?:[1-9])\s*month', 2),
            (r'\bafter\s+(?:[1-9])\s*month', 2),
            (r'\b(?:early|immediate)\s+follow-?up\b', 1),
            
            # Quarters indicating <1 year
            (r'\b(?:first|second|third)\s+quarter\b', 1),
            (r'\bQ[1-3]\b', 1)
        ]
        
        # Adequate duration patterns (>=1 year) - NO ISSUE
        self.adequate_duration_patterns = [
            # Explicit adequate durations (Strong indicators: 3 points)
            (r'\b(?:1|2|3|4|5)\+?\s*(?:year|yr)s?\b', 3),          # 1+ years
            (r'\b(?:12|15|18|24|36|48|60)\s*(?:month|mo)s?\b', 3), # 12+ months
            (r'\bone\s+year\s+follow-?up\b', 3),
            (r'\blong-?term\s+follow-?up\b', 3),
            (r'\bextended\s+follow-?up\b', 3),
            
            # Duration ranges indicating >=1 year (Moderate: 2 points)
            (r'\b(?:1-2|2-3|1-5)\s*year', 2),
            (r'\b12-(?:18|24|36)\s*month', 2),
            (r'\bmedium-?term\s+follow-?up\b', 2),
            (r'\bintermediate\s+follow-?up\b', 2),
            
            # General long-term indicators (Weak: 1 point)
            (r'\bchronic\s+follow-?up\b', 1),
            (r'\bmulti-?year\b', 1),
            (r'\bfollow(?:ed|ing)?\s+(?:for|up)\s+(?:12|24)\s*month', 2)
        ]
        
        # No follow-up patterns - NOT PROVIDED
        self.no_followup_patterns = [
            # Cross-sectional indicators (Strong: 3 points)
            (r'\bcross-?sectional\s+(?:study|design|analysis)\b', 3),
            (r'\bsingle\s+(?:time\s*point|assessment|visit)\b', 3),
            (r'\bbaseline\s+(?:only|data|assessment)\b', 2),
            (r'\bno\s+follow-?up\b', 3),
            (r'\bsnapshot\s+(?:study|analysis)\b', 2),
            
            # One-time assessment indicators (Moderate: 2 points)
            (r'\bone-?time\s+(?:assessment|measurement|evaluation)\b', 2),
            (r'\bsingle\s+(?:measurement|observation)\b', 1),
            (r'\bat\s+(?:admission|baseline|enrollment)\s+only\b', 2)
        ]
        
        # Duration extraction patterns (for numerical analysis)
        self.duration_extraction_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(?:year|yr)s?\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:month|mo)s?\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:week|wk)s?\b'
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            FollowUpCode.ADEQUATE: 
                "Follow-up duration appears adequate (≥1 year) for assessing meaningful clinical outcomes, "
                "sustained efficacy, and safety profile. Sufficient time for most treatment effects and "
                "adverse events to emerge.",
                
            FollowUpCode.LESS_THAN_ONE_YEAR:
                "The follow-up duration is under one year, which may be insufficient to capture "
                "meaningful clinical outcomes, especially for chronic conditions or interventions with "
                "delayed effects. This limits the ability to assess sustained efficacy, adverse events, "
                "and long-term risk.",
                
            FollowUpCode.NOT_PROVIDED:
                "The duration of follow-up was not provided in the abstract. For studies less than a "
                "year, important clinical outcomes may not have had sufficient time to develop, "
                "limiting the ability to draw meaningful conclusions about long-term risks or benefits."
        }
    
    def check_followup_duration(self, text: str) -> FollowUpResult:
        """
        Simple heuristic check: Is follow-up <1 year or not provided?
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            FollowUpResult with binary classification
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                FollowUpCode.NOT_PROVIDED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"]
            )
        
        # Extract features
        features = self._extract_followup_features(text, reasoning)
        
        # Extract explicit durations
        extracted_durations = self._extract_durations(text, reasoning)
        features.extracted_durations = extracted_durations
        
        # Calculate pattern scores
        short_score = self._score_patterns(text, self.short_duration_patterns)
        adequate_score = self._score_patterns(text, self.adequate_duration_patterns)
        no_followup_score = self._score_patterns(text, self.no_followup_patterns)
        
        features.short_duration_score = short_score
        features.adequate_duration_score = adequate_score
        features.no_followup_score = no_followup_score
        
        reasoning.append(f"Pattern scores - Short: {short_score}, Adequate: {adequate_score}, No follow-up: {no_followup_score}")
        
        # Make simple binary classification
        return self._make_simple_classification(features, reasoning)
    
    def _extract_followup_features(self, text: str, reasoning: List[str]) -> FollowUpFeatures:
        """Extract basic follow-up features from text"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Calculate temporal density (proportion of time-related terms)
        temporal_terms = len(re.findall(
            r'\b(?:follow|time|duration|period|month|year|week|day|baseline)\b', 
            clean_text
        ))
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        temporal_density = temporal_terms / total_words if total_words > 0 else 0
        
        # Check if explicit duration mentioned
        has_explicit_duration = bool(re.search(
            r'\b\d+(?:\.\d+)?\s*(?:year|month|week|day)', clean_text
        ))
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)
        
        return FollowUpFeatures(
            explicit_duration_months=None,  # Will be set by duration extraction
            max_duration_found=None,
            short_duration_score=0,         # Will be set later
            adequate_duration_score=0,
            no_followup_score=0,
            temporal_density=temporal_density,
            has_explicit_duration=has_explicit_duration,
            detected_patterns=detected_patterns,
            extracted_durations=[]
        )
    
    def _extract_durations(self, text: str, reasoning: List[str]) -> List[str]:
        """Extract and convert duration mentions to months"""
        durations_in_months = []
        extracted_strings = []
        
        # Year patterns
        year_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:year|yr)s?\b', text.lower())
        for match in year_matches:
            months = float(match) * 12
            durations_in_months.append(months)
            extracted_strings.append(f"{match} years ({months:.1f} months)")
        
        # Month patterns
        month_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:month|mo)s?\b', text.lower())
        for match in month_matches:
            months = float(match)
            durations_in_months.append(months)
            extracted_strings.append(f"{match} months")
        
        # Week patterns (convert to months)
        week_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:week|wk)s?\b', text.lower())
        for match in week_matches:
            months = float(match) / 4.33  # Average weeks per month
            durations_in_months.append(months)
            extracted_strings.append(f"{match} weeks ({months:.1f} months)")
        
        if durations_in_months:
            reasoning.append(f"Extracted durations: {extracted_strings}")
        else:
            reasoning.append("No explicit durations found in text")
        
        return extracted_strings
    
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
            ('short', self.short_duration_patterns),
            ('adequate', self.adequate_duration_patterns),
            ('no_followup', self.no_followup_patterns)
        ]
        
        for category, patterns in all_pattern_sets:
            for pattern, score in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(f"{category}: {pattern[:30]}...")
        
        return detected[:5]  # Limit for readability
    
    def _make_simple_classification(self, features: FollowUpFeatures, reasoning: List[str]) -> FollowUpResult:
        """Make simple binary classification: <1 year, not provided, or adequate"""
        
        # Extract numerical durations if available
        durations_months = []
        for duration_str in features.extracted_durations:
            # Extract months from strings like "24 months" or "2 years (24.0 months)"
            month_match = re.search(r'\((\d+(?:\.\d+)?)\s*months?\)', duration_str)
            if month_match:
                durations_months.append(float(month_match.group(1)))
            else:
                # Direct month extraction
                month_match = re.search(r'^(\d+(?:\.\d+)?)\s*months?', duration_str)
                if month_match:
                    durations_months.append(float(month_match.group(1)))
        
        if durations_months:
            features.explicit_duration_months = max(durations_months)
            features.max_duration_found = max(durations_months)
        
        # SIMPLE BINARY LOGIC - exactly matching your heuristic table
        
        # 1. Check explicit numerical duration first (highest confidence)
        if features.explicit_duration_months is not None:
            duration = features.explicit_duration_months
            reasoning.append(f"Using explicit duration: {duration:.1f} months")
            
            if duration < 12:  # Less than 1 year
                reasoning.append(f"Duration < 12 months: MINOR ISSUE")
                return self._create_result(
                    FollowUpCode.LESS_THAN_ONE_YEAR,
                    0.95,
                    f"Short follow-up: {duration:.1f} months (<1 year)",
                    features,
                    reasoning
                )
            else:  # 12+ months = adequate
                reasoning.append(f"Duration ≥ 12 months: ADEQUATE")
                return self._create_result(
                    FollowUpCode.ADEQUATE,
                    0.95,
                    f"Adequate follow-up: {duration:.1f} months (≥1 year)",
                    features,
                    reasoning
                )
        
        # 2. If no explicit duration, use pattern scores
        short_score = features.short_duration_score
        adequate_score = features.adequate_duration_score
        no_followup_score = features.no_followup_score
        
        # Cross-sectional or explicit "no follow-up" = NOT PROVIDED
        if no_followup_score >= 3:
            reasoning.append("Clear indicators of no follow-up or cross-sectional design")
            return self._create_result(
                FollowUpCode.NOT_PROVIDED,
                0.9,
                "No follow-up duration provided (cross-sectional or single assessment)",
                features,
                reasoning
            )
        
        # Strong short-term indicators = LESS THAN ONE YEAR
        if short_score >= 3 and short_score > adequate_score:
            reasoning.append("Strong short-term follow-up indicators")
            return self._create_result(
                FollowUpCode.LESS_THAN_ONE_YEAR,
                0.8,
                "Short-term follow-up indicated (<1 year)",
                features,
                reasoning
            )
        
        # Strong adequate indicators = ADEQUATE
        if adequate_score >= 3 and adequate_score > short_score:
            reasoning.append("Strong adequate follow-up indicators")
            return self._create_result(
                FollowUpCode.ADEQUATE,
                0.8,
                "Adequate follow-up indicated (≥1 year)",
                features,
                reasoning
            )
        
        # Default: If no clear indicators = NOT PROVIDED
        reasoning.append("No clear follow-up duration indicators found")
        return self._create_result(
            FollowUpCode.NOT_PROVIDED,
            0.7,
            "Follow-up duration not clearly specified",
            features,
            reasoning
        )
    
    def _create_result(self, code: FollowUpCode, confidence: float, 
                      message: str, features: FollowUpFeatures, 
                      reasoning: List[str]) -> FollowUpResult:
        """Create a FollowUpResult object"""
        return FollowUpResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code]
        )
    
    def _create_empty_features(self) -> FollowUpFeatures:
        """Create empty features object for error cases"""
        return FollowUpFeatures(
            explicit_duration_months=None, max_duration_found=None,
            short_duration_score=0, adequate_duration_score=0, no_followup_score=0,
            temporal_density=0, has_explicit_duration=False,
            detected_patterns=[], extracted_durations=[]
        )



def test_followup_classifier():
    """Test the follow-up duration classifier with simple binary logic"""
    print("🕐 Testing Simple Follow-up Duration Heuristic Classifier")
    print("🎯 Binary Classification: <1 year (Minor Issue) | Not Provided (Minor Issue) | Adequate")
    print("=" * 80)
    
    try:
        classifier = FollowUpDurationClassifier()
        print("✅ Follow-up classifier initialized successfully\n")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        return
    
    test_cases = [
        # Less than 1 year - MINOR ISSUE
        ("This 6-month study evaluated the efficacy of the intervention. Patients were assessed monthly for 6 months following treatment initiation.", 
         "< 1 year"),
        
        ("Follow-up duration was 9 months. Primary endpoint was assessed at 6 and 9 months post-treatment.",
         "< 1 year"),
        
        # Adequate (≥1 year) - NO ISSUE  
        ("Participants were followed for 18 months after randomization. Primary endpoint was assessed at 12 months with safety follow-up extending to 24 months.",
         "≥ 1 year"),
        
        ("This randomized controlled trial followed 1000 patients for 2 years to assess cardiovascular outcomes.",
         "≥ 1 year"),
        
        # Not provided - MINOR ISSUE
        ("This randomized trial compared drug A versus placebo in 500 patients. Primary outcome was clinical response. Secondary outcomes included safety measures.",
         "Not provided"),
        
        ("This cross-sectional study examined the prevalence of hypertension in 2000 adults. Data were collected at a single time point.",
         "Not provided"),
        
        # Edge case: exactly 1 year
        ("Follow-up was conducted for 12 months after treatment initiation.",
         "Exactly 1 year"),
    ]
    
    for i, (text, expected) in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: Expected {expected}")
        print(f"📝 Input: {text[:80]}...")
        print("-" * 50)
        
        result = classifier.check_followup_duration(text)
        
        # Simple status indicator
        if result.code == FollowUpCode.LESS_THAN_ONE_YEAR:
            status = "❌ MINOR ISSUE"
        elif result.code == FollowUpCode.NOT_PROVIDED:
            status = "❌ MINOR ISSUE" 
        else:
            status = "✅ ADEQUATE"
        
        print(f"🏷️  Classification: {result.code.name} {status}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"💬 Message: {result.message}")
        
        # Show extracted duration if found
        if result.features.extracted_durations:
            print(f"📊 Extracted: {result.features.extracted_durations}")
        
        if result.features.explicit_duration_months:
            months = result.features.explicit_duration_months
            print(f"🕐 Duration: {months:.1f} months ({'<12' if months < 12 else '≥12'} months)")
        
        # Show key reasoning
        print(f"🧠 Reasoning: {result.reasoning[-1]}")
        
        # Show clinical implications (truncated)
        print(f"🏥 Clinical Impact: {result.clinical_implications[:100]}...")
        
        print("=" * 50)



def run_check(abstract : str):# just a wrapper method
    classifier = FollowUpDurationClassifier()
    result = classifier.check_followup_duration(abstract)
    return result