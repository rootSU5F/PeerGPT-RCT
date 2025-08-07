
import re
import spacy
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class MulticentreCode(Enum):
    """Multicentre classification codes - Simple binary assessment"""
    MULTICENTRE_CONFIRMED = 0     # Clear evidence of multiple centres
    SINGLE_CENTRE_MENTIONED = 1   # Explicitly mentioned as single centre
    NOT_MENTIONED = 2            # No mention - assume single centre (MINOR ISSUE)


@dataclass
class MulticentreFeatures:
    """Container for multicentre-related features"""
    # Pattern scores
    multicentre_indicators_score: int    # Clear multicentre evidence
    single_centre_indicators_score: int  # Explicit single centre mentions
    centre_ambiguous_score: int          # Unclear centre information
    
    # Extracted information
    number_of_centres: Optional[int]     # Extracted number if mentioned
    centre_locations: List[str]          # Geographic locations mentioned
    
    # Text characteristics
    geographic_density: float            # Density of location/geographic terms
    institutional_mentions: int          # Number of institutions mentioned
    has_explicit_centre_info: bool       # Any explicit centre information
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_centre_info: List[str]


@dataclass
class MulticentreResult:
    code: MulticentreCode
    confidence: float
    message: str
    features: MulticentreFeatures
    reasoning: List[str]
    clinical_implications: str


class MulticentreClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_multicentre_patterns()
        self._setup_clinical_implications()
    
    def _setup_multicentre_patterns(self):
        """Define patterns for multicentre classification"""
        
        # Multicentre indicators (Strong evidence: 3 points)
        self.multicentre_patterns = [
            # Explicit multicentre mentions
            (r'\bmulti-?cent(?:er|re)\b', 3),
            (r'\bmultiple\s+cent(?:er|re)s\b', 3),
            (r'\bmulti-?site\b', 3),
            (r'\bmultiple\s+sites?\b', 3),
            (r'\bmulti-?institutional\b', 3),
            (r'\bmultiple\s+institutions?\b', 3),
            
            # Numerical centre indicators (Strong: 3 points)
            (r'\b(\d+)\s+cent(?:er|re)s?\b', 3),           # "5 centres", "12 centers"
            (r'\b(\d+)\s+sites?\b', 3),                    # "8 sites", "15 site"
            (r'\b(\d+)\s+hospitals?\b', 2),                # "4 hospitals"
            (r'\b(\d+)\s+institutions?\b', 2),             # "6 institutions"
            
            # International/geographic indicators (Strong: 3 points)
            (r'\binternational\s+(?:study|trial|collaboration)\b', 3),
            (r'\bmulti-?national\b', 3),
            (r'\bglobal\s+(?:study|trial)\b', 3),
            (r'\bworldwide\s+(?:study|trial)\b', 2),
            
            # Network/consortium indicators (Moderate: 2 points)
            (r'\bnetwork\s+of\s+cent(?:er|re)s\b', 2),
            (r'\bconsortium\b', 2),
            (r'\bcollaborative\s+(?:study|trial)\b', 2),
            (r'\bparticipating\s+cent(?:er|re)s\b', 2),
            (r'\bcoordinating\s+cent(?:er|re)\b', 2),
            
            # Geographic diversity indicators (Moderate: 2 points)
            (r'\b(?:across|from)\s+(?:multiple|several)\s+(?:countries|regions|cities)\b', 2),
            (r'\binternational\s+(?:sites?|locations?)\b', 2),
            (r'\b(?:europe|asia|america|africa|australia).*?(?:and|,).*?(?:europe|asia|america|africa|australia)\b', 2),
            
            # Recruitment language suggesting multiple sites (Weak: 1 point)
            (r'\brecruited\s+(?:from|across)\s+multiple\b', 1),
            (r'\benrolled\s+(?:from|at)\s+(?:multiple|various|several)\b', 1)
        ]
        
        # Single centre indicators (Explicit single centre mentions)
        self.single_centre_patterns = [
            # Explicit single centre (Strong: 3 points)
            (r'\bsingle-?cent(?:er|re)\b', 3),
            (r'\bone\s+cent(?:er|re)\b', 3),
            (r'\bsingle\s+site\b', 3),
            (r'\bone\s+site\b', 3),
            (r'\bsingle\s+institution\b', 3),
            (r'\bone\s+institution\b', 3),
            (r'\bsingle\s+hospital\b', 3),
            
            # Single location indicators (Moderate: 2 points)
            (r'\bat\s+our\s+(?:institution|hospital|centre|center)\b', 2),
            (r'\bat\s+this\s+(?:institution|hospital|centre|center)\b', 2),
            (r'\bat\s+a\s+single\s+(?:institution|hospital|centre|center)\b', 2),
            (r'\bmonocentric\b', 3),
            
            # Institution-specific language (Weak: 1 point)
            (r'\bat\s+our\s+(?:department|clinic|facility)\b', 1),
            (r'\bin\s+our\s+(?:department|clinic|facility)\b', 1)
        ]
        
        # Ambiguous centre information (neither clearly single nor multi)
        self.ambiguous_centre_patterns = [
            # Vague institutional mentions (Weak: 1 point)
            (r'\bat\s+(?:the\s+)?(?:hospital|clinic|medical\s+center)\b', 1),
            (r'\bfrom\s+(?:the\s+)?(?:hospital|clinic|medical\s+center)\b', 1),
            (r'\bparticipants?\s+were\s+recruited\b', 1),
            (r'\bpatients?\s+were\s+enrolled\b', 1),
            
            # Generic study location language
            (r'\bstudy\s+(?:site|location|setting)\b', 1),
            (r'\btreatment\s+(?:site|facility)\b', 1)
        ]
        
        # Geographic location patterns (for extraction)
        self.geographic_patterns = [
            r'\b(?:USA|US|United\s+States|America)\b',
            r'\b(?:UK|United\s+Kingdom|Britain|England|Scotland|Wales)\b',
            r'\b(?:Canada|Australia|Germany|France|Italy|Spain|Netherlands|Sweden|Denmark|Norway|Finland)\b',
            r'\b(?:China|Japan|Korea|India|Brazil|Mexico|Argentina|Chile)\b',
            r'\b(?:New\s+York|London|Paris|Berlin|Tokyo|Sydney|Toronto|Boston|Chicago|Los\s+Angeles)\b'
        ]
        
        # Number extraction patterns for centres
        self.centre_number_patterns = [
            r'\b(\d+)\s+cent(?:er|re)s?\b',
            r'\b(\d+)\s+sites?\b',
            r'\b(\d+)\s+hospitals?\b',
            r'\b(\d+)\s+institutions?\b'
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            MulticentreCode.MULTICENTRE_CONFIRMED: 
                "Multicentre design enhances external validity and generalizability of findings. "
                "Results are more likely to be reproducible across different clinical settings, "
                "populations, and healthcare systems. Reduces single-site bias and increases "
                "confidence in treatment effects.",
                
            MulticentreCode.SINGLE_CENTRE_MENTIONED:
                "Single-centre design explicitly mentioned. While this allows for standardized "
                "protocols and reduced variability, it may limit generalizability to other "
                "settings and populations. Results should be interpreted with consideration "
                "of potential site-specific factors.",
                
            MulticentreCode.NOT_MENTIONED:
                "The abstract lacked details on whether the study was single-centred or "
                "multicentred. This often means the study was single-centred. Single-centre "
                "studies, even RCTs, are often hard to replicate."
        }
    
    def check_multicentre_status(self, text: str) -> MulticentreResult:
        """
        Simple heuristic check: Is multicentre status mentioned?
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            MulticentreResult with classification
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                MulticentreCode.NOT_MENTIONED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"]
            )
        
        # Extract features
        features = self._extract_multicentre_features(text, reasoning)
        
        # Calculate pattern scores
        multicentre_score = self._score_patterns(text, self.multicentre_patterns)
        single_centre_score = self._score_patterns(text, self.single_centre_patterns)
        ambiguous_score = self._score_patterns(text, self.ambiguous_centre_patterns)
        
        features.multicentre_indicators_score = multicentre_score
        features.single_centre_indicators_score = single_centre_score
        features.centre_ambiguous_score = ambiguous_score
        
        reasoning.append(f"Pattern scores - Multicentre: {multicentre_score}, "
                        f"Single-centre: {single_centre_score}, Ambiguous: {ambiguous_score}")
        
        # Extract number of centres if mentioned
        centre_numbers = self._extract_centre_numbers(text, reasoning)
        if centre_numbers:
            features.number_of_centres = max(centre_numbers)
        
        # Extract geographic locations
        locations = self._extract_locations(text, reasoning)
        features.centre_locations = locations
        
        # Make classification
        return self._make_multicentre_classification(features, reasoning)
    
    def _extract_multicentre_features(self, text: str, reasoning: List[str]) -> MulticentreFeatures:
        """Extract multicentre-related features from text"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Calculate geographic density (proportion of location/geographic terms)
        geographic_terms = len(re.findall(
            r'\b(?:centre|center|site|hospital|institution|clinic|international|national|country|city|location|facility)\b', 
            clean_text
        ))
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        geographic_density = geographic_terms / total_words if total_words > 0 else 0
        
        # Count institutional mentions
        institutional_mentions = len(re.findall(
            r'\b(?:hospital|clinic|medical\s+center|institution|university|department)\b',
            clean_text
        ))
        
        # Check if any explicit centre information mentioned
        has_explicit_centre_info = bool(re.search(
            r'\b(?:centre|center|site|multicentre|single.centre|multi.site)\b', clean_text
        ))
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)
        
        # Extract centre-related information
        extracted_info = self._extract_centre_info(clean_text)
        
        return MulticentreFeatures(
            multicentre_indicators_score=0,      # Will be set later
            single_centre_indicators_score=0,
            centre_ambiguous_score=0,
            number_of_centres=None,              # Will be set by number extraction
            centre_locations=[],                 # Will be set by location extraction
            geographic_density=geographic_density,
            institutional_mentions=institutional_mentions,
            has_explicit_centre_info=has_explicit_centre_info,
            detected_patterns=detected_patterns,
            extracted_centre_info=extracted_info
        )
    
    def _extract_centre_numbers(self, text: str, reasoning: List[str]) -> List[int]:
        """Extract number of centres mentioned"""
        numbers = []
        
        for pattern in self.centre_number_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    num = int(match)
                    if 1 <= num <= 1000:  # Reasonable range for number of centres
                        numbers.append(num)
                except ValueError:
                    continue
        
        if numbers:
            reasoning.append(f"Extracted centre numbers: {numbers}")
        
        return numbers
    
    def _extract_locations(self, text: str, reasoning: List[str]) -> List[str]:
        """Extract geographic locations mentioned"""
        locations = []
        
        for pattern in self.geographic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.extend(matches)
        
        # Remove duplicates while preserving order
        unique_locations = []
        for loc in locations:
            if loc not in unique_locations:
                unique_locations.append(loc)
        
        if unique_locations:
            reasoning.append(f"Geographic locations found: {unique_locations}")
        
        return unique_locations[:5]  # Limit for readability
    
    def _extract_centre_info(self, text: str) -> List[str]:
        """Extract any centre-related information for transparency"""
        info = []
        
        # Look for centre-related sentences
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if re.search(r'\b(?:centre|center|site|multicentre|hospital|institution)\b', 
                        sentence.lower()):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10:  # Avoid very short fragments
                    info.append(clean_sentence[:100])  # Truncate long sentences
        
        return info[:3]  # Limit to first 3 relevant sentences
    
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
            ('multicentre', self.multicentre_patterns),
            ('single_centre', self.single_centre_patterns),
            ('ambiguous', self.ambiguous_centre_patterns)
        ]
        
        for category, patterns in all_pattern_sets:
            for pattern, score in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(f"{category}: {pattern[:40]}...")
        
        return detected[:6]  # Limit for readability
    
    def _make_multicentre_classification(self, features: MulticentreFeatures, 
                                       reasoning: List[str]) -> MulticentreResult:
        """Make simple classification: multicentre confirmed, single-centre, or not mentioned"""
        
        multicentre_score = features.multicentre_indicators_score
        single_centre_score = features.single_centre_indicators_score
        ambiguous_score = features.centre_ambiguous_score
        
        # SIMPLE BINARY LOGIC - exactly matching the heuristic
        
        # 1. Strong multicentre evidence (≥3 points) = MULTICENTRE_CONFIRMED
        if multicentre_score >= 3:
            reasoning.append(f"Strong multicentre evidence (score: {multicentre_score})")
            
            # Extra confidence if number of centres specified
            confidence = 0.95
            message = "Multicentre study confirmed"
            
            if features.number_of_centres:
                message += f" ({features.number_of_centres} centres)"
                confidence = 0.98
            
            if len(features.centre_locations) > 1:
                message += f" with multiple locations: {', '.join(features.centre_locations[:3])}"
                confidence = 0.98
            
            return self._create_result(
                MulticentreCode.MULTICENTRE_CONFIRMED,
                confidence,
                message,
                features,
                reasoning
            )
        
        # 2. Strong single-centre evidence (≥3 points) = SINGLE_CENTRE_MENTIONED
        elif single_centre_score >= 3:
            reasoning.append(f"Explicit single-centre mentioned (score: {single_centre_score})")
            return self._create_result(
                MulticentreCode.SINGLE_CENTRE_MENTIONED,
                0.9,
                "Single-centre study explicitly mentioned",
                features,
                reasoning
            )
        
        # 3. Moderate multicentre evidence but not strong = MULTICENTRE_CONFIRMED (lower confidence)
        elif multicentre_score >= 2 and multicentre_score > single_centre_score:
            reasoning.append(f"Moderate multicentre evidence (score: {multicentre_score})")
            return self._create_result(
                MulticentreCode.MULTICENTRE_CONFIRMED,
                0.75,
                f"Multicentre study likely (pattern score: {multicentre_score})",
                features,
                reasoning
            )
        
        # 4. Some single-centre evidence but not strong
        elif single_centre_score >= 2:
            reasoning.append(f"Some single-centre indicators (score: {single_centre_score})")
            return self._create_result(
                MulticentreCode.SINGLE_CENTRE_MENTIONED,
                0.7,
                f"Single-centre study suggested (pattern score: {single_centre_score})",
                features,
                reasoning
            )
        
        # 5. Only ambiguous evidence or no clear evidence = NOT_MENTIONED (MINOR ISSUE)
        else:
            if ambiguous_score > 0:
                reasoning.append(f"Only ambiguous centre information (score: {ambiguous_score})")
                message = "Centre information ambiguous - assume single centre"
            else:
                reasoning.append("No clear centre information found")
                message = "Multicentre status not mentioned - assume single centre"
            
            return self._create_result(
                MulticentreCode.NOT_MENTIONED,
                0.8,
                message,
                features,
                reasoning
            )
    
    def _create_result(self, code: MulticentreCode, confidence: float, 
                      message: str, features: MulticentreFeatures, 
                      reasoning: List[str]) -> MulticentreResult:
        """Create a MulticentreResult object"""
        return MulticentreResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code]
        )
    
    def _create_empty_features(self) -> MulticentreFeatures:
        """Create empty features object for error cases"""
        return MulticentreFeatures(
            multicentre_indicators_score=0, single_centre_indicators_score=0,
            centre_ambiguous_score=0, number_of_centres=None, centre_locations=[],
            geographic_density=0, institutional_mentions=0, has_explicit_centre_info=False,
            detected_patterns=[], extracted_centre_info=[]
        )



def run_check(abstract : str):# just a wrapper method
    classifier = MulticentreClassifier()
    result = classifier.check_multicentre_status(abstract)
    return result
