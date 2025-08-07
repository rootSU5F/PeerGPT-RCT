import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class BlindingCode(Enum):
    """Blinding classification codes"""
    PROPERLY_BLINDED = 0           # Double/triple-blind design - Good methodological design
    NOT_BLINDED = 1               # Not provided/open-label - METHODOLOGICAL ISSUE (lack of blinding)
    NO_MENTION_OF_BLINDING = 2    # No mention of blinding - METHODOLOGICAL ISSUE (potential lack of blinding)
    BLINDING_UNCLEAR = 3          # Blinding status unclear or inadequate


@dataclass
class BlindingFeatures:
    """Container for blinding features"""
    # Blinding detection
    blinding_mentioned: bool                   # Any blinding explicitly mentioned
    blinding_level: str                       # Level of blinding (none, single, double, triple)
    blinding_phrases: List[str]               # Specific blinding phrases found
    proper_blinding_mentioned: bool           # Double/triple-blind mentioned
    
    # Explicit blinding statements
    double_blind_mentioned: bool              # Double-blind explicitly stated
    triple_blind_mentioned: bool              # Triple-blind explicitly stated
    single_blind_mentioned: bool              # Single-blind mentioned
    masked_mentioned: bool                    # Masked/masking mentioned
    
    # Lack of blinding indicators
    open_label_mentioned: bool                # Open-label design mentioned
    unblinded_mentioned: bool                 # Unblinded explicitly mentioned
    not_blinded_mentioned: bool               # "Not blinded" explicitly mentioned
    no_blinding_mentioned: bool               # "No blinding" explicitly mentioned
    
    # Blinding quality indicators
    participant_blinding: bool                # Participants blinded
    investigator_blinding: bool               # Investigators blinded
    outcome_assessor_blinding: bool           # Outcome assessors blinded
    statistician_blinding: bool               # Statisticians blinded
    
    # Blinding implementation details
    matching_placebo_mentioned: bool          # Matching placebo mentioned
    identical_appearance_mentioned: bool      # Identical appearance mentioned
    sham_procedure_mentioned: bool            # Sham procedure mentioned
    blinding_method_described: bool           # Method of blinding described
    
    # Study design context
    placebo_controlled: bool                  # Placebo-controlled trial
    active_controlled: bool                   # Active-controlled trial
    surgical_intervention: bool               # Surgical intervention (harder to blind)
    device_intervention: bool                 # Device intervention (harder to blind)
    
    # Blinding assessment
    blinding_success_assessed: bool           # Blinding success evaluated
    blinding_integrity_mentioned: bool        # Blinding integrity discussed
    emergency_unblinding_mentioned: bool      # Emergency unblinding procedures
    
    # Bias risk indicators
    performance_bias_risk: bool               # Risk of performance bias
    detection_bias_risk: bool                 # Risk of detection bias
    subjective_outcomes: bool                 # Subjective outcomes present
    objective_outcomes_only: bool             # Only objective outcomes
    
    # Text characteristics
    methods_section_present: bool             # Methods section identified
    design_description_quality: float        # Quality of design description
    blinding_specificity_score: float        # How specifically blinding is described
    
    # Detected patterns for transparency
    detected_patterns: List[str]
    extracted_blinding_info: List[str]


@dataclass
class BlindingResult:
    code: BlindingCode
    confidence: float
    message: str
    features: BlindingFeatures
    reasoning: List[str]
    methodological_implications: str
    bias_risk_assessment: str


class BlindingClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_blinding_patterns()
        self._setup_methodological_implications()
    
    def _setup_blinding_patterns(self):
        """Define patterns for blinding detection - HIGHEST ACCURACY"""
        
        # Proper blinding indicators (highest priority)
        self.proper_blinding_indicators = [
            # Double-blind patterns (priority 4)
            (r'\bdouble[-\s]blind(?:ed)?\b', 4, 'double_blind'),
            (r'\bdouble[-\s]masked\b', 4, 'double_masked'),
            (r'\bdouble[-\s]blind(?:ed)?\s+(?:placebo[-\s]controlled\s+)?trial\b', 4, 'double_blind_trial'),
            (r'\brandomized\s+double[-\s]blind(?:ed)?\b', 4, 'randomized_double_blind'),
            
            # Triple-blind patterns (priority 4)
            (r'\btriple[-\s]blind(?:ed)?\b', 4, 'triple_blind'),
            (r'\btriple[-\s]masked\b', 4, 'triple_masked'),
            
            # Comprehensive blinding descriptions (priority 3)
            (r'\b(?:participants?|patients?)\s+and\s+(?:investigators?|researchers?)\s+(?:were\s+)?blind(?:ed)?\b', 3, 'participants_investigators_blinded'),
            (r'\b(?:investigators?|researchers?)\s+and\s+(?:participants?|patients?)\s+(?:were\s+)?blind(?:ed)?\b', 3, 'investigators_participants_blinded'),
            (r'\ball\s+(?:study\s+)?personnel\s+(?:were\s+)?blind(?:ed)?\b', 2, 'all_personnel_blinded'),
            
            # Specific role blinding (priority 3)
            (r'\b(?:outcome\s+)?assessors?\s+(?:were\s+)?blind(?:ed)?\b', 3, 'assessors_blinded'),
            (r'\b(?:data\s+)?analysts?\s+(?:were\s+)?blind(?:ed)?\b', 2, 'analysts_blinded'),
            (r'\bstatisticians?\s+(?:were\s+)?blind(?:ed)?\b', 2, 'statisticians_blinded'),
        ]
        
        # Single blinding indicators (moderate priority)
        self.single_blinding_indicators = [
            # Single-blind patterns (priority 2)
            (r'\bsingle[-\s]blind(?:ed)?\b', 2, 'single_blind'),
            (r'\bsingle[-\s]masked\b', 2, 'single_masked'),
            (r'\b(?:participants?|patients?)\s+(?:were\s+)?blind(?:ed)?\b', 1, 'participants_blinded'),
            (r'\b(?:investigators?|researchers?)\s+(?:were\s+)?blind(?:ed)?\b', 1, 'investigators_blinded'),
            
            # Observer blinded (priority 2)
            (r'\bobserver[-\s]blind(?:ed)?\b', 2, 'observer_blinded'),
            (r'\bassessor[-\s]blind(?:ed)?\b', 2, 'assessor_blinded'),
            (r'\bevaluator[-\s]blind(?:ed)?\b', 2, 'evaluator_blinded'),
        ]
        
        # Lack of blinding indicators (high priority for detection)
        self.lack_of_blinding_indicators = [
            # Explicit no blinding (priority 4)
            (r'\bopen[-\s]label\b', 4, 'open_label'),
            (r'\bunblinded\b', 4, 'unblinded'),
            (r'\bnot\s+blind(?:ed)?\b', 4, 'not_blinded'),
            (r'\bno\s+blinding\b', 4, 'no_blinding'),
            (r'\bwithout\s+blinding\b', 3, 'without_blinding'),
            
            # Open design patterns (priority 3)
            (r'\bopen[-\s]label\s+(?:randomized\s+)?(?:controlled\s+)?trial\b', 4, 'open_label_trial'),
            (r'\bopen[-\s]design\b', 3, 'open_design'),
            (r'\bunblinded\s+(?:randomized\s+)?(?:controlled\s+)?trial\b', 4, 'unblinded_trial'),
            
            # Impossible to blind scenarios (priority 2)
            (r'\bimpossible\s+to\s+blind\b', 3, 'impossible_to_blind'),
            (r'\bcannot\s+be\s+blind(?:ed)?\b', 2, 'cannot_be_blinded'),
            (r'\bblinding\s+(?:was\s+)?not\s+(?:possible|feasible)\b', 3, 'blinding_not_possible'),
        ]
        
        # Blinding implementation patterns
        self.blinding_implementation_patterns = [
            # Placebo implementation (priority 3)
            (r'\bmatching\s+placebo\b', 3, 'matching_placebo'),
            (r'\bidentical[-\s]appearing\s+(?:placebo|tablets?|capsules?)\b', 3, 'identical_appearing'),
            (r'\bplacebo\s+(?:tablets?|capsules?|pills?)\s+(?:of\s+)?identical\s+appearance\b', 3, 'identical_placebo'),
            (r'\bindistinguishable\s+(?:from\s+)?(?:active\s+)?(?:treatment|drug|medication)\b', 2, 'indistinguishable'),
            
            # Sham procedures (priority 3)
            (r'\bsham\s+(?:procedure|operation|surgery|intervention)\b', 3, 'sham_procedure'),
            (r'\bsham[-\s]controlled\b', 3, 'sham_controlled'),
            (r'\bmock\s+(?:procedure|intervention)\b', 2, 'mock_procedure'),
            
            # Masking methods (priority 2)
            (r'\bmasked\s+(?:to\s+)?(?:treatment\s+)?(?:assignment|allocation)\b', 2, 'masked_assignment'),
            (r'\bconcealed\s+(?:treatment\s+)?allocation\b', 2, 'concealed_allocation'),
            (r'\bblinded\s+(?:to\s+)?(?:treatment\s+)?(?:assignment|group)\b', 2, 'blinded_assignment'),
        ]
        
        # Blinding assessment patterns
        self.blinding_assessment_patterns = [
            # Blinding success evaluation (priority 2)
            (r'\bblinding\s+(?:success|integrity)\s+(?:was\s+)?(?:assessed|evaluated|tested)\b', 2, 'blinding_assessed'),
            (r'\bassessment\s+of\s+blinding\s+(?:success|integrity)\b', 2, 'blinding_assessment'),
            (r'\bguessing\s+(?:of\s+)?treatment\s+allocation\b', 1, 'guessing_treatment'),
            
            # Emergency unblinding (priority 1)
            (r'\bemergency\s+unblinding\b', 1, 'emergency_unblinding'),
            (r'\bcode\s+break(?:ing)?\s+(?:procedures?|envelopes?)\b', 1, 'code_breaking'),
            (r'\bunblinding\s+procedures?\b', 1, 'unblinding_procedures'),
        ]
        
        # Study design context patterns
        self.design_context_patterns = [
            # Placebo-controlled context (priority 2)
            (r'\bplacebo[-\s]controlled\b', 2, 'placebo_controlled'),
            (r'\bactive[-\s]controlled\b', 2, 'active_controlled'),
            (r'\bcomparator[-\s]controlled\b', 1, 'comparator_controlled'),
            
            # Intervention types that affect blinding (priority 2)
            (r'\bsurgical\s+(?:intervention|procedure|trial)\b', 2, 'surgical_intervention'),
            (r'\bdevice\s+(?:intervention|trial|study)\b', 2, 'device_intervention'),
            (r'\bbehavioral\s+intervention\b', 1, 'behavioral_intervention'),
            (r'\bpsychological\s+intervention\b', 1, 'psychological_intervention'),
        ]
        
        # Outcome type patterns (affects bias risk)
        self.outcome_type_patterns = [
            # Subjective outcomes (higher bias risk without blinding)
            (r'\b(?:pain|quality\s+of\s+life|satisfaction|depression|anxiety)\s+(?:score|scale|assessment)\b', 2, 'subjective_outcome'),
            (r'\b(?:patient|physician)[-\s]reported\s+outcomes?\b|\bPRO\b', 2, 'patient_reported_outcome'),
            (r'\b(?:visual\s+analog\s+scale|VAS|likert\s+scale)\b', 2, 'subjective_scale'),
            
            # Objective outcomes (lower bias risk)
            (r'\b(?:mortality|death|survival)\b', 1, 'objective_mortality'),
            (r'\b(?:laboratory|biomarker|radiological)\s+(?:outcomes?|measures?|results?)\b', 1, 'objective_laboratory'),
            (r'\b(?:blood\s+pressure|heart\s+rate|temperature)\b', 1, 'objective_physiological'),
        ]
    
    def _setup_methodological_implications(self):
        """Define methodological implications for each classification"""
        self.methodological_implications = {
            BlindingCode.PROPERLY_BLINDED:
                "Proper blinding (double/triple-blind) prevents performance and detection bias by "
                "ensuring participants, investigators, and outcome assessors remain unaware of "
                "treatment allocation. This maintains treatment fidelity and objective outcome assessment.",
                
            BlindingCode.NOT_BLINDED:
                "The lack of blinding introduces the potential for time-varying confounding and "
                "outcome adjudication bias. Without blinding, participants and investigators may "
                "alter behavior based on treatment knowledge, affecting both treatment delivery and outcome assessment.",
                
            BlindingCode.NO_MENTION_OF_BLINDING:
                "The abstract doesn't mention blinding. The lack of blinding introduces the potential "
                "for time-varying confounding and outcome adjudication bias. Proper blinding is essential "
                "for minimizing performance and detection bias in clinical trials.",
                
            BlindingCode.BLINDING_UNCLEAR:
                "Blinding status is unclear or inadequately described, making it impossible to assess "
                "the risk of performance and detection bias. Clear description of blinding procedures "
                "is essential for evaluating trial quality and interpreting results."
        }
    
    def check_blinding(self, text: str) -> BlindingResult:
        """
        Analyze trial design to detect blinding adequacy
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            BlindingResult with blinding assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                BlindingCode.BLINDING_UNCLEAR,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess blinding status"
            )
        
        # Extract comprehensive blinding features
        features = self._extract_blinding_features(text, reasoning)
        
        # Analyze blinding quality and implementation
        blinding_analysis = self._analyze_blinding_quality(features, reasoning)
        
        # Assess bias risk from blinding status
        bias_risk_assessment = self._assess_bias_risk(features, reasoning)
        
        # Make blinding classification
        return self._make_blinding_classification(features, reasoning, bias_risk_assessment)
    
    def _extract_blinding_features(self, text: str, reasoning: List[str]) -> BlindingFeatures:
        """Extract comprehensive blinding features with HIGHEST ACCURACY"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Extract proper blinding indicators
        blinding_mentioned = False
        proper_blinding_mentioned = False
        blinding_phrases = []
        blinding_level = "none"
        
        # Check for proper blinding (double/triple)
        double_blind_mentioned = False
        triple_blind_mentioned = False
        
        for pattern, priority, context in self.proper_blinding_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                blinding_mentioned = True
                blinding_phrases.append(match.group(0))
                if 'double' in context:
                    double_blind_mentioned = True
                    proper_blinding_mentioned = True
                    blinding_level = "double"
                elif 'triple' in context:
                    triple_blind_mentioned = True
                    proper_blinding_mentioned = True
                    blinding_level = "triple"
        
        # Check for single blinding
        single_blind_mentioned = False
        for pattern, priority, context in self.single_blinding_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                blinding_mentioned = True
                blinding_phrases.append(match.group(0))
                if 'single' in context and blinding_level == "none":
                    single_blind_mentioned = True
                    blinding_level = "single"
        
        # Check for lack of blinding
        open_label_mentioned = False
        unblinded_mentioned = False
        not_blinded_mentioned = False
        no_blinding_mentioned = False
        
        for pattern, priority, context in self.lack_of_blinding_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                if 'open_label' in context:
                    open_label_mentioned = True
                    blinding_level = "open"
                elif 'unblinded' in context:
                    unblinded_mentioned = True
                    blinding_level = "open"
                elif 'not_blinded' in context:
                    not_blinded_mentioned = True
                    blinding_level = "open"
                elif 'no_blinding' in context:
                    no_blinding_mentioned = True
                    blinding_level = "open"
        
        # Check for masked terminology
        masked_mentioned = bool(re.search(r'\bmasked\b', text, re.IGNORECASE))
        if masked_mentioned and not blinding_mentioned:
            blinding_mentioned = True
            blinding_phrases.append("masked")
        
        # Analyze specific role blinding
        participant_blinding = bool(re.search(
            r'\b(?:participants?|patients?)\s+(?:were\s+)?blind(?:ed)?\b', text, re.IGNORECASE
        ))
        investigator_blinding = bool(re.search(
            r'\b(?:investigators?|researchers?|physicians?)\s+(?:were\s+)?blind(?:ed)?\b', text, re.IGNORECASE
        ))
        outcome_assessor_blinding = bool(re.search(
            r'\b(?:outcome\s+)?assessors?\s+(?:were\s+)?blind(?:ed)?\b', text, re.IGNORECASE
        ))
        statistician_blinding = bool(re.search(
            r'\b(?:data\s+)?(?:analysts?|statisticians?)\s+(?:were\s+)?blind(?:ed)?\b', text, re.IGNORECASE
        ))
        
        # Check blinding implementation details
        matching_placebo_mentioned = False
        identical_appearance_mentioned = False
        sham_procedure_mentioned = False
        blinding_method_described = False
        
        for pattern, priority, context in self.blinding_implementation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                blinding_method_described = True
                if 'placebo' in context:
                    matching_placebo_mentioned = True
                elif 'identical' in context or 'indistinguishable' in context:
                    identical_appearance_mentioned = True
                elif 'sham' in context or 'mock' in context:
                    sham_procedure_mentioned = True
        
        # Check study design context
        placebo_controlled = bool(re.search(r'\bplacebo[-\s]controlled\b', text, re.IGNORECASE))
        active_controlled = bool(re.search(r'\bactive[-\s]controlled\b', text, re.IGNORECASE))
        
        surgical_intervention = bool(re.search(
            r'\bsurgical\s+(?:intervention|procedure|trial)\b', text, re.IGNORECASE
        ))
        device_intervention = bool(re.search(
            r'\bdevice\s+(?:intervention|trial|study)\b', text, re.IGNORECASE
        ))
        
        # Check blinding assessment
        blinding_success_assessed = False
        blinding_integrity_mentioned = False
        emergency_unblinding_mentioned = False
        
        for pattern, priority, context in self.blinding_assessment_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if 'assessed' in context or 'assessment' in context:
                    blinding_success_assessed = True
                    blinding_integrity_mentioned = True
                elif 'emergency' in context or 'code' in context:
                    emergency_unblinding_mentioned = True
        
        # Analyze outcome types for bias risk
        subjective_outcomes = False
        objective_outcomes_only = True
        
        subjective_count = 0
        objective_count = 0
        
        for pattern, priority, context in self.outcome_type_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if 'subjective' in context:
                subjective_count += matches
            elif 'objective' in context:
                objective_count += matches
        
        if subjective_count > 0:
            subjective_outcomes = True
            objective_outcomes_only = False
        
        # Assess bias risk
        performance_bias_risk = not proper_blinding_mentioned and not single_blind_mentioned
        detection_bias_risk = not outcome_assessor_blinding and not proper_blinding_mentioned
        
        # Check for methods section
        methods_section_present = bool(re.search(
            r'\b(?:methods|design|study\s+design|procedures?)\b', text, re.IGNORECASE
        ))
        
        # Calculate design description quality
        design_description_quality = self._calculate_design_quality(
            blinding_mentioned, blinding_method_described, methods_section_present
        )
        
        # Calculate blinding specificity score
        blinding_specificity_score = self._calculate_blinding_specificity(
            blinding_phrases, participant_blinding, investigator_blinding, 
            outcome_assessor_blinding, blinding_method_described
        )
        
        # Get detected patterns for transparency
        detected_patterns = self._get_detected_patterns(
            text, blinding_phrases, blinding_level, open_label_mentioned
        )
        
        # Create extracted blinding info
        extracted_blinding_info = []
        if proper_blinding_mentioned:
            extracted_blinding_info.append(f"Proper blinding: {blinding_level}")
        if open_label_mentioned:
            extracted_blinding_info.append("Open-label design")
        if unblinded_mentioned:
            extracted_blinding_info.append("Unblinded design")
        if single_blind_mentioned and not proper_blinding_mentioned:
            extracted_blinding_info.append("Single-blind design")
        if matching_placebo_mentioned:
            extracted_blinding_info.append("Matching placebo")
        if sham_procedure_mentioned:
            extracted_blinding_info.append("Sham procedure")
        if not blinding_mentioned:
            extracted_blinding_info.append("No blinding mentioned")
        
        # Add to reasoning
        if proper_blinding_mentioned:
            reasoning.append(f"Proper blinding detected: {blinding_level}-blind")
        elif open_label_mentioned or unblinded_mentioned:
            reasoning.append("Open-label/unblinded design detected")
        elif single_blind_mentioned:
            reasoning.append("Single-blind design detected")
        elif not blinding_mentioned:
            reasoning.append("No mention of blinding found")
        
        return BlindingFeatures(
            blinding_mentioned=blinding_mentioned,
            blinding_level=blinding_level,
            blinding_phrases=blinding_phrases,
            proper_blinding_mentioned=proper_blinding_mentioned,
            double_blind_mentioned=double_blind_mentioned,
            triple_blind_mentioned=triple_blind_mentioned,
            single_blind_mentioned=single_blind_mentioned,
            masked_mentioned=masked_mentioned,
            open_label_mentioned=open_label_mentioned,
            unblinded_mentioned=unblinded_mentioned,
            not_blinded_mentioned=not_blinded_mentioned,
            no_blinding_mentioned=no_blinding_mentioned,
            participant_blinding=participant_blinding,
            investigator_blinding=investigator_blinding,
            outcome_assessor_blinding=outcome_assessor_blinding,
            statistician_blinding=statistician_blinding,
            matching_placebo_mentioned=matching_placebo_mentioned,
            identical_appearance_mentioned=identical_appearance_mentioned,
            sham_procedure_mentioned=sham_procedure_mentioned,
            blinding_method_described=blinding_method_described,
            placebo_controlled=placebo_controlled,
            active_controlled=active_controlled,
            surgical_intervention=surgical_intervention,
            device_intervention=device_intervention,
            blinding_success_assessed=blinding_success_assessed,
            blinding_integrity_mentioned=blinding_integrity_mentioned,
            emergency_unblinding_mentioned=emergency_unblinding_mentioned,
            performance_bias_risk=performance_bias_risk,
            detection_bias_risk=detection_bias_risk,
            subjective_outcomes=subjective_outcomes,
            objective_outcomes_only=objective_outcomes_only,
            methods_section_present=methods_section_present,
            design_description_quality=design_description_quality,
            blinding_specificity_score=blinding_specificity_score,
            detected_patterns=detected_patterns,
            extracted_blinding_info=extracted_blinding_info
        )
    
    def _calculate_design_quality(self, blinding_mentioned: bool, 
                                 method_described: bool, methods_present: bool) -> float:
        """Calculate overall design description quality"""
        score = 0.0
        
        if blinding_mentioned:
            score += 0.4
        if method_described:
            score += 0.3
        if methods_present:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_blinding_specificity(self, blinding_phrases: List[str], 
                                      participant_blinding: bool, investigator_blinding: bool,
                                      assessor_blinding: bool, method_described: bool) -> float:
        """Calculate how specifically blinding is described"""
        score = 0.0
        
        # Base score for blinding mentions
        score += min(len(blinding_phrases) * 0.1, 0.3)
        
        # Score for specific role blinding
        if participant_blinding:
            score += 0.2
        if investigator_blinding:
            score += 0.2
        if assessor_blinding:
            score += 0.2
        
        # Score for method description
        if method_described:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_blinding_quality(self, features: BlindingFeatures, 
                                 reasoning: List[str]) -> str:
        """Analyze blinding quality and implementation"""
        
        if features.proper_blinding_mentioned and features.blinding_method_described:
            analysis = f"HIGH quality blinding ({features.blinding_level}-blind with implementation details)"
        elif features.proper_blinding_mentioned:
            analysis = f"GOOD quality blinding ({features.blinding_level}-blind design)"
        elif features.single_blind_mentioned:
            analysis = "MODERATE quality blinding (single-blind design)"
        elif features.open_label_mentioned or features.unblinded_mentioned:
            analysis = "NO blinding (open-label/unblinded design)"
        elif not features.blinding_mentioned:
            analysis = "UNCLEAR blinding status (not mentioned)"
        else:
            analysis = "UNCLEAR blinding quality"
        
        reasoning.append(f"Blinding quality: {analysis}")
        return analysis
    
    def _assess_bias_risk(self, features: BlindingFeatures, 
                         reasoning: List[str]) -> str:
        """Assess risk of bias from blinding status"""
        
        bias_risks = []
        
        # Performance bias risk
        if features.performance_bias_risk:
            bias_risks.append("High performance bias risk (participants/investigators not blinded)")
        
        # Detection bias risk
        if features.detection_bias_risk:
            bias_risks.append("High detection bias risk (outcome assessors not blinded)")
        
        # Specific context risks
        if features.subjective_outcomes and not features.proper_blinding_mentioned:
            bias_risks.append("Subjective outcomes without proper blinding increase bias risk")
        
        if features.open_label_mentioned or features.unblinded_mentioned:
            bias_risks.append("Open-label design introduces time-varying confounding")
        
        if not features.blinding_mentioned:
            bias_risks.append("Unknown blinding status prevents bias assessment")
        
        # Overall assessment
        if features.proper_blinding_mentioned and not bias_risks:
            assessment = "LOW bias risk: proper blinding implemented"
        elif features.single_blind_mentioned and not features.subjective_outcomes:
            assessment = "MODERATE bias risk: single-blind with objective outcomes"
        elif bias_risks:
            assessment = f"HIGH bias risk: {'; '.join(bias_risks[:2])}"
        else:
            assessment = "MODERATE bias risk: some methodological limitations"
        
        reasoning.append(f"Bias risk assessment: {assessment}")
        return assessment
    
    def _get_detected_patterns(self, text: str, blinding_phrases: List[str], 
                              blinding_level: str, open_label: bool) -> List[str]:
        """Get list of detected patterns for transparency"""
        detected = []
        
        # Add blinding patterns
        for phrase in set(blinding_phrases[:3]):  # Top unique phrases
            detected.append(f"blinding: {phrase}")
        
        # Add blinding level
        if blinding_level != "none":
            detected.append(f"level: {blinding_level}")
        
        # Add specific patterns
        if open_label:
            detected.append("design: open_label")
        if re.search(r'\bdouble[-\s]blind\b', text, re.IGNORECASE):
            detected.append("design: double_blind")
        if re.search(r'\bplacebo[-\s]controlled\b', text, re.IGNORECASE):
            detected.append("control: placebo_controlled")
        if re.search(r'\bmatching\s+placebo\b', text, re.IGNORECASE):
            detected.append("implementation: matching_placebo")
        
        return detected[:6]  # Limit for readability
    
    def _make_blinding_classification(self, features: BlindingFeatures, 
                                    reasoning: List[str], 
                                    bias_risk_assessment: str) -> BlindingResult:
        """Make blinding classification based on methodological heuristic"""
        
        # BLINDING CLASSIFICATION LOGIC
        
        # 1. Proper blinding detected (double/triple-blind)
        if features.proper_blinding_mentioned:
            reasoning.append(f"Proper blinding detected: {features.blinding_level}-blind design")
            
            # Calculate confidence based on implementation details
            confidence = 0.9
            if features.blinding_method_described:
                confidence = 0.95
            elif features.placebo_controlled:
                confidence = 0.92
            
            return self._create_result(
                BlindingCode.PROPERLY_BLINDED,
                confidence,
                f"Properly blinded trial ({features.blinding_level}-blind design)",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 2. Explicit lack of blinding (open-label, unblinded)
        elif (features.open_label_mentioned or features.unblinded_mentioned or 
              features.not_blinded_mentioned or features.no_blinding_mentioned):
            reasoning.append("Explicit lack of blinding detected")
            
            # High confidence for explicit mentions
            confidence = 0.9
            if features.open_label_mentioned:
                confidence = 0.95
            
            return self._create_result(
                BlindingCode.NOT_BLINDED,
                confidence,
                "Not blinded - open-label/unblinded design explicitly mentioned",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 3. Single-blind design (partial blinding)
        elif features.single_blind_mentioned:
            reasoning.append("Single-blind design detected - partial blinding only")
            
            # Single-blind is better than no blinding but not optimal
            if features.subjective_outcomes:
                reasoning.append("Single-blind with subjective outcomes increases bias risk")
                return self._create_result(
                    BlindingCode.NOT_BLINDED,
                    0.7,
                    "Single-blind design with subjective outcomes - inadequate blinding",
                    features,
                    reasoning,
                    bias_risk_assessment
                )
            else:
                return self._create_result(
                    BlindingCode.PROPERLY_BLINDED,
                    0.6,
                    "Single-blind design - partial blinding implemented",
                    features,
                    reasoning,
                    bias_risk_assessment
                )
        
        # 4. Placebo-controlled but blinding not explicitly mentioned
        elif features.placebo_controlled and not features.blinding_mentioned:
            reasoning.append("Placebo-controlled trial suggests blinding but not explicitly stated")
            
            return self._create_result(
                BlindingCode.NO_MENTION_OF_BLINDING,
                0.7,
                "Placebo-controlled trial but blinding not explicitly mentioned",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 5. Some blinding indicators but unclear level
        elif features.blinding_mentioned:
            reasoning.append("Blinding mentioned but level/implementation unclear")
            
            # Check if context suggests proper blinding
            if (features.participant_blinding and features.investigator_blinding and 
                features.outcome_assessor_blinding):
                reasoning.append("Multiple roles blinded suggests comprehensive blinding")
                return self._create_result(
                    BlindingCode.PROPERLY_BLINDED,
                    0.7,
                    "Comprehensive blinding of multiple roles - likely adequate",
                    features,
                    reasoning,
                    bias_risk_assessment
                )
            else:
                return self._create_result(
                    BlindingCode.BLINDING_UNCLEAR,
                    0.6,
                    "Blinding mentioned but implementation unclear",
                    features,
                    reasoning,
                    bias_risk_assessment
                )
        
        # 6. Surgical or device intervention (often impossible to blind)
        elif features.surgical_intervention or features.device_intervention:
            reasoning.append("Surgical/device intervention - blinding often not feasible")
            
            return self._create_result(
                BlindingCode.NOT_BLINDED,
                0.8,
                "Surgical/device intervention - blinding likely not implemented",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 7. Active-controlled trial without blinding mention
        elif features.active_controlled and not features.blinding_mentioned:
            reasoning.append("Active-controlled trial without blinding mention")
            
            return self._create_result(
                BlindingCode.NO_MENTION_OF_BLINDING,
                0.8,
                "Active-controlled trial - blinding status not mentioned",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 8. Default: No mention of blinding
        else:
            reasoning.append("No mention of blinding found in abstract")
            
            return self._create_result(
                BlindingCode.NO_MENTION_OF_BLINDING,
                0.85,
                "No mention of blinding - potential methodological limitation",
                features,
                reasoning,
                bias_risk_assessment
            )
    
    def _create_result(self, code: BlindingCode, confidence: float, 
                      message: str, features: BlindingFeatures, 
                      reasoning: List[str], bias_risk_assessment: str) -> BlindingResult:
        """Create a BlindingResult object"""
        return BlindingResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            methodological_implications=self.methodological_implications[code],
            bias_risk_assessment=bias_risk_assessment
        )
    
    def _create_empty_features(self) -> BlindingFeatures:
        """Create empty features object for error cases"""
        return BlindingFeatures(
            blinding_mentioned=False, blinding_level="none", blinding_phrases=[],
            proper_blinding_mentioned=False, double_blind_mentioned=False, triple_blind_mentioned=False,
            single_blind_mentioned=False, masked_mentioned=False, open_label_mentioned=False,
            unblinded_mentioned=False, not_blinded_mentioned=False, no_blinding_mentioned=False,
            participant_blinding=False, investigator_blinding=False, outcome_assessor_blinding=False,
            statistician_blinding=False, matching_placebo_mentioned=False, identical_appearance_mentioned=False,
            sham_procedure_mentioned=False, blinding_method_described=False, placebo_controlled=False,
            active_controlled=False, surgical_intervention=False, device_intervention=False,
            blinding_success_assessed=False, blinding_integrity_mentioned=False, emergency_unblinding_mentioned=False,
            performance_bias_risk=True, detection_bias_risk=True, subjective_outcomes=False,
            objective_outcomes_only=False, methods_section_present=False, design_description_quality=0,
            blinding_specificity_score=0, detected_patterns=[], extracted_blinding_info=[]
        )





def run_check(abstract : str):# just a wrapper method
    classifier = BlindingClassifier()
    result = classifier.check_blinding(abstract)
    return result
