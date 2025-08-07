import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class PlaceboControlCode(Enum):
    """Placebo control classification codes"""
    PLACEBO_CONTROLLED = 0           # Placebo control mentioned - Good methodological design
    DRUG_VS_STANDARD_CARE = 1       # Drug vs standard care - METHODOLOGICAL ISSUE (lack of placebo)
    NO_PLACEBO_MENTIONED = 2        # No mention of placebo - METHODOLOGICAL ISSUE (potential lack of placebo)
    CONTROL_UNCLEAR = 3             # Control group not clearly defined


@dataclass
class PlaceboControlFeatures:
    """Container for placebo control features"""
    # Placebo detection
    placebo_mentioned: bool                    # Placebo explicitly mentioned
    placebo_controlled_mentioned: bool         # "Placebo-controlled" explicitly stated
    double_blind_mentioned: bool              # Double-blind mentioned (implies placebo)
    placebo_phrases: List[str]                # Specific placebo phrases found
    
    # Control group identification
    control_group_mentioned: bool             # Control group explicitly mentioned
    control_group_description: List[str]      # Description of control group
    standard_care_control: bool               # Standard care as control
    active_control_mentioned: bool            # Active comparator control
    
    # Study design indicators
    randomized_mentioned: bool                # Randomized trial mentioned
    blinding_level: str                       # Level of blinding (none, single, double, triple)
    trial_design_described: bool              # Trial design explicitly described
    
    # Comparison patterns
    drug_vs_standard_care: bool               # Drug A vs standard care pattern
    drug_vs_placebo: bool                     # Drug vs placebo pattern
    drug_vs_active_control: bool              # Drug vs active drug pattern
    no_clear_comparison: bool                 # No clear comparison described
    
    # Intervention description
    intervention_described: bool              # Intervention clearly described
    comparator_described: bool                # Comparator clearly described
    treatment_arms_count: int                 # Number of treatment arms identified
    
    # Methodological quality indicators
    allocation_concealment_mentioned: bool    # Allocation concealment mentioned
    intention_to_treat_mentioned: bool        # ITT analysis mentioned
    per_protocol_mentioned: bool              # Per-protocol analysis mentioned
    
    # Bias risk indicators
    open_label_mentioned: bool                # Open-label design mentioned
    single_blind_mentioned: bool              # Single-blind mentioned
    unblinded_mentioned: bool                 # Unblinded mentioned
    
    # Text characteristics
    methods_section_present: bool             # Methods section identified
    design_specificity_score: float          # How specifically design is described
    comparison_clarity_score: float          # How clearly comparison is described
    
    # Detected patterns for transparency
    detected_patterns: List[str]
    extracted_design_info: List[str]


@dataclass
class PlaceboControlResult:
    code: PlaceboControlCode
    confidence: float
    message: str
    features: PlaceboControlFeatures
    reasoning: List[str]
    methodological_implications: str
    bias_risk_assessment: str


class PlaceboControlClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_placebo_patterns()
        self._setup_methodological_implications()
    
    def _setup_placebo_patterns(self):
        """Define patterns for placebo control detection - HIGHEST ACCURACY"""
        
        # Placebo indicators (highest priority)
        self.placebo_indicators = [
            # Explicit placebo mentions (priority 4)
            (r'\bplacebo[-\s]controlled\b', 4, 'placebo_controlled'),
            (r'\brandomized\s+placebo[-\s]controlled\b', 4, 'randomized_placebo_controlled'),
            (r'\bdouble[-\s]blind\s+placebo[-\s]controlled\b', 4, 'double_blind_placebo'),
            (r'\bplacebo[-\s]controlled\s+trial\b', 4, 'placebo_controlled_trial'),
            
            # Placebo mentions in comparison (priority 3)
            (r'\bversus\s+placebo\b|\bvs\.?\s+placebo\b', 3, 'vs_placebo'),
            (r'\bcompared\s+(?:to\s+|with\s+)?placebo\b', 3, 'compared_to_placebo'),
            (r'\b(?:drug|treatment|intervention)\s+(?:vs\.?|versus)\s+placebo\b', 3, 'drug_vs_placebo'),
            (r'\bplacebo\s+(?:group|arm|control)\b', 3, 'placebo_group'),
            
            # General placebo mentions (priority 2)
            (r'\bplacebo\b(?!\s*effect)', 2, 'placebo_mentioned'),
            (r'\b(?:matching\s+)?placebo\s+(?:tablets?|capsules?|pills?)\b', 3, 'placebo_medication'),
            (r'\bidentical[-\s]appearing\s+placebo\b', 3, 'identical_placebo'),
            
            # Blinding indicators that suggest placebo (priority 2)
            (r'\bdouble[-\s]blind(?:ed)?\b', 2, 'double_blind'),
            (r'\btriple[-\s]blind(?:ed)?\b', 3, 'triple_blind'),
            (r'\bblind(?:ed)?\s+(?:to\s+)?(?:treatment\s+)?assignment\b', 2, 'treatment_blinding'),
        ]
        
        # Standard care control indicators
        self.standard_care_indicators = [
            # Explicit standard care mentions (priority 4)
            (r'\b(?:versus\s+|vs\.?\s+|compared\s+(?:to\s+|with\s+)?)standard\s+(?:of\s+)?care\b', 4, 'vs_standard_care'),
            (r'\b(?:drug|treatment|intervention)\s+(?:vs\.?|versus)\s+standard\s+(?:of\s+)?care\b', 4, 'drug_vs_standard_care'),
            (r'\bstandard\s+(?:of\s+)?care\s+(?:group|arm|control)\b', 3, 'standard_care_group'),
            
            # Usual care patterns (priority 3)
            (r'\b(?:versus\s+|vs\.?\s+|compared\s+(?:to\s+|with\s+)?)usual\s+care\b', 3, 'vs_usual_care'),
            (r'\busual\s+care\s+(?:group|arm|control)\b', 3, 'usual_care_group'),
            (r'\broutine\s+care\s+(?:group|arm|control)\b', 2, 'routine_care_group'),
            
            # Best supportive care patterns (priority 3)
            (r'\bbest\s+supportive\s+care\b|\bBSC\b', 3, 'best_supportive_care'),
            (r'\b(?:versus\s+|vs\.?\s+)(?:best\s+)?supportive\s+care\b', 3, 'vs_supportive_care'),
            
            # Conservative management patterns (priority 2)
            (r'\bconservative\s+(?:management|treatment)\b', 2, 'conservative_management'),
            (r'\b(?:versus\s+|vs\.?\s+)conservative\s+(?:management|treatment)\b', 3, 'vs_conservative'),
            
            # No treatment patterns (priority 3)
            (r'\b(?:versus\s+|vs\.?\s+|compared\s+(?:to\s+|with\s+)?)no\s+(?:treatment|intervention)\b', 3, 'vs_no_treatment'),
            (r'\bno\s+(?:treatment|intervention)\s+(?:group|arm|control)\b', 3, 'no_treatment_group'),
        ]
        
        # Active control indicators
        self.active_control_indicators = [
            # Active comparator patterns (priority 3)
            (r'\b(?:versus\s+|vs\.?\s+|compared\s+(?:to\s+|with\s+)?)(?:active\s+)?(?:control|comparator)\b', 2, 'active_control'),
            (r'\bactive[-\s]controlled\s+trial\b', 3, 'active_controlled_trial'),
            (r'\bhead[-\s]to[-\s]head\s+comparison\b', 3, 'head_to_head'),
            
            # Specific drug comparisons (priority 2)
            (r'\b(?:versus\s+|vs\.?\s+)\w+(?:mycin|cillin|statin|prazole|olol|pine|sartan)\b', 2, 'vs_specific_drug'),
            (r'\bcompared\s+(?:to\s+|with\s+)\w+(?:mycin|cillin|statin|prazole|olol|pine|sartan)\b', 2, 'compared_to_drug'),
        ]
        
        # Control group description patterns
        self.control_description_patterns = [
            # Control group mentions (priority 2)
            (r'\bcontrol\s+(?:group|arm)\b', 2, 'control_group'),
            (r'\b(?:treatment\s+)?arms?\b', 1, 'treatment_arms'),
            (r'\b(?:study\s+)?groups?\b', 1, 'study_groups'),
            (r'\brandomized\s+to\s+(?:receive\s+)?([^.]+?)(?:\.|;|or)', 2, 'randomized_to'),
            
            # Comparison structure (priority 2)
            (r'\b(?:two|three|four)\s+(?:treatment\s+)?(?:groups?|arms?)\b', 2, 'numbered_groups'),
            (r'\b(?:intervention|experimental)\s+(?:group|arm)\b', 2, 'intervention_group'),
            (r'\b(?:comparator|control)\s+(?:group|arm)\b', 2, 'comparator_group'),
        ]
        
        # Blinding and design patterns
        self.blinding_patterns = [
            # Blinding levels (priority varies)
            (r'\bopen[-\s]label\b', 3, 'open_label'),
            (r'\bunblinded\b', 3, 'unblinded'),
            (r'\bsingle[-\s]blind(?:ed)?\b', 2, 'single_blind'),
            (r'\bdouble[-\s]blind(?:ed)?\b', 2, 'double_blind'),
            (r'\btriple[-\s]blind(?:ed)?\b', 3, 'triple_blind'),
            
            # Study design mentions (priority 2)
            (r'\brandomized\s+controlled\s+trial\b|\bRCT\b', 2, 'randomized_controlled_trial'),
            (r'\bprospective\s+(?:randomized\s+)?trial\b', 2, 'prospective_trial'),
            (r'\bmulti[-\s]center\s+trial\b', 1, 'multicenter_trial'),
            
            # Methodological quality indicators (priority 2)
            (r'\ballocation\s+concealment\b', 2, 'allocation_concealment'),
            (r'\bintention[-\s]to[-\s]treat\b|\bITT\b', 2, 'intention_to_treat'),
            (r'\bper[-\s]protocol\s+analysis\b', 2, 'per_protocol'),
        ]
        
        # Drug vs standard care specific patterns
        self.drug_vs_standard_patterns = [
            # Direct drug vs standard care patterns (priority 4)
            (r'\b\w+(?:mab|nib|tide|stat|pril|sartan|mycin|cillin)\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 4, 'specific_drug_vs_standard'),
            (r'\b(?:drug\s+)?[A-Z]\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 3, 'drug_A_vs_standard'),
            (r'\b(?:experimental\s+)?(?:drug|agent|treatment)\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 3, 'experimental_vs_standard'),
            
            # Novel therapy vs standard patterns (priority 3)
            (r'\bnovel\s+(?:therapy|treatment|drug)\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 3, 'novel_vs_standard'),
            (r'\bnew\s+(?:therapy|treatment|drug)\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 2, 'new_vs_standard'),
            
            # Intervention vs standard patterns (priority 2)
            (r'\bintervention\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 2, 'intervention_vs_standard'),
            (r'\btreatment\s+(?:versus\s+|vs\.?\s+)standard\s+(?:of\s+)?care\b', 2, 'treatment_vs_standard'),
        ]
    
    def _setup_methodological_implications(self):
        """Define methodological implications for each classification"""
        self.methodological_implications = {
            PlaceboControlCode.PLACEBO_CONTROLLED:
                "Placebo-controlled design provides optimal control for non-specific treatment effects, "
                "patient and investigator expectations, and natural disease progression. This design "
                "minimizes bias and allows for causal inference about treatment efficacy.",
                
            PlaceboControlCode.DRUG_VS_STANDARD_CARE:
                "Comparing drug to standard care without placebo control introduces potential for "
                "time-varying confounding and outcome adjudication bias. Patients and investigators "
                "are aware of treatment assignment, which can influence outcome assessment and care patterns.",
                
            PlaceboControlCode.NO_PLACEBO_MENTIONED:
                "No mention of placebo control suggests potential methodological limitations. If the trial "
                "lacked placebo control, this introduces risk for time-varying confounding, outcome "
                "adjudication bias, and difficulty in distinguishing specific from non-specific treatment effects.",
                
            PlaceboControlCode.CONTROL_UNCLEAR:
                "Control group not clearly defined, making it impossible to assess the adequacy of "
                "the comparison and potential for bias. Clear control group definition is essential "
                "for interpreting trial results and assessing methodological quality."
        }
    
    def check_placebo_control(self, text: str) -> PlaceboControlResult:
        """
        Analyze trial design to detect placebo control adequacy
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            PlaceboControlResult with placebo control assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                PlaceboControlCode.CONTROL_UNCLEAR,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess control group design"
            )
        
        # Extract comprehensive placebo control features
        features = self._extract_placebo_features(text, reasoning)
        
        # Analyze blinding and design quality
        design_analysis = self._analyze_design_quality(features, reasoning)
        
        # Assess bias risk
        bias_risk_assessment = self._assess_bias_risk(features, reasoning)
        
        # Make placebo control classification
        return self._make_placebo_classification(features, reasoning, bias_risk_assessment)
    
    def _extract_placebo_features(self, text: str, reasoning: List[str]) -> PlaceboControlFeatures:
        """Extract comprehensive placebo control features with HIGHEST ACCURACY"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Extract placebo indicators
        placebo_mentioned = False
        placebo_controlled_mentioned = False
        placebo_phrases = []
        
        for pattern, priority, context in self.placebo_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                placebo_mentioned = True
                placebo_phrases.append(match.group(0))
                if 'controlled' in context:
                    placebo_controlled_mentioned = True
        
        # Detect double-blind (strong indicator of placebo)
        double_blind_mentioned = bool(re.search(r'\bdouble[-\s]blind(?:ed)?\b', text, re.IGNORECASE))
        
        # Extract control group information
        control_group_mentioned = False
        control_group_description = []
        standard_care_control = False
        active_control_mentioned = False
        
        # Check for standard care control
        for pattern, priority, context in self.standard_care_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                control_group_mentioned = True
                standard_care_control = True
                control_group_description.append(match.group(0))
        
        # Check for active control
        for pattern, priority, context in self.active_control_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                control_group_mentioned = True
                active_control_mentioned = True
                control_group_description.append(match.group(0))
        
        # Check for general control group mentions
        for pattern, priority, context in self.control_description_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                control_group_mentioned = True
        
        # Analyze study design
        randomized_mentioned = bool(re.search(r'\brandomized\b', text, re.IGNORECASE))
        trial_design_described = bool(re.search(
            r'\b(?:design|study|trial|methods)\b.*?\b(?:randomized|controlled|blind)\b', 
            text, re.IGNORECASE
        ))
        
        # Determine blinding level
        blinding_level = self._determine_blinding_level(text)
        
        # Analyze comparison patterns
        drug_vs_standard_care = False
        drug_vs_placebo = False
        drug_vs_active_control = False
        no_clear_comparison = True
        
        # Check for drug vs standard care pattern
        for pattern, priority, context in self.drug_vs_standard_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                drug_vs_standard_care = True
                no_clear_comparison = False
                break
        
        # Check for drug vs placebo pattern
        if re.search(r'\b(?:drug|treatment|intervention)\s+(?:vs\.?|versus)\s+placebo\b', text, re.IGNORECASE):
            drug_vs_placebo = True
            no_clear_comparison = False
        
        # Check for drug vs active control pattern
        if active_control_mentioned and not standard_care_control:
            drug_vs_active_control = True
            no_clear_comparison = False
        
        # Extract intervention description
        intervention_described = bool(re.search(
            r'\b(?:intervention|treatment|drug|therapy)\b.*?\b(?:administered|given|received)\b',
            text, re.IGNORECASE
        ))
        
        comparator_described = bool(re.search(
            r'\b(?:control|comparator|placebo|standard\s+care)\b.*?\b(?:group|arm)\b',
            text, re.IGNORECASE
        ))
        
        # Count treatment arms
        treatment_arms_count = self._count_treatment_arms(text)
        
        # Check methodological quality indicators
        allocation_concealment_mentioned = bool(re.search(
            r'\ballocation\s+concealment\b', text, re.IGNORECASE
        ))
        intention_to_treat_mentioned = bool(re.search(
            r'\bintention[-\s]to[-\s]treat\b|\bITT\b', text, re.IGNORECASE
        ))
        per_protocol_mentioned = bool(re.search(
            r'\bper[-\s]protocol\s+analysis\b', text, re.IGNORECASE
        ))
        
        # Check bias risk indicators
        open_label_mentioned = bool(re.search(r'\bopen[-\s]label\b', text, re.IGNORECASE))
        single_blind_mentioned = bool(re.search(r'\bsingle[-\s]blind(?:ed)?\b', text, re.IGNORECASE))
        unblinded_mentioned = bool(re.search(r'\bunblinded\b', text, re.IGNORECASE))
        
        # Check for methods section
        methods_section_present = bool(re.search(
            r'\b(?:methods|design|study\s+design)\b', text, re.IGNORECASE
        ))
        
        # Calculate design specificity score
        design_specificity_score = self._calculate_design_specificity(text)
        
        # Calculate comparison clarity score
        comparison_clarity_score = self._calculate_comparison_clarity(
            drug_vs_placebo, drug_vs_standard_care, drug_vs_active_control, control_group_description
        )
        
        # Get detected patterns for transparency
        detected_patterns = self._get_detected_patterns(text, placebo_phrases, control_group_description)
        
        # Create extracted design info
        extracted_design_info = []
        if placebo_mentioned:
            extracted_design_info.append("Placebo mentioned")
        if placebo_controlled_mentioned:
            extracted_design_info.append("Placebo-controlled design")
        if standard_care_control:
            extracted_design_info.append("Standard care control")
        if drug_vs_standard_care:
            extracted_design_info.append("Drug vs standard care comparison")
        if double_blind_mentioned:
            extracted_design_info.append("Double-blind design")
        if open_label_mentioned:
            extracted_design_info.append("Open-label design")
        
        # Add to reasoning
        if placebo_mentioned:
            reasoning.append(f"Placebo indicators found: {placebo_phrases[:2]}")
        if drug_vs_standard_care:
            reasoning.append("Drug vs standard care pattern detected")
        if not placebo_mentioned and not standard_care_control:
            reasoning.append("No clear control group mentioned")
        if blinding_level != "none":
            reasoning.append(f"Blinding level: {blinding_level}")
        
        return PlaceboControlFeatures(
            placebo_mentioned=placebo_mentioned,
            placebo_controlled_mentioned=placebo_controlled_mentioned,
            double_blind_mentioned=double_blind_mentioned,
            placebo_phrases=placebo_phrases,
            control_group_mentioned=control_group_mentioned,
            control_group_description=control_group_description,
            standard_care_control=standard_care_control,
            active_control_mentioned=active_control_mentioned,
            randomized_mentioned=randomized_mentioned,
            blinding_level=blinding_level,
            trial_design_described=trial_design_described,
            drug_vs_standard_care=drug_vs_standard_care,
            drug_vs_placebo=drug_vs_placebo,
            drug_vs_active_control=drug_vs_active_control,
            no_clear_comparison=no_clear_comparison,
            intervention_described=intervention_described,
            comparator_described=comparator_described,
            treatment_arms_count=treatment_arms_count,
            allocation_concealment_mentioned=allocation_concealment_mentioned,
            intention_to_treat_mentioned=intention_to_treat_mentioned,
            per_protocol_mentioned=per_protocol_mentioned,
            open_label_mentioned=open_label_mentioned,
            single_blind_mentioned=single_blind_mentioned,
            unblinded_mentioned=unblinded_mentioned,
            methods_section_present=methods_section_present,
            design_specificity_score=design_specificity_score,
            comparison_clarity_score=comparison_clarity_score,
            detected_patterns=detected_patterns,
            extracted_design_info=extracted_design_info
        )
    
    def _determine_blinding_level(self, text: str) -> str:
        """Determine the level of blinding in the study"""
        if re.search(r'\btriple[-\s]blind(?:ed)?\b', text, re.IGNORECASE):
            return "triple"
        elif re.search(r'\bdouble[-\s]blind(?:ed)?\b', text, re.IGNORECASE):
            return "double"
        elif re.search(r'\bsingle[-\s]blind(?:ed)?\b', text, re.IGNORECASE):
            return "single"
        elif re.search(r'\bopen[-\s]label\b|\bunblinded\b', text, re.IGNORECASE):
            return "open"
        else:
            return "none"
    
    def _count_treatment_arms(self, text: str) -> int:
        """Count the number of treatment arms mentioned"""
        # Look for explicit numbers
        arm_counts = re.findall(r'\b(?:two|three|four|2|3|4)\s+(?:treatment\s+)?(?:arms?|groups?)\b', text, re.IGNORECASE)
        if arm_counts:
            count_map = {'two': 2, 'three': 3, 'four': 4, '2': 2, '3': 3, '4': 4}
            for count in arm_counts:
                number = count.split()[0].lower()
                if number in count_map:
                    return count_map[number]
        
        # Count based on randomization pattern
        randomization_pattern = re.search(r'\brandomized\s+(\d+):(\d+)', text, re.IGNORECASE)
        if randomization_pattern:
            return 2  # Most common is 1:1 randomization
        
        # Default assumption for randomized trials
        if re.search(r'\brandomized\b', text, re.IGNORECASE):
            return 2
        
        return 1
    
    def _calculate_design_specificity(self, text: str) -> float:
        """Calculate how specifically the study design is described"""
        score = 0.0
        
        # Base score for design mentions
        if re.search(r'\brandomized\s+controlled\s+trial\b', text, re.IGNORECASE):
            score += 0.3
        elif re.search(r'\brandomized\b.*?\btrial\b', text, re.IGNORECASE):
            score += 0.2
        
        # Score for blinding specification
        if re.search(r'\b(?:double|triple)[-\s]blind\b', text, re.IGNORECASE):
            score += 0.2
        elif re.search(r'\bopen[-\s]label\b', text, re.IGNORECASE):
            score += 0.1
        
        # Score for methodological details
        if re.search(r'\ballocation\s+concealment\b', text, re.IGNORECASE):
            score += 0.1
        if re.search(r'\brandomization\s+(?:sequence|method)\b', text, re.IGNORECASE):
            score += 0.1
        if re.search(r'\bstratified\s+randomization\b', text, re.IGNORECASE):
            score += 0.1
        
        # Score for analysis plan
        if re.search(r'\bintention[-\s]to[-\s]treat\b', text, re.IGNORECASE):
            score += 0.1
        if re.search(r'\bper[-\s]protocol\b', text, re.IGNORECASE):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_comparison_clarity(self, drug_vs_placebo: bool, drug_vs_standard: bool, 
                                    drug_vs_active: bool, control_descriptions: List[str]) -> float:
        """Calculate how clearly the comparison is described"""
        score = 0.0
        
        # Base score for clear comparison patterns
        if drug_vs_placebo:
            score += 0.4  # Highest score for placebo comparison
        elif drug_vs_active:
            score += 0.3  # High score for active control
        elif drug_vs_standard:
            score += 0.2  # Lower score for standard care
        
        # Score for control group description clarity
        score += min(len(control_descriptions) * 0.1, 0.3)
        
        # Score for specific control description
        for desc in control_descriptions:
            if len(desc.split()) >= 3:  # More detailed description
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _analyze_design_quality(self, features: PlaceboControlFeatures, 
                               reasoning: List[str]) -> str:
        """Analyze overall design quality"""
        
        quality_score = 0
        
        if features.placebo_controlled_mentioned:
            quality_score += 3
        if features.double_blind_mentioned:
            quality_score += 2
        if features.randomized_mentioned:
            quality_score += 1
        if features.allocation_concealment_mentioned:
            quality_score += 1
        
        if quality_score >= 5:
            analysis = "HIGH quality design with placebo control and proper blinding"
        elif quality_score >= 3:
            analysis = "MODERATE quality design with some methodological safeguards"
        elif quality_score >= 1:
            analysis = "BASIC design quality - limited methodological safeguards"
        else:
            analysis = "LOW quality design - major methodological limitations"
        
        reasoning.append(f"Design quality assessment: {analysis}")
        return analysis
    
    def _assess_bias_risk(self, features: PlaceboControlFeatures, 
                         reasoning: List[str]) -> str:
        """Assess risk of bias from control design"""
        
        bias_risks = []
        
        if not features.placebo_mentioned:
            bias_risks.append("No placebo control - risk of performance bias")
        
        if features.open_label_mentioned or features.unblinded_mentioned:
            bias_risks.append("Open-label design - high risk of performance and detection bias")
        
        if features.drug_vs_standard_care:
            bias_risks.append("Drug vs standard care - risk of time-varying confounding")
        
        if features.no_clear_comparison:
            bias_risks.append("Unclear comparison - cannot assess bias risk")
        
        if not features.randomized_mentioned:
            bias_risks.append("No randomization mentioned - risk of selection bias")
        
        if bias_risks:
            assessment = f"HIGH bias risk: {'; '.join(bias_risks[:2])}"
        elif features.single_blind_mentioned:
            assessment = "MODERATE bias risk: single-blind design"
        elif features.placebo_controlled_mentioned and features.double_blind_mentioned:
            assessment = "LOW bias risk: placebo-controlled, double-blind design"
        else:
            assessment = "MODERATE bias risk: some methodological limitations"
        
        reasoning.append(f"Bias risk assessment: {assessment}")
        return assessment
    
    def _get_detected_patterns(self, text: str, placebo_phrases: List[str], 
                              control_descriptions: List[str]) -> List[str]:
        """Get list of detected patterns for transparency"""
        detected = []
        
        # Add placebo patterns
        for phrase in set(placebo_phrases[:3]):  # Top unique phrases
            detected.append(f"placebo: {phrase}")
        
        # Add control patterns
        for desc in set(control_descriptions[:2]):  # Top unique descriptions
            detected.append(f"control: {desc}")
        
        # Add design patterns
        if re.search(r'\bdouble[-\s]blind\b', text, re.IGNORECASE):
            detected.append("design: double_blind")
        if re.search(r'\bopen[-\s]label\b', text, re.IGNORECASE):
            detected.append("design: open_label")
        if re.search(r'\brandomized\b', text, re.IGNORECASE):
            detected.append("design: randomized")
        
        return detected[:6]  # Limit for readability
    
    def _make_placebo_classification(self, features: PlaceboControlFeatures, 
                                   reasoning: List[str], 
                                   bias_risk_assessment: str) -> PlaceboControlResult:
        """Make placebo control classification based on methodological heuristic"""
        
        # PLACEBO CONTROL CLASSIFICATION LOGIC
        
        # 1. Clear placebo control mentioned
        if features.placebo_controlled_mentioned or (features.placebo_mentioned and features.double_blind_mentioned):
            reasoning.append("Placebo-controlled design detected")
            
            # Calculate confidence based on strength of evidence
            confidence = 0.9
            if features.placebo_controlled_mentioned and features.double_blind_mentioned:
                confidence = 0.95
            elif features.placebo_controlled_mentioned:
                confidence = 0.9
            else:
                confidence = 0.8
            
            return self._create_result(
                PlaceboControlCode.PLACEBO_CONTROLLED,
                confidence,
                "Placebo-controlled trial design - optimal methodological control",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 2. Drug vs standard care pattern (major methodological issue)
        elif features.drug_vs_standard_care:
            reasoning.append("Drug vs standard care comparison without placebo control")
            
            # Higher confidence if explicit "drug A vs standard care" pattern
            confidence = 0.85
            if any('specific_drug' in pattern or 'drug_A' in pattern 
                   for pattern in features.detected_patterns):
                confidence = 0.9
            
            return self._create_result(
                PlaceboControlCode.DRUG_VS_STANDARD_CARE,
                confidence,
                "Drug vs standard care comparison - lacks placebo control",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 3. Active control mentioned (better than standard care, but not placebo)
        elif features.active_control_mentioned and not features.standard_care_control:
            reasoning.append("Active control comparison - no placebo mentioned")
            
            return self._create_result(
                PlaceboControlCode.NO_PLACEBO_MENTIONED,
                0.7,
                "Active control comparison - placebo control not mentioned",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 4. Standard care control without drug vs standard care pattern
        elif features.standard_care_control and not features.drug_vs_standard_care:
            reasoning.append("Standard care control mentioned - no placebo control")
            
            return self._create_result(
                PlaceboControlCode.NO_PLACEBO_MENTIONED,
                0.75,
                "Standard care control - no placebo mentioned",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 5. Some control group mentioned but unclear
        elif features.control_group_mentioned:
            reasoning.append("Control group mentioned but type unclear")
            
            # Check if randomized trial (suggests proper design)
            if features.randomized_mentioned and features.double_blind_mentioned:
                reasoning.append("Randomized double-blind suggests placebo control")
                return self._create_result(
                    PlaceboControlCode.PLACEBO_CONTROLLED,
                    0.6,
                    "Randomized double-blind trial - likely placebo-controlled",
                    features,
                    reasoning,
                    bias_risk_assessment
                )
            else:
                return self._create_result(
                    PlaceboControlCode.NO_PLACEBO_MENTIONED,
                    0.6,
                    "Control group mentioned but placebo not specified",
                    features,
                    reasoning,
                    bias_risk_assessment
                )
        
        # 6. Randomized trial with double-blind (suggests placebo)
        elif features.randomized_mentioned and features.double_blind_mentioned:
            reasoning.append("Randomized double-blind trial suggests placebo control")
            
            return self._create_result(
                PlaceboControlCode.PLACEBO_CONTROLLED,
                0.6,
                "Double-blind randomized trial - likely placebo-controlled",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 7. Open-label or unblinded trial
        elif features.open_label_mentioned or features.unblinded_mentioned:
            reasoning.append("Open-label/unblinded design - no placebo control")
            
            return self._create_result(
                PlaceboControlCode.NO_PLACEBO_MENTIONED,
                0.85,
                "Open-label design - no placebo control",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 8. Randomized trial mentioned but control unclear
        elif features.randomized_mentioned:
            reasoning.append("Randomized trial but control group not clearly described")
            
            return self._create_result(
                PlaceboControlCode.NO_PLACEBO_MENTIONED,
                0.65,
                "Randomized trial - control group not clearly specified",
                features,
                reasoning,
                bias_risk_assessment
            )
        
        # 9. Default: Control unclear
        else:
            reasoning.append("Control group design not clearly described")
            return self._create_result(
                PlaceboControlCode.CONTROL_UNCLEAR,
                0.8,
                "Control group design not clearly defined",
                features,
                reasoning,
                "Cannot assess methodological quality"
            )
    
    def _create_result(self, code: PlaceboControlCode, confidence: float, 
                      message: str, features: PlaceboControlFeatures, 
                      reasoning: List[str], bias_risk_assessment: str) -> PlaceboControlResult:
        """Create a PlaceboControlResult object"""
        return PlaceboControlResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            methodological_implications=self.methodological_implications[code],
            bias_risk_assessment=bias_risk_assessment
        )
    
    def _create_empty_features(self) -> PlaceboControlFeatures:
        """Create empty features object for error cases"""
        return PlaceboControlFeatures(
            placebo_mentioned=False, placebo_controlled_mentioned=False, double_blind_mentioned=False,
            placebo_phrases=[], control_group_mentioned=False, control_group_description=[],
            standard_care_control=False, active_control_mentioned=False, randomized_mentioned=False,
            blinding_level="none", trial_design_described=False, drug_vs_standard_care=False,
            drug_vs_placebo=False, drug_vs_active_control=False, no_clear_comparison=True,
            intervention_described=False, comparator_described=False, treatment_arms_count=0,
            allocation_concealment_mentioned=False, intention_to_treat_mentioned=False,
            per_protocol_mentioned=False, open_label_mentioned=False, single_blind_mentioned=False,
            unblinded_mentioned=False, methods_section_present=False, design_specificity_score=0,
            comparison_clarity_score=0, detected_patterns=[], extracted_design_info=[]
        )



def run_check(abstract : str):# just a wrapper method
    classifier = PlaceboControlClassifier()
    result = classifier.check_placebo_control(abstract)
    return result