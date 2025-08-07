import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class EventCountCode(Enum):
    """Event count classification codes"""
    ADEQUATE_EVENTS = 0           # ≥30 events in both groups - Good statistical power
    INSUFFICIENT_EVENTS = 1       # <30 events in either group - MAJOR ISSUE (underpowered)
    EVENT_COUNT_UNCLEAR = 2       # Event counts not clearly reported
    CONTINUOUS_OUTCOME = 3        # Continuous outcome study - event analysis not applicable


@dataclass
class EventCountFeatures:
    """Container for event count analysis features"""
    # Event detection
    primary_events_detected: bool           # Primary outcome events found
    event_numbers_found: List[int]         # All numeric event counts found
    group_event_counts: Dict[str, List[int]]  # Events by group (intervention/control)
    
    # Statistical reporting
    total_events_reported: Optional[int]    # Total events across groups
    events_per_group_clear: bool           # Clear per-group event reporting
    percentage_events_found: List[float]   # Event percentages found
    
    # Sample size context
    total_sample_size: Optional[int]       # Total study sample size
    group_sample_sizes: Dict[str, int]     # Sample sizes by group
    overall_event_rate: Optional[float]    # Overall event rate if calculable
    
    # Results section analysis
    results_section_present: bool          # Results section identified
    statistical_results_found: List[str]   # Statistical test results
    confidence_intervals_found: List[str]  # Confidence intervals
    p_values_found: List[float]           # P-values found
    
    # Power/sample size mentions
    power_analysis_mentioned: bool         # Power calculation mentioned
    sample_size_justification: bool       # Sample size justification provided
    early_termination_mentioned: bool     # Early stopping mentioned
    futility_mentioned: bool              # Futility mentioned
    
    # Event rate indicators
    low_event_rate_indicators: List[str]   # Phrases suggesting low events
    rare_event_language: List[str]         # Language about rare events
    small_study_indicators: List[str]      # Indicators of small study
    
    # Numerical extraction details
    primary_outcome_numbers: List[int]     # Numbers in primary outcome context
    results_numbers: List[int]             # Numbers in results context
    extracted_ratios: List[str]           # Extracted n/N ratios
    
    # Context analysis
    study_design_type: str                # Study design (RCT, observational, etc.)
    intervention_vs_control_clear: bool    # Clear group distinction
    follow_up_duration_mentioned: bool    # Follow-up time reported
    
    # Confidence assessment
    event_extraction_confidence: float    # Confidence in event extraction
    group_assignment_confidence: float    # Confidence in group assignment
    
    # Detected patterns for transparency
    detected_patterns: List[str]
    extracted_event_info: List[str]


@dataclass
class EventCountResult:
    code: EventCountCode
    confidence: float
    message: str
    features: EventCountFeatures
    reasoning: List[str]
    clinical_implications: str
    interpretation_assessment: str
    recommended_actions: List[str]


class EventCountClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for advanced NLP"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_event_patterns()
        self._setup_clinical_implications()
    
    def _setup_event_patterns(self):
        """Define comprehensive patterns for event detection - IMPROVED COVERAGE"""
        
        # Primary outcome event patterns (EXPANDED)
        self.primary_outcome_event_patterns = [
            # Direct primary outcome events with numbers
            (r'primary\s+outcome\s+(?:occurred\s+in\s+|was\s+observed\s+in\s+|was\s+reached\s+by\s+)?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?|subjects?)', 4, 'primary_outcome_direct'),
            (r'primary\s+(?:end\s*)?point\s+(?:occurred\s+in\s+|was\s+observed\s+in\s+|was\s+reached\s+by\s+)?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?|subjects?)', 4, 'primary_endpoint_direct'),
            
            # Primary outcome with percentages only (CRITICAL FIX)
            (r'primary\s+outcome.*?(\d+(?:\.\d+)?)\s*%', 3, 'primary_outcome_percent'),
            (r'primary\s+(?:end\s*)?point.*?(\d+(?:\.\d+)?)\s*%', 3, 'primary_endpoint_percent'),
            
            # Primary outcome in different sentence structures
            (r'(?:for\s+the\s+)?primary\s+outcome.*?(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 3, 'primary_flexible'),
            (r'(?:the\s+)?primary\s+(?:end\s*)?point.*?(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 3, 'primary_endpoint_flexible'),
            
            # Results section primary outcome (EXPANDED)
            (r'(?:Results?|RESULTS?).*?primary\s+outcome.*?(\d+(?:\.\d+)?)\s*%', 3, 'results_primary_percent'),
            (r'(?:Results?|RESULTS?).*?primary\s+outcome.*?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?)', 3, 'results_primary'),
            (r'(?:Results?|RESULTS?).*?primary\s+(?:end\s*)?point.*?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?)', 3, 'results_primary_endpoint'),
        ]
        
        # Event reporting patterns (MUCH MORE COMPREHENSIVE)
        self.event_reporting_patterns = [
            # Standard n/N format
            (r'(\d+)\s*/\s*(\d+)\s*(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 3, 'n_over_N'),
            (r'(\d+)\s+of\s+(\d+)\s+(?:patients?|participants?|subjects?)\s*(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 3, 'n_of_N'),
            
            # Percentage with absolute numbers
            (r'(\d+)\s*(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?\s+(?:patients?|participants?|subjects?)', 2, 'absolute_with_percent'),
            (r'(\d+(?:\.\d+)?%)\s*(?:\(\s*(\d+)\s*(?:of\s+(\d+))?\s*\))', 2, 'percent_with_absolute'),
            
            # Percentage only (CRITICAL FIX)
            (r'(\d+(?:\.\d+)?)\s*%\s+(?:of\s+(?:patients?|participants?|subjects?))?', 2, 'percent_only'),
            (r'(?:in\s+)?(\d+(?:\.\d+)?)\s*%\s+(?:of\s+(?:the\s+)?(?:patients?|participants?|subjects?))?', 2, 'percent_only_flexible'),
            
            # Event occurrence language (EXPANDED)
            (r'(?:occurred\s+in|observed\s+in|seen\s+in|found\s+in|developed\s+in)\s+(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?)', 2, 'occurred_in'),
            (r'(?:developed|experienced|had|suffered|presented\s+with)\s+.*?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?)', 2, 'developed'),
            (r'(\d+)\s+(?:patients?|participants?|subjects?)\s+(?:developed|experienced|had|suffered)', 2, 'patients_developed'),
            
            # Comparative results (CRITICAL)
            (r'(\d+(?:\.\d+)?)\s*%\s+(?:vs\.?|versus|compared\s+(?:with|to))\s+(\d+(?:\.\d+)?)\s*%', 3, 'percent_comparison'),
            (r'(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?\s+(?:vs\.?|versus)\s+(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 4, 'absolute_comparison'),
            
            # Death/mortality specific (IMPORTANT)
            (r'(\d+)\s+(?:deaths?|died|mortality)', 2, 'death_absolute'),
            (r'(?:mortality|death)\s+(?:rate\s+)?(?:was\s+)?(\d+(?:\.\d+)?)\s*%', 2, 'mortality_percent'),
            (r'(\d+(?:\.\d+)?)\s*%\s+(?:mortality|died|death\s+rate)', 2, 'percent_mortality'),
            
            # Survival analysis results (NEW)
            (r'(\d+(?:\.\d+)?)\s*%\s+survived?', 2, 'survival_percent'),
            (r'survival\s+(?:rate\s+)?(?:was\s+)?(\d+(?:\.\d+)?)\s*%', 2, 'survival_rate'),
            
            # Time-to-event results (NEW)
            (r'(\d+)\s+(?:events?|outcomes?)\s+(?:occurred|observed)', 2, 'events_occurred'),
            (r'(?:events?|outcomes?)\s+(?:occurred\s+in\s+)?(\d+)', 2, 'events_in_number'),
        ]
        
        # Group identification patterns (ENHANCED)
        self.group_identification_patterns = [
            # Intervention group indicators
            (r'(?:intervention|treatment|active|drug|experimental)\s+(?:group|arm)', 'intervention'),
            (r'(?:receiving|treated\s+with|given|assigned\s+to)\s+\w+', 'intervention'),
            (r'(?:randomized|randomised)\s+to\s+(?:receive\s+)?\w+', 'intervention'),
            
            # Control group indicators  
            (r'(?:control|placebo|standard|usual\s+care)\s+(?:group|arm)', 'control'),
            (r'(?:receiving|treated\s+with|given|assigned\s+to)\s+(?:placebo|standard\s+care)', 'control'),
            (r'(?:randomized|randomised)\s+to\s+(?:receive\s+)?(?:placebo|standard\s+care)', 'control'),
            
            # Group A/B patterns
            (r'group\s+[A1]\b', 'group_A'),
            (r'group\s+[B2]\b', 'group_B'),
            (r'arm\s+[A1]\b', 'arm_A'),
            (r'arm\s+[B2]\b', 'arm_B'),
            
            # More flexible group patterns (NEW)
            (r'(?:first|1st)\s+group', 'group_1'),
            (r'(?:second|2nd)\s+group', 'group_2'),
            (r'(?:study|treatment)\s+group', 'study_group'),
        ]
        
        # Results section event patterns (GREATLY EXPANDED)
        self.results_section_patterns = [
            # Direct results statements
            (r'(?:Results?|RESULTS?).*?(?:The\s+)?(?:primary\s+)?(?:outcome|endpoint)\s+(?:was\s+)?(?:reached|achieved|occurred)\s+(?:in\s+)?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?)', 4, 'results_outcome_occurred'),
            (r'(?:Results?|RESULTS?).*?(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?\s+(?:vs\.?|versus|compared\s+to)\s+(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 4, 'results_comparison'),
            (r'(?:Results?|RESULTS?).*?(?:intervention|treatment)\s+(?:group|arm)\s*:\s*(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 3, 'results_intervention'),
            (r'(?:Results?|RESULTS?).*?(?:control|placebo)\s+(?:group|arm)\s*:\s*(\d+)\s+(?:\(\s*(\d+(?:\.\d+)?%)\s*\))?', 3, 'results_control'),
            
            # Percentage-only results (CRITICAL FIX)
            (r'(?:Results?|RESULTS?).*?(\d+(?:\.\d+)?)\s*%\s+(?:vs\.?|versus)\s+(\d+(?:\.\d+)?)\s*%', 3, 'results_percent_vs'),
            (r'(?:Results?|RESULTS?).*?(?:intervention|treatment).*?(\d+(?:\.\d+)?)\s*%', 2, 'results_intervention_percent'),
            (r'(?:Results?|RESULTS?).*?(?:control|placebo).*?(\d+(?:\.\d+)?)\s*%', 2, 'results_control_percent'),
            
            # Statistical results with events
            (r'(?:odds\s+ratio|OR|risk\s+ratio|RR|hazard\s+ratio|HR)\s*[:\s]*(\d+(?:\.\d+)?)', 2, 'statistical_ratio'),
            (r'(?:p\s*=\s*|p\s*<\s*|p\s*>\s*)(\d+(?:\.\d+)?)', 2, 'p_value'),
            (r'95%\s*(?:CI|confidence\s+interval)\s*[:\s]*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', 2, 'confidence_interval'),
            
            # Summary statistics (NEW)
            (r'(?:mean|median|average).*?(\d+(?:\.\d+)?)', 1, 'summary_stat'),
            (r'(?:total|overall).*?(\d+)\s+(?:events?|cases|outcomes?)', 2, 'total_events'),
            
            # Conclusions section results (IMPORTANT FIX)
            (r'(?:Conclusions?|CONCLUSIONS?).*?(\d+(?:\.\d+)?)\s*%', 2, 'conclusions_percent'),
            (r'(?:Conclusions?|CONCLUSIONS?).*?(\d+)\s+(?:of\s+)?(\d+)?\s+(?:patients?|participants?)', 2, 'conclusions_absolute'),
        ]
        
        # Low event rate indicators (ENHANCED)
        self.low_event_indicators = [
            # Direct low event language
            (r'\b(?:low|rare|few|infrequent)\s+(?:event\s+)?(?:rate|incidence|occurrence)\b', 3, 'low_event_language'),
            (r'\b(?:rare|uncommon|infrequent)\s+(?:events?|outcomes?|endpoints?)\b', 3, 'rare_event_language'),
            (r'\bfew\s+(?:events?|outcomes?|cases)\b', 2, 'few_events'),
            
            # Power/statistical issues
            (r'\b(?:underpowered|under\s*powered|insufficient\s+power)\b', 4, 'underpowered'),
            (r'\b(?:small\s+effect\s+size|limited\s+power)\b', 3, 'limited_power'),
            (r'\b(?:wide\s+confidence\s+intervals?|large\s+confidence\s+intervals?)\b', 2, 'wide_ci'),
            
            # Study limitations
            (r'\b(?:small\s+sample\s+size|limited\s+sample\s+size)\b', 3, 'small_sample'),
            (r'\b(?:stopped\s+(?:early\s+)?(?:for\s+)?futility|futility\s+analysis)\b', 4, 'futility_stopping'),
            (r'\b(?:interim\s+analysis|early\s+termination)\b', 2, 'early_termination'),
        ]
        
        # Sample size extraction patterns (ENHANCED)
        self.sample_size_patterns = [
            # Total sample size
            (r'(?:randomized|randomised|enrolled|included)\s+(\d+)\s+(?:patients?|participants?|subjects?)', 3, 'randomized_total'),
            (r'(?:sample\s+size|study\s+population)\s+(?:was\s+|of\s+)?(\d+)', 2, 'sample_size'),
            (r'(?:n\s*=\s*)(\d+)', 2, 'n_equals'),
            (r'(?:total\s+of\s+)?(\d+)\s+(?:patients?|participants?|subjects?)\s+(?:were\s+)?(?:enrolled|randomized|included)', 3, 'total_enrolled'),
            
            # Group sample sizes (ENHANCED)
            (r'(?:intervention|treatment|active)\s+(?:group|arm)\s*(?:\(\s*)?(?:n\s*=\s*)?(\d+)', 3, 'intervention_n'),
            (r'(?:control|placebo|standard)\s+(?:group|arm)\s*(?:\(\s*)?(?:n\s*=\s*)?(\d+)', 3, 'control_n'),
            (r'(\d+)\s+(?:in\s+the\s+)?(?:intervention|treatment|active)\s+(?:group|arm)', 3, 'n_intervention'),
            (r'(\d+)\s+(?:in\s+the\s+)?(?:control|placebo|standard)\s+(?:group|arm)', 3, 'n_control'),
        ]
        
        # High precision numerical extraction (NEW)
        self.numerical_context_patterns = [
            # Numbers in specific contexts
            (r'(?:primary\s+)?(?:outcome|endpoint).*?(\d+)', 3, 'outcome_number'),
            (r'(?:results|findings).*?(\d+)', 2, 'results_number'),
            (r'(?:events?|cases|incidents?).*?(\d+)', 2, 'event_number'),
            (r'(\d+)\s+(?:events?|cases|incidents?)', 2, 'number_events'),
            (r'(?:mortality|death).*?(\d+)', 2, 'mortality_number'),
            (r'(\d+)\s+died', 2, 'died_number'),
        ]
        
        # Continuous outcome detection patterns (NEW)
        self.continuous_outcome_patterns = [
            # Weight/BMI outcomes
            (r'(?:percent|%)\s+change\s+in\s+(?:weight|body\s+weight|BMI)', 4, 'weight_change_percent'),
            (r'(?:weight|BMI)\s+(?:change|reduction|loss)', 3, 'weight_change'),
            (r'(?:mean|average)\s+(?:weight|BMI)\s+(?:change|reduction)', 3, 'mean_weight_change'),
            (r'least[-\s]squares\s+mean.*?(?:weight|BMI)', 4, 'lsm_weight'),
            
            # Other continuous outcomes
            (r'(?:percent|%)\s+change\s+in\s+(?:blood\s+pressure|cholesterol|glucose|HbA1c)', 4, 'biomarker_change_percent'),
            (r'(?:mean|average)\s+change\s+in', 3, 'mean_change'),
            (r'(?:difference\s+in\s+)?(?:mean|average)', 3, 'mean_difference'),
            (r'least[-\s]squares\s+mean', 4, 'least_squares_mean'),
            (r'(?:change|reduction|increase)\s+from\s+baseline', 3, 'baseline_change'),
            
            # Primary outcome indicators for continuous
            (r'primary\s+(?:end\s*)?point\s+was\s+(?:the\s+)?(?:percent|%)\s+change', 4, 'primary_continuous'),
            (r'primary\s+outcome\s+was\s+(?:the\s+)?(?:percent|%)\s+change', 4, 'primary_continuous'),
            (r'primary\s+(?:end\s*)?point\s+was\s+(?:the\s+)?(?:change|difference)', 3, 'primary_change'),
            
            # Effect size indicators
            (r'(?:effect\s+size|Cohen)', 2, 'effect_size'),
            (r'(?:standardized\s+)?mean\s+difference', 3, 'standardized_mean_diff'),
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            EventCountCode.ADEQUATE_EVENTS:
                "Adequate event counts (≥30 in both groups) provide sufficient statistical power "
                "to detect clinically meaningful effect sizes. Results are likely reliable and "
                "confidence intervals appropriately narrow for clinical interpretation.",
                
            EventCountCode.INSUFFICIENT_EVENTS:
                "Low event counts (<30 in either group) indicate insufficient statistical power. "
                "This significantly limits the ability to detect true treatment effects and "
                "leads to wide confidence intervals that preclude definitive conclusions. "
                "Results may be due to chance rather than true treatment effects.",
                
            EventCountCode.EVENT_COUNT_UNCLEAR:
                "Event counts not clearly reported, making it impossible to assess statistical "
                "power and reliability of results. This represents a critical reporting deficiency "
                "that limits clinical interpretation.",
                
            EventCountCode.CONTINUOUS_OUTCOME:
                "This study uses continuous outcomes (e.g., percent change in weight, blood pressure) "
                "rather than binary events. Statistical power for continuous outcomes depends on "
                "sample size and effect size, not event counts. Standard event count analysis is not applicable."
        }
        
        self.recommended_actions = {
            EventCountCode.ADEQUATE_EVENTS: [
                "Results can be interpreted with confidence",
                "Effect sizes and confidence intervals are reliable",
                "Clinical decision-making can proceed based on findings"
            ],
            
            EventCountCode.INSUFFICIENT_EVENTS: [
                "Interpret results with extreme caution",
                "Consider wider confidence intervals than reported",
                "Larger studies needed before clinical implementation",
                "Pool data with similar studies if possible",
                "Consider study as hypothesis-generating only"
            ],
            
            EventCountCode.EVENT_COUNT_UNCLEAR: [
                "Request detailed event reporting from authors",
                "Cannot assess reliability of results",
                "Consider study as providing limited evidence",
                "Seek additional studies with clear reporting"
            ],
            
            EventCountCode.CONTINUOUS_OUTCOME: [
                "Assess statistical power based on sample size and effect size",
                "Look for confidence intervals around mean differences",
                "Standard event count rules do not apply",
                "Focus on clinical significance of effect size"
            ]
        }
    
    def _is_continuous_outcome_study(self, text: str, reasoning: List[str]) -> bool:
        """Detect if this is a continuous outcome study (not event-based)"""
        
        continuous_indicators = []
        
        for pattern, priority, context in self.continuous_outcome_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                continuous_indicators.append((context, priority, len(matches)))
                reasoning.append(f"Continuous outcome indicator: {context} (priority: {priority})")
        
        # Calculate continuous outcome score
        total_score = sum(priority * count for _, priority, count in continuous_indicators)
        
        # High confidence thresholds
        if total_score >= 8:  # Strong evidence
            reasoning.append(f"Strong continuous outcome evidence (score: {total_score})")
            return True
        elif total_score >= 4:  # Moderate evidence
            reasoning.append(f"Moderate continuous outcome evidence (score: {total_score})")
            return True
        
        return False
    def check_event_count(self, text: str) -> EventCountResult:
        """
        Analyze text to detect insufficient event counts - IMPROVED VERSION
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            EventCountResult with event count assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                EventCountCode.EVENT_COUNT_UNCLEAR,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess event counts"
            )
        
        # NEW: Check if this is a continuous outcome study FIRST
        if self._is_continuous_outcome_study(text, reasoning):
            # Extract some basic features for transparency
            sections = self._extract_sections(text)
            features = self._create_continuous_outcome_features(text, sections, reasoning)
            
            reasoning.append("Continuous outcome study detected - event count analysis not applicable")
            return self._create_result(
                EventCountCode.CONTINUOUS_OUTCOME,
                0.95,
                "Continuous outcome study (e.g., % weight change) - event count analysis not applicable",
                features,
                reasoning,
                "This study uses continuous outcomes rather than binary events. Statistical power depends on sample size and effect size, not event counts."
            )
        
        # Continue with existing event analysis for binary outcome studies
        # Extract sections for better analysis
        sections = self._extract_sections(text)
        
        # Extract comprehensive event count features
        features = self._extract_event_features(text, sections, reasoning)
        
        # Perform high-accuracy event count analysis
        return self._classify_event_count(features, reasoning)
    
    def _create_continuous_outcome_features(self, text: str, sections: Dict[str, str], reasoning: List[str]) -> EventCountFeatures:
        """Create features object for continuous outcome studies"""
        
        # Extract basic information for transparency
        total_sample_size = self._extract_sample_size(text, reasoning)
        group_sample_sizes = self._extract_group_sample_sizes(text, reasoning)
        study_design_type = self._identify_study_design(text)
        
        # Look for treatment effects (the percentages in continuous studies)
        treatment_effects = []
        effect_patterns = [
            r'([-−]?\d+(?:\.\d+)?)\s*%.*?(?:CI|confidence\s+interval)',  # Effect with CI
            r'was\s+([-−]?\d+(?:\.\d+)?)\s*%',  # "was -20.2%"
            r'([-−]?\d+(?:\.\d+)?)\s*%.*?with\s+(?:tirzepatide|semaglutide|intervention|treatment)',  # Effect with treatment
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    effect = float(match)
                    treatment_effects.append(abs(effect))  # Store absolute value
                    reasoning.append(f"Treatment effect found: {effect}%")
                except ValueError:
                    continue
        
        detected_patterns = []
        if treatment_effects:
            detected_patterns.append(f"continuous: treatment_effects_found")
        if total_sample_size:
            detected_patterns.append(f"sample_size: {total_sample_size}")
        if group_sample_sizes:
            detected_patterns.append(f"groups: {len(group_sample_sizes)}_groups")
        
        extracted_event_info = [
            f"Study type: Continuous outcome",
            f"Treatment effects found: {len(treatment_effects)}",
            f"Sample size: {total_sample_size or 'unclear'}"
        ]
        
        return EventCountFeatures(
            primary_events_detected=False,  # Not applicable for continuous
            event_numbers_found=[],
            group_event_counts={'intervention': [], 'control': [], 'unknown': []},
            total_events_reported=None,
            events_per_group_clear=False,
            percentage_events_found=treatment_effects,  # Store treatment effects here
            total_sample_size=total_sample_size,
            group_sample_sizes=group_sample_sizes,
            overall_event_rate=None,
            results_section_present=bool(re.search(r'\b(?:Results?|RESULTS?)\b', text)),
            statistical_results_found=[],
            confidence_intervals_found=[],
            p_values_found=[],
            power_analysis_mentioned=False,
            sample_size_justification=False,
            early_termination_mentioned=False,
            futility_mentioned=False,
            low_event_rate_indicators=[],
            rare_event_language=[],
            small_study_indicators=[],
            primary_outcome_numbers=[],
            results_numbers=[],
            extracted_ratios=[],
            study_design_type=study_design_type,
            intervention_vs_control_clear=True,  # Usually clear in RCTs
            follow_up_duration_mentioned=bool(re.search(r'\b(?:follow[-\s]up|followed\s+for|week|month)\b', text, re.IGNORECASE)),
            event_extraction_confidence=0.0,  # Not applicable
            group_assignment_confidence=0.9 if group_sample_sizes else 0.5,
            detected_patterns=detected_patterns,
            extracted_event_info=extracted_event_info
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections of the abstract"""
        sections = {}
        
        # Extract Results section
        results_match = re.search(r'(?:Results?|RESULTS?).*?(?=Conclusions?|Discussion|$)', text, re.IGNORECASE | re.DOTALL)
        if results_match:
            sections['results'] = results_match.group(0)
        
        # Extract Conclusions section
        conclusions_match = re.search(r'(?:Conclusions?|CONCLUSIONS?).*?$', text, re.IGNORECASE | re.DOTALL)
        if conclusions_match:
            sections['conclusions'] = conclusions_match.group(0)
        
        # Extract Methods section
        methods_match = re.search(r'(?:Methods?|METHODS?).*?(?=Results?|Conclusions?|$)', text, re.IGNORECASE | re.DOTALL)
        if methods_match:
            sections['methods'] = methods_match.group(0)
        
        return sections
    
    def _extract_event_features(self, text: str, sections: Dict[str, str], reasoning: List[str]) -> EventCountFeatures:
        """Extract event count features with maximum precision - IMPROVED"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Initialize tracking variables
        primary_events_detected = False
        event_numbers_found = []
        group_event_counts = {'intervention': [], 'control': [], 'unknown': []}
        
        # Focus analysis on Results section if available
        analysis_text = sections.get('results', text)
        if 'results' in sections:
            reasoning.append("Focusing analysis on Results section")
        
        # Extract primary outcome events (HIGHEST PRIORITY)
        primary_outcome_numbers = []
        for pattern, priority, context in self.primary_outcome_event_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                primary_events_detected = True
                try:
                    # Handle both absolute numbers and percentages
                    if '%' in match.group(0):
                        # Extract percentage
                        percent_val = float(match.group(1))
                        reasoning.append(f"Primary outcome percentage found: {percent_val}% ({context})")
                        # Try to convert to absolute numbers if we have sample size
                        # For now, store the percentage info
                        primary_outcome_numbers.append(int(percent_val))  # Store as int for processing
                    else:
                        # Extract absolute count
                        event_count = int(match.group(1))
                        primary_outcome_numbers.append(event_count)
                        event_numbers_found.append(event_count)
                        
                        # Try to extract total N if present
                        if len(match.groups()) > 1 and match.group(2):
                            try:
                                total_n = int(match.group(2))
                                if total_n > event_count:  # Sanity check
                                    event_numbers_found.append(total_n)
                            except (ValueError, IndexError):
                                pass
                                
                        reasoning.append(f"Primary outcome events found: {event_count} ({context})")
                except (ValueError, IndexError):
                    continue
        
        # Extract all event reporting patterns - COMPREHENSIVE
        results_numbers = []
        extracted_ratios = []
        percentage_events_found = []
        
        for pattern, priority, context in self.event_reporting_patterns:
            matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
            for match in matches:
                try:
                    if '%' in match.group(0) or 'percent' in context:
                        # Handle percentage
                        percent_val = float(match.group(1))
                        percentage_events_found.append(percent_val)
                        reasoning.append(f"Event percentage detected: {percent_val}% ({context})")
                    else:
                        # Handle absolute numbers
                        event_count = int(match.group(1))
                        event_numbers_found.append(event_count)
                        results_numbers.append(event_count)
                        
                        # Extract ratio information
                        if len(match.groups()) > 1 and match.group(2):
                            try:
                                total_n = int(match.group(2))
                                extracted_ratios.append(f"{event_count}/{total_n}")
                                event_numbers_found.append(total_n)
                            except (ValueError, IndexError):
                                pass
                                
                        reasoning.append(f"Event pattern detected: {event_count} ({context})")
                except (ValueError, IndexError):
                    continue
        
        # Extract results section patterns - ENHANCED
        statistical_results_found = []
        confidence_intervals_found = []
        p_values_found = []
        
        for pattern, priority, context in self.results_section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                statistical_results_found.append(match.group(0)[:100])  # First 100 chars
                
                # Extract numerical values
                for group in match.groups():
                    if group:
                        try:
                            # Try to extract as number
                            if '.' in group:
                                num_val = float(group)
                                if 0 < num_val < 1:  # Likely p-value or proportion
                                    p_values_found.append(num_val)
                                elif num_val <= 100 and 'percent' in context:  # Likely percentage
                                    percentage_events_found.append(num_val)
                                elif num_val < 10:  # Likely ratio
                                    pass  # Statistical ratio
                                else:  # Likely event count
                                    event_numbers_found.append(int(num_val))
                            else:
                                num_val = int(group)
                                if num_val < 1000:  # Reasonable event count
                                    event_numbers_found.append(num_val)
                                    results_numbers.append(num_val)
                        except (ValueError, TypeError):
                            continue
        
        # Calculate events from percentages if we have sample sizes
        total_sample_size = self._extract_sample_size(text, reasoning)
        group_sample_sizes = self._extract_group_sample_sizes(text, reasoning)
        
        # Convert percentages to absolute numbers where possible
        if percentage_events_found and total_sample_size:
            for percent in percentage_events_found:
                if 0 < percent <= 100:  # Valid percentage
                    estimated_events = int((percent / 100) * total_sample_size)
                    if estimated_events > 0:
                        event_numbers_found.append(estimated_events)
                        reasoning.append(f"Converted {percent}% to ~{estimated_events} events (based on N={total_sample_size})")
        
        # Identify group assignments (CRITICAL FOR ACCURACY)
        intervention_indicators = 0
        control_indicators = 0
        
        for pattern, group_type in self.group_identification_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if any(x in group_type for x in ['intervention', 'treatment', 'active', 'A', '1']):
                    intervention_indicators += len(matches)
                elif any(x in group_type for x in ['control', 'placebo', 'B', '2']):
                    control_indicators += len(matches)
        
        intervention_vs_control_clear = (intervention_indicators > 0 and control_indicators > 0)
        
        # Assign events to groups using advanced pattern matching
        group_event_counts = self._assign_events_to_groups(text, event_numbers_found, percentage_events_found, 
                                                          group_sample_sizes, reasoning)
        
        # Calculate overall event rate if possible
        overall_event_rate = None
        if event_numbers_found and total_sample_size:
            # Use largest reasonable event count as likely total events
            reasonable_events = [n for n in event_numbers_found if 0 < n < total_sample_size]
            if reasonable_events:
                max_events = max(reasonable_events)
                overall_event_rate = max_events / total_sample_size
        elif percentage_events_found:
            # Use highest percentage as overall rate
            overall_event_rate = max(percentage_events_found) / 100 if percentage_events_found else None
        
        # Detect low event rate indicators
        low_event_rate_indicators = []
        rare_event_language = []
        small_study_indicators = []
        
        for pattern, priority, context in self.low_event_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if 'low_event' in context or 'rare_event' in context or 'few_events' in context:
                    low_event_rate_indicators.extend(matches)
                    rare_event_language.extend(matches)
                elif 'power' in context or 'underpowered' in context:
                    small_study_indicators.extend(matches)
                elif 'small' in context or 'futility' in context:
                    small_study_indicators.extend(matches)
        
        # Check for power analysis and study design
        power_analysis_mentioned = bool(re.search(
            r'\b(?:power\s+analysis|power\s+calculation|sample\s+size\s+calculation)\b',
            text, re.IGNORECASE
        ))
        
        sample_size_justification = bool(re.search(
            r'\b(?:sample\s+size.*?justif|power.*?detect|alpha.*?beta)\b',
            text, re.IGNORECASE
        ))
        
        early_termination_mentioned = bool(re.search(
            r'\b(?:stopped\s+early|early\s+termination|interim\s+analysis)\b',
            text, re.IGNORECASE
        ))
        
        futility_mentioned = bool(re.search(
            r'\b(?:futility|stopped.*?futility|futility.*?analysis)\b',
            text, re.IGNORECASE
        ))
        
        # Determine study design
        study_design_type = self._identify_study_design(text)
        
        # Calculate confidence scores
        event_extraction_confidence = self._calculate_extraction_confidence(
            primary_events_detected, len(event_numbers_found), len(extracted_ratios), 
            len(percentage_events_found)
        )
        
        group_assignment_confidence = self._calculate_group_confidence(
            intervention_vs_control_clear, len(group_event_counts['intervention']), 
            len(group_event_counts['control'])
        )
        
        # Check for clear results section
        results_section_present = bool(re.search(
            r'\b(?:Results?|RESULTS?)\b', text
        ))
        
        # Create transparency information
        detected_patterns = self._get_detected_patterns(text, primary_events_detected, 
                                                      len(event_numbers_found), len(percentage_events_found))
        
        extracted_event_info = []
        if primary_events_detected:
            extracted_event_info.append(f"Primary outcome events: {len(primary_outcome_numbers)}")
        if event_numbers_found:
            extracted_event_info.append(f"Total event numbers found: {len(event_numbers_found)}")
        if percentage_events_found:
            extracted_event_info.append(f"Event percentages found: {len(percentage_events_found)}")
        if group_event_counts['intervention'] or group_event_counts['control']:
            extracted_event_info.append(f"Group events: {len(group_event_counts['intervention'])} intervention, {len(group_event_counts['control'])} control")
        
        return EventCountFeatures(
            primary_events_detected=primary_events_detected,
            event_numbers_found=event_numbers_found,
            group_event_counts=group_event_counts,
            total_events_reported=max(event_numbers_found) if event_numbers_found else None,
            events_per_group_clear=(len(group_event_counts['intervention']) > 0 and len(group_event_counts['control']) > 0),
            percentage_events_found=percentage_events_found,
            total_sample_size=total_sample_size,
            group_sample_sizes=group_sample_sizes,
            overall_event_rate=overall_event_rate,
            results_section_present=results_section_present,
            statistical_results_found=statistical_results_found,
            confidence_intervals_found=confidence_intervals_found,
            p_values_found=p_values_found,
            power_analysis_mentioned=power_analysis_mentioned,
            sample_size_justification=sample_size_justification,
            early_termination_mentioned=early_termination_mentioned,
            futility_mentioned=futility_mentioned,
            low_event_rate_indicators=low_event_rate_indicators,
            rare_event_language=rare_event_language,
            small_study_indicators=small_study_indicators,
            primary_outcome_numbers=primary_outcome_numbers,
            results_numbers=results_numbers,
            extracted_ratios=extracted_ratios,
            study_design_type=study_design_type,
            intervention_vs_control_clear=intervention_vs_control_clear,
            follow_up_duration_mentioned=bool(re.search(r'\b(?:follow[-\s]up|followed\s+for)\b', text, re.IGNORECASE)),
            event_extraction_confidence=event_extraction_confidence,
            group_assignment_confidence=group_assignment_confidence,
            detected_patterns=detected_patterns,
            extracted_event_info=extracted_event_info
        )
    
    def _assign_events_to_groups(self, text: str, event_numbers: List[int], 
                               percentages: List[float], group_sizes: Dict[str, int],
                               reasoning: List[str]) -> Dict[str, List[int]]:
        """Assign event counts to intervention/control groups - ENHANCED"""
        
        group_events = {'intervention': [], 'control': [], 'unknown': []}
        
        # Look for explicit group mentions with numbers
        intervention_patterns = [
            r'(?:intervention|treatment|active|drug)\s+(?:group|arm).*?(\d+(?:\.\d+)?)\s*%',
            r'(?:intervention|treatment|active|drug)\s+(?:group|arm).*?(\d+)',
            r'(\d+(?:\.\d+)?)\s*%.*?(?:intervention|treatment|active|drug)\s+(?:group|arm)',
            r'(\d+).*?(?:intervention|treatment|active|drug)\s+(?:group|arm)',
            r'(?:randomized|randomised)\s+to\s+(?:receive\s+)?\w+.*?(\d+)',
        ]
        
        control_patterns = [
            r'(?:control|placebo|standard)\s+(?:group|arm).*?(\d+(?:\.\d+)?)\s*%',
            r'(?:control|placebo|standard)\s+(?:group|arm).*?(\d+)',
            r'(\d+(?:\.\d+)?)\s*%.*?(?:control|placebo|standard)\s+(?:group|arm)',
            r'(\d+).*?(?:control|placebo|standard)\s+(?:group|arm)',
            r'(?:randomized|randomised)\s+to\s+(?:receive\s+)?(?:placebo|standard).*?(\d+)',
        ]
        
        # Extract intervention group events
        for pattern in intervention_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    if '%' in match.group(0):
                        # Handle percentage
                        percent_val = float(match.group(1))
                        if 'intervention' in group_sizes:
                            event_count = int((percent_val / 100) * group_sizes['intervention'])
                            group_events['intervention'].append(event_count)
                            reasoning.append(f"Intervention group: {percent_val}% = ~{event_count} events")
                        else:
                            # Store percentage for later conversion
                            group_events['intervention'].append(int(percent_val))
                            reasoning.append(f"Intervention group: {percent_val}% (percentage)")
                    else:
                        # Handle absolute number
                        event_count = int(match.group(1))
                        if event_count in event_numbers or event_count < 1000:  # Reasonable event count
                            group_events['intervention'].append(event_count)
                            reasoning.append(f"Intervention group events: {event_count}")
                except (ValueError, IndexError):
                    continue
        
        # Extract control group events
        for pattern in control_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    if '%' in match.group(0):
                        # Handle percentage
                        percent_val = float(match.group(1))
                        if 'control' in group_sizes:
                            event_count = int((percent_val / 100) * group_sizes['control'])
                            group_events['control'].append(event_count)
                            reasoning.append(f"Control group: {percent_val}% = ~{event_count} events")
                        else:
                            # Store percentage for later conversion
                            group_events['control'].append(int(percent_val))
                            reasoning.append(f"Control group: {percent_val}% (percentage)")
                    else:
                        # Handle absolute number
                        event_count = int(match.group(1))
                        if event_count in event_numbers or event_count < 1000:  # Reasonable event count
                            group_events['control'].append(event_count)
                            reasoning.append(f"Control group events: {event_count}")
                except (ValueError, IndexError):
                    continue
        
        # If we have comparison patterns (X vs Y), assign to groups
        comparison_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s+(?:vs\.?|versus|compared\s+(?:with|to))\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+)\s+(?:\(\s*\d+(?:\.\d+)?%\s*\))?\s+(?:vs\.?|versus)\s+(\d+)\s+(?:\(\s*\d+(?:\.\d+)?%\s*\))?',
        ]
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if '%' in match.group(0):
                        # Handle percentage comparison
                        percent1 = float(match.group(1))
                        percent2 = float(match.group(2))
                        
                        # Convert to absolute if we have group sizes
                        if 'intervention' in group_sizes and 'control' in group_sizes:
                            event1 = int((percent1 / 100) * group_sizes['intervention'])
                            event2 = int((percent2 / 100) * group_sizes['control'])
                            group_events['intervention'].append(event1)
                            group_events['control'].append(event2)
                            reasoning.append(f"Comparison: {percent1}% vs {percent2}% = ~{event1} vs {event2} events")
                    else:
                        # Handle absolute comparison
                        event1 = int(match.group(1))
                        event2 = int(match.group(2))
                        
                        # First number typically intervention, second control
                        if event1 not in group_events['intervention'] and event1 not in group_events['control']:
                            group_events['intervention'].append(event1)
                        if event2 not in group_events['intervention'] and event2 not in group_events['control']:
                            group_events['control'].append(event2)
                            
                        reasoning.append(f"Comparison pattern: {event1} vs {event2}")
                except (ValueError, IndexError):
                    continue
        
        # Assign remaining unassigned events to unknown
        assigned_events = set(group_events['intervention'] + group_events['control'])
        for event in event_numbers:
            if event not in assigned_events and event < 1000:  # Reasonable event count
                group_events['unknown'].append(event)
        
        return group_events
    
    def _extract_sample_size(self, text: str, reasoning: List[str]) -> Optional[int]:
        """Extract total sample size with high accuracy - ENHANCED"""
        
        sample_size_candidates = []
        
        for pattern, priority, context in self.sample_size_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    size = int(match.group(1))
                    # Reasonable sample size range
                    if 10 <= size <= 100000:  # Reasonable sample size range
                        sample_size_candidates.append((size, priority, context))
                        reasoning.append(f"Sample size candidate: {size} ({context})")
                except (ValueError, IndexError):
                    continue
        
        # Return highest priority, most reasonable sample size
        if sample_size_candidates:
            # Sort by priority (descending) then by size (descending for total N)
            sample_size_candidates.sort(key=lambda x: (-x[1], -x[0]))
            return sample_size_candidates[0][0]
        
        return None
    
    def _extract_group_sample_sizes(self, text: str, reasoning: List[str]) -> Dict[str, int]:
        """Extract sample sizes for each group - ENHANCED"""
        
        group_sizes = {}
        
        # Intervention group size
        intervention_patterns = [
            r'(?:intervention|treatment|active)\s+(?:group|arm)\s*(?:\(\s*)?(?:n\s*=\s*)?(\d+)',
            r'(?:n\s*=\s*)?(\d+)\s+(?:in\s+the\s+)?(?:intervention|treatment|active)\s+(?:group|arm)',
            r'(\d+)\s+(?:patients?|participants?)\s+(?:were\s+)?(?:randomized|assigned)\s+to\s+(?:receive\s+)?(?:intervention|treatment|active)',
        ]
        
        for pattern in intervention_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    size = int(match.group(1))
                    if 10 <= size <= 10000:  # Reasonable group size
                        group_sizes['intervention'] = size
                        reasoning.append(f"Intervention group size: {size}")
                        break
                except (ValueError, IndexError):
                    continue
        
        # Control group size
        control_patterns = [
            r'(?:control|placebo|standard)\s+(?:group|arm)\s*(?:\(\s*)?(?:n\s*=\s*)?(\d+)',
            r'(?:n\s*=\s*)?(\d+)\s+(?:in\s+the\s+)?(?:control|placebo|standard)\s+(?:group|arm)',
            r'(\d+)\s+(?:patients?|participants?)\s+(?:were\s+)?(?:randomized|assigned)\s+to\s+(?:receive\s+)?(?:placebo|standard\s+care)',
        ]
        
        for pattern in control_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    size = int(match.group(1))
                    if 10 <= size <= 10000:  # Reasonable group size
                        group_sizes['control'] = size
                        reasoning.append(f"Control group size: {size}")
                        break
                except (ValueError, IndexError):
                    continue
        
        return group_sizes
    
    def _identify_study_design(self, text: str) -> str:
        """Identify study design type"""
        
        if re.search(r'\b(?:randomized|randomised|RCT|clinical\s+trial)\b', text, re.IGNORECASE):
            return 'RCT'
        elif re.search(r'\b(?:cohort|prospective|longitudinal)\b', text, re.IGNORECASE):
            return 'cohort'
        elif re.search(r'\b(?:case[-\s]control|retrospective)\b', text, re.IGNORECASE):
            return 'case_control'
        elif re.search(r'\b(?:cross[-\s]sectional|survey)\b', text, re.IGNORECASE):
            return 'cross_sectional'
        elif re.search(r'\b(?:systematic\s+review|meta[-\s]analysis)\b', text, re.IGNORECASE):
            return 'systematic_review'
        else:
            return 'unclear'
    
    def _calculate_extraction_confidence(self, primary_detected: bool, 
                                       total_numbers: int, ratios: int, percentages: int) -> float:
        """Calculate confidence in event extraction - ENHANCED"""
        
        confidence = 0.0
        
        # Base confidence from primary outcome detection
        if primary_detected:
            confidence += 0.4
        
        # Confidence from number of event numbers found
        if total_numbers >= 4:
            confidence += 0.3
        elif total_numbers >= 2:
            confidence += 0.2
        elif total_numbers >= 1:
            confidence += 0.1
        
        # Confidence from ratio patterns (n/N format)
        if ratios >= 2:
            confidence += 0.2
        elif ratios >= 1:
            confidence += 0.1
        
        # Confidence from percentage reporting (NEW)
        if percentages >= 2:
            confidence += 0.2
        elif percentages >= 1:
            confidence += 0.1
        
        # Boost for clear results section
        confidence += 0.1  # Always some base confidence
        
        return min(confidence, 1.0)
    
    def _calculate_group_confidence(self, groups_clear: bool, 
                                  intervention_events: int, control_events: int) -> float:
        """Calculate confidence in group assignment"""
        
        if groups_clear and intervention_events > 0 and control_events > 0:
            return 0.9
        elif groups_clear and (intervention_events > 0 or control_events > 0):
            return 0.7
        elif intervention_events > 0 or control_events > 0:
            return 0.5
        elif groups_clear:
            return 0.3
        else:
            return 0.1
    
    def _get_detected_patterns(self, text: str, primary_detected: bool, 
                             event_count: int, percent_count: int) -> List[str]:
        """Get detected patterns for transparency - ENHANCED"""
        
        patterns = []
        
        if primary_detected:
            patterns.append("primary_outcome: events_detected")
        if event_count >= 4:
            patterns.append("event_numbers: multiple_found")
        elif event_count >= 1:
            patterns.append("event_numbers: some_found")
        if percent_count >= 2:
            patterns.append("percentages: multiple_found")
        elif percent_count >= 1:
            patterns.append("percentages: some_found")
        
        if re.search(r'\d+\s*/\s*\d+', text):
            patterns.append("ratios: n_over_N_format")
        if re.search(r'(\d+(?:\.\d+)?)\s*%\s+(?:vs\.?|versus)\s+(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE):
            patterns.append("comparison: percent_vs_percent")
        if re.search(r'\b(?:intervention|treatment).*?\d+.*?(?:control|placebo).*?\d+\b', text, re.IGNORECASE):
            patterns.append("groups: comparison_detected")
        if re.search(r'\bfutility\b', text, re.IGNORECASE):
            patterns.append("futility: mentioned")
        
        return patterns[:6]  # Limit for readability
    
    def _classify_event_count(self, features: EventCountFeatures, 
                            reasoning: List[str]) -> EventCountResult:
        """
        Classify event count with improved accuracy - FIXED VERSION
        
        CLINICAL HEURISTIC: <30 events in either group = INSUFFICIENT POWER
        """
        
        # STEP 1: Check if we have ANY evidence of event reporting
        has_event_evidence = (
            features.primary_events_detected or 
            features.event_numbers_found or 
            features.percentage_events_found or
            features.statistical_results_found or
            features.results_section_present
        )
        
        if not has_event_evidence:
            reasoning.append("No event reporting evidence found in text")
            return self._create_result(
                EventCountCode.EVENT_COUNT_UNCLEAR,
                0.95,
                "No event counts or percentages found in abstract",
                features,
                reasoning,
                "Cannot assess statistical power"
            )
        
        # STEP 2: Extract the most reliable event counts
        reliable_event_counts = self._extract_reliable_event_counts(features, reasoning)
        
        # STEP 3: If no reliable counts but we have percentages and sample sizes, convert
        if not reliable_event_counts and features.percentage_events_found:
            converted_counts = self._convert_percentages_to_counts(features, reasoning)
            if converted_counts:
                reliable_event_counts = converted_counts
        
        # STEP 4: If still no reliable counts but we have evidence of results
        if not reliable_event_counts:
            # Check if this might be a study with results reported differently
            if features.statistical_results_found or features.confidence_intervals_found:
                reasoning.append("Statistical results found but event counts not extractable")
                return self._create_result(
                    EventCountCode.EVENT_COUNT_UNCLEAR,
                    0.7,
                    "Results reported but specific event counts unclear",
                    features,
                    reasoning,
                    "Cannot determine per-group event counts for power assessment"
                )
            else:
                reasoning.append("Some event evidence but cannot determine group-specific counts")
                return self._create_result(
                    EventCountCode.EVENT_COUNT_UNCLEAR,
                    0.85,
                    "Event reporting present but group assignment unclear",
                    features,
                    reasoning,
                    "Cannot assess per-group statistical power"
                )
        
        # STEP 5: Apply the <30 events heuristic with high precision
        min_group_events = min(reliable_event_counts)
        max_group_events = max(reliable_event_counts)
        
        reasoning.append(f"Reliable event counts identified: {reliable_event_counts}")
        reasoning.append(f"Minimum group events: {min_group_events}")
        
        # STEP 6: Apply clinical decision rule
        if min_group_events < 30:
            # INSUFFICIENT EVENTS - MAJOR ISSUE
            
            # Calculate confidence based on extraction quality
            confidence = self._calculate_insufficient_confidence(features, min_group_events, reasoning)
            
            # Determine severity of issue
            if min_group_events < 10:
                severity = "SEVERE"
                message = f"Severely underpowered: minimum {min_group_events} events (rule of thumb: need ≥30)"
            elif min_group_events < 20:
                severity = "MAJOR"
                message = f"Major power issue: minimum {min_group_events} events (rule of thumb: need ≥30)"
            else:
                severity = "MODERATE"
                message = f"Borderline power issue: minimum {min_group_events} events (rule of thumb: need ≥30)"
            
            reasoning.append(f"{severity} statistical power issue detected")
            
            # Check for supporting evidence of power issues
            power_issue_evidence = []
            if features.futility_mentioned:
                power_issue_evidence.append("futility mentioned")
            if features.early_termination_mentioned:
                power_issue_evidence.append("early termination")
            if features.low_event_rate_indicators:
                power_issue_evidence.append("low event rate language")
            if features.small_study_indicators:
                power_issue_evidence.append("small study indicators")
            
            if power_issue_evidence:
                reasoning.append(f"Supporting evidence: {', '.join(power_issue_evidence)}")
                confidence = min(confidence + 0.05, 0.98)  # Boost confidence slightly
            
            interpretation = (f"Minimum {min_group_events} events indicates insufficient statistical "
                            f"power to detect clinically meaningful effects. Wide confidence intervals "
                            f"and potential for chance findings.")
            
            return self._create_result(
                EventCountCode.INSUFFICIENT_EVENTS,
                confidence,
                message,
                features,
                reasoning,
                interpretation
            )
        
        else:
            # ADEQUATE EVENTS
            reasoning.append(f"Adequate statistical power: minimum {min_group_events} events ≥30")
            
            # Calculate confidence based on extraction quality
            confidence = self._calculate_adequate_confidence(features, min_group_events, reasoning)
            
            interpretation = (f"Adequate event counts (minimum {min_group_events} events) provide "
                            f"sufficient statistical power for reliable effect estimation.")
            
            return self._create_result(
                EventCountCode.ADEQUATE_EVENTS,
                confidence,
                f"Adequate statistical power: minimum {min_group_events} events per group",
                features,
                reasoning,
                interpretation
            )
    
    def _extract_reliable_event_counts(self, features: EventCountFeatures, 
                                     reasoning: List[str]) -> List[int]:
        """Extract most reliable event counts for classification - ENHANCED"""
        
        reliable_counts = []
        
        # PRIORITY 1: Clear per-group event counts
        if features.events_per_group_clear:
            if features.group_event_counts['intervention']:
                reliable_counts.extend(features.group_event_counts['intervention'])
            if features.group_event_counts['control']:
                reliable_counts.extend(features.group_event_counts['control'])
            
            if reliable_counts:
                reasoning.append(f"Using per-group event counts: {reliable_counts}")
                # Remove duplicates and unreasonable values
                reliable_counts = list(set([c for c in reliable_counts if 0 < c < 10000]))
                return reliable_counts
        
        # PRIORITY 2: Primary outcome events (if we have comparison pattern)
        if features.primary_outcome_numbers and len(features.primary_outcome_numbers) >= 2:
            reasoning.append(f"Using primary outcome event counts: {features.primary_outcome_numbers}")
            return features.primary_outcome_numbers[:2]  # Take first two as likely comparison
        
        # PRIORITY 3: Results numbers with comparison pattern
        if len(features.results_numbers) >= 2:
            # Look for reasonable event counts (not sample sizes)
            reasonable_events = [n for n in features.results_numbers if 1 <= n <= 1000]
            if len(reasonable_events) >= 2:
                reasoning.append(f"Using results comparison numbers: {reasonable_events[:2]}")
                return reasonable_events[:2]
        
        # PRIORITY 4: Try to infer from ratios
        if features.extracted_ratios:
            ratio_events = []
            for ratio in features.extracted_ratios:
                try:
                    parts = ratio.split('/')
                    if len(parts) == 2:
                        events = int(parts[0])
                        total = int(parts[1])
                        if 0 < events < total:  # Sanity check
                            ratio_events.append(events)
                except:
                    continue
            
            if len(ratio_events) >= 2:
                reasoning.append(f"Using ratio-derived event counts: {ratio_events[:2]}")
                return ratio_events[:2]
        
        # PRIORITY 5: Use any reasonable event numbers if we have evidence of groups
        if features.intervention_vs_control_clear and features.event_numbers_found:
            # Filter for reasonable event counts
            reasonable_events = [n for n in features.event_numbers_found 
                               if 1 <= n <= (features.total_sample_size or 1000)]
            
            if len(reasonable_events) >= 2:
                # Take smallest reasonable counts as likely per-group events
                reasonable_events.sort()
                reasoning.append(f"Using inferred group events: {reasonable_events[:2]}")
                return reasonable_events[:2]
        
        return []
    
    def _convert_percentages_to_counts(self, features: EventCountFeatures, 
                                    reasoning: List[str]) -> List[int]:
        """Convert percentages to absolute counts where possible"""
        
        converted_counts = []
        
        # If we have group sample sizes, convert percentages
        if features.group_sample_sizes and features.percentage_events_found:
            intervention_size = features.group_sample_sizes.get('intervention')
            control_size = features.group_sample_sizes.get('control')
            
            # Try to match percentages to groups
            if len(features.percentage_events_found) >= 2 and intervention_size and control_size:
                # Assume first percentage is intervention, second is control
                int_events = int((features.percentage_events_found[0] / 100) * intervention_size)
                ctrl_events = int((features.percentage_events_found[1] / 100) * control_size)
                
                converted_counts = [int_events, ctrl_events]
                reasoning.append(f"Converted percentages to events: {features.percentage_events_found[0]}% = {int_events}, {features.percentage_events_found[1]}% = {ctrl_events}")
        
        # If we have total sample size, convert overall percentages
        elif features.total_sample_size and features.percentage_events_found:
            for percent in features.percentage_events_found[:2]:  # Take first two
                if 0 < percent <= 100:
                    events = int((percent / 100) * features.total_sample_size)
                    converted_counts.append(events)
                    reasoning.append(f"Converted {percent}% to {events} events (total N={features.total_sample_size})")
        
        return converted_counts
    
    def _calculate_insufficient_confidence(self, features: EventCountFeatures, 
                                         min_events: int, reasoning: List[str]) -> float:
        """Calculate confidence for insufficient events classification"""
        base_confidence = 0.8
        # Boost confidence based on extraction quality
        if features.event_extraction_confidence >= 0.8:
            base_confidence += 0.10
        elif features.event_extraction_confidence >= 0.6:
            base_confidence += 0.05
        
        # Boost confidence based on group assignment quality
        if features.group_assignment_confidence >= 0.8:
            base_confidence += 0.05
        
        # Boost confidence for very low event counts (more obvious)
        if min_events < 15:
            base_confidence += 0.05
        elif min_events < 5:
            base_confidence += 0.10
        
        # Boost confidence if primary outcome events were detected
        if features.primary_events_detected:
            base_confidence += 0.03
        
        return min(base_confidence, 0.98)  # Cap at 98%
    
    def _calculate_adequate_confidence(self, features: EventCountFeatures, 
                                     min_events: int, reasoning: List[str]) -> float:
        """Calculate confidence for adequate events classification"""
        
        base_confidence = 0.75  # Base confidence for ≥30 rule
        
        # Boost confidence based on extraction quality
        if features.event_extraction_confidence >= 0.8:
            base_confidence += 0.10
        elif features.event_extraction_confidence >= 0.6:
            base_confidence += 0.05
        
        # Boost confidence based on group assignment quality
        if features.group_assignment_confidence >= 0.8:
            base_confidence += 0.08
        
        # Boost confidence for clearly adequate event counts
        if min_events >= 50:
            base_confidence += 0.05
        elif min_events >= 100:
            base_confidence += 0.08
        
        # Boost confidence if primary outcome events were detected
        if features.primary_events_detected:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)  # Cap at 95%
    
    def _create_result(self, code: EventCountCode, confidence: float, 
                      message: str, features: EventCountFeatures, 
                      reasoning: List[str], interpretation_assessment: str) -> EventCountResult:
        """Create an EventCountResult object"""
        return EventCountResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code],
            interpretation_assessment=interpretation_assessment,
            recommended_actions=self.recommended_actions[code]
        )
    
    def _create_empty_features(self) -> EventCountFeatures:
        """Create empty features object for error cases"""
        return EventCountFeatures(
            primary_events_detected=False, event_numbers_found=[], group_event_counts={},
            total_events_reported=None, events_per_group_clear=False, percentage_events_found=[],
            total_sample_size=None, group_sample_sizes={}, overall_event_rate=None,
            results_section_present=False, statistical_results_found=[], confidence_intervals_found=[],
            p_values_found=[], power_analysis_mentioned=False, sample_size_justification=False,
            early_termination_mentioned=False, futility_mentioned=False, low_event_rate_indicators=[],
            rare_event_language=[], small_study_indicators=[], primary_outcome_numbers=[],
            results_numbers=[], extracted_ratios=[], study_design_type='unclear',
            intervention_vs_control_clear=False, follow_up_duration_mentioned=False,
            event_extraction_confidence=0.0, group_assignment_confidence=0.0,
            detected_patterns=[], extracted_event_info=[]
        )


def run_check(abstract: str):
    """Wrapper method to run event count check - IMPROVED"""
    classifier = EventCountClassifier()
    result = classifier.check_event_count(abstract)
    return result


# Test function for debugging
def test_event_classifier(abstract_text: str):
    """Test the event classifier with debug output"""
    print("=" * 60)
    print("TESTING EVENT COUNT CLASSIFIER")
    print("=" * 60)
    
    classifier = EventCountClassifier()
    result = classifier.check_event_count(abstract_text)
    
    print(f"\n🔍 CLASSIFICATION RESULT:")
    print(f"Code: {result.code}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Message: {result.message}")
    
    print(f"\n📊 EXTRACTED FEATURES:")
    print(f"Primary events detected: {result.features.primary_events_detected}")
    print(f"Event numbers found: {result.features.event_numbers_found}")
    print(f"Percentage events found: {result.features.percentage_events_found}")
    print(f"Group event counts: {result.features.group_event_counts}")
    print(f"Sample sizes: {result.features.group_sample_sizes}")
    print(f"Results section present: {result.features.results_section_present}")
    
    print(f"\n💭 REASONING:")
    for i, reason in enumerate(result.reasoning, 1):
        print(f"{i}. {reason}")
    
    print(f"\n🔬 DETECTED PATTERNS:")
    for pattern in result.features.detected_patterns:
        print(f"- {pattern}")
    
    print(f"\n📋 EXTRACTED INFO:")
    for info in result.features.extracted_event_info:
        print(f"- {info}")
    
    return result
abst='''
Abstract
Background
Tirzepatide and semaglutide are highly effective medications for obesity management. The efficacy and safety of tirzepatide as compared with semaglutide in adults with obesity but without type 2 diabetes is unknown.
Methods
In this phase 3b, open-label, controlled trial, adult participants with obesity but without type 2 diabetes were randomly assigned in a 1:1 ratio to receive the maximum tolerated dose of tirzepatide (10 mg or 15 mg) or the maximum tolerated dose of semaglutide (1.7 mg or 2.4 mg) subcutaneously once weekly for 72 weeks. The primary end point was the percent change in weight from baseline to week 72. Key secondary end points included weight reductions of at least 10%, 15%, 20%, and 25% and a change in waist circumference from baseline to week 72.

Research Summary
Tirzepatide vs. Semaglutide for the Treatment of Obesity
Results

    Results
A total of 751 participants underwent randomization. The least-squares mean percent change in weight at week 72 was −20.2% (95% confidence interval [CI], −21.4 to −19.1) with tirzepatide and −13.7% (95% CI, −14.9 to −12.6) with semaglutide (P<0.001). The least-squares mean change in waist circumference was −18.4 cm (95% CI, −19.6 to −17.2) with tirzepatide and −13.0 cm (95% CI, −14.3 to −11.7) with semaglutide (P<0.001). Participants in the tirzepatide group were more likely than those in the semaglutide group to have weight reductions of at least 10%, 15%, 20%, and 25%. The most common adverse events in both treatment groups were gastrointestinal, and most were mild to moderate in severity and occurred primarily during dose escalation.
Conclusions
Among participants with obesity but without diabetes, treatment with tirzepatide was superior to treatment with semaglutide with respect to reduction in body weight and waist circumference at week 72. (Funded by Eli Lilly; SURMOUNT-5 ClinicalTrials.gov number, NCT05822830.)
'''
test_result = test_event_classifier(abst)