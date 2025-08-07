import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class CompositeOutcomeCode(Enum):
    """Composite outcome classification codes"""
    NO_COMPOSITE_OUTCOME = 0           # Single, clear primary outcome - Good
    COMPOSITE_OUTCOME_PRESENT = 1      # Composite outcome used - MAJOR ISSUE (interpretation difficulties)
    OUTCOME_UNCLEAR = 2                # Primary outcome not clearly defined


@dataclass
class CompositeOutcomeFeatures:
    """Container for composite outcome features"""
    # Primary outcome detection
    primary_outcome_mentioned: bool           # Primary outcome explicitly mentioned
    primary_outcome_text: List[str]          # Extracted primary outcome descriptions
    primary_outcome_count: int               # Number of primary outcomes mentioned
    
    # Composite outcome indicators
    composite_keywords_found: List[str]      # Composite-indicating keywords found
    composite_patterns_detected: List[str]   # Specific composite patterns detected
    component_count_estimate: int            # Estimated number of outcome components
    
    # Outcome component analysis
    detected_components: List[str]           # Individual outcome components found
    component_types: List[str]              # Types of outcomes (mortality, morbidity, etc.)
    clinical_significance_varies: bool       # Evidence that components vary in significance
    
    # Multiple endpoint indicators
    multiple_endpoints_mentioned: bool       # Multiple endpoints explicitly mentioned
    secondary_outcomes_present: bool         # Secondary outcomes mentioned
    endpoint_hierarchy_clear: bool           # Clear hierarchy of outcomes
    
    # Linguistic patterns
    conjunction_density: float               # Density of "and/or" connectors in outcome context
    outcome_complexity_score: float         # Overall complexity of outcome description
    
    # Statistical considerations
    multiple_testing_mentioned: bool        # Multiple testing/corrections mentioned
    alpha_adjustment_mentioned: bool        # Alpha adjustment for multiple comparisons
    
    # Text characteristics
    outcome_description_length: int          # Length of outcome descriptions
    outcome_specificity_score: float        # How specific/detailed outcome descriptions are
    
    # Context analysis
    methods_section_found: bool              # Methods section identified
    outcome_context_analysis: str            # Context where outcomes were found
    
    # Detected patterns for transparency
    detected_patterns: List[str]
    extracted_outcome_info: List[str]


@dataclass
class CompositeOutcomeResult:
    code: CompositeOutcomeCode
    confidence: float
    message: str
    features: CompositeOutcomeFeatures
    reasoning: List[str]
    clinical_implications: str
    interpretation_assessment: str


class CompositeOutcomeClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_composite_patterns()
        self._setup_clinical_implications()
    
    def _setup_composite_patterns(self):
        """Define patterns for composite outcome detection - IMPROVED ACCURACY"""
        
        # Direct composite outcome indicators (highest priority)
        self.composite_indicators = [
            # Explicit composite mentions (priority 4)
            (r'\bcomposite\s+(?:primary\s+)?outcome\b', 4, 'explicit_composite'),
            (r'\bcomposite\s+(?:end\s*)?point\b', 4, 'explicit_composite'),
            (r'\bcomposite\s+measure\b', 3, 'composite_measure'),
            (r'\bcombined\s+(?:primary\s+)?outcome\b', 3, 'combined_outcome'),
            (r'\bcombined\s+(?:end\s*)?point\b', 3, 'combined_endpoint'),
            
            # Multiple component patterns (priority 3)
            (r'\bprimary\s+outcome\s+(?:was\s+|were\s+)?(?:a\s+)?composite\b', 4, 'primary_composite'),
            (r'\bprimary\s+(?:end\s*)?point\s+(?:was\s+|were\s+)?(?:a\s+)?composite\b', 4, 'primary_composite'),
            (r'\b(?:the\s+)?composite\s+(?:of\s+)?(?:death|mortality)\s+(?:and|or)\b', 3, 'composite_with_death'),
            
            # MACE and specific composite patterns (priority 3)
            (r'\b(?:major\s+adverse\s+)?(?:cardiac|cardiovascular)\s+events?\b', 3, 'MACE'),
            (r'\bMACE\b', 4, 'MACE_acronym'),
            (r'\b(?:major\s+adverse\s+)?(?:cardiac|cardiovascular)\s+and\s+cerebrovascular\s+events?\b', 3, 'MACCE'),
            
            # Time-to-event composite patterns (priority 3)
            (r'\btime\s+to\s+(?:first\s+)?(?:occurrence\s+of\s+)?(?:any\s+of\s+the\s+following|composite)\b', 3, 'time_to_composite'),
            (r'\bfirst\s+occurrence\s+of\s+(?:any\s+of\s+)?(?:the\s+following|death|mortality)\s+(?:and|or)\b', 3, 'first_occurrence_composite'),
            
            # Multiple primary outcomes (priority 2)
            (r'\b(?:two|three|four|multiple)\s+primary\s+outcomes?\b', 3, 'numbered_primary'),
            (r'\bco-primary\s+outcomes?\b', 3, 'co_primary'),
        ]
        
        # Primary outcome extraction patterns - CONTEXT AWARE
        self.primary_outcome_patterns = [
            # Standard primary outcome mentions (priority 3)
            (r'\bprimary\s+outcome\s+(?:was\s+|were\s+)?([^.;]+?)(?:\.|;|Results|Conclusions)', 3, 'primary_outcome'),
            (r'\bprimary\s+(?:end\s*)?point\s+(?:was\s+|were\s+)?([^.;]+?)(?:\.|;|Results|Conclusions)', 3, 'primary_endpoint'),
            (r'\bmain\s+outcome\s+(?:measure\s+)?(?:was\s+|were\s+)?([^.;]+?)(?:\.|;|Results|Conclusions)', 2, 'main_outcome'),
            
            # Primary outcome in methods (priority 2)
            (r'Methods.*?primary\s+outcome\s+(?:was\s+|were\s+)?([^.;]+?)(?:\.|;)', 2, 'methods_primary'),
            (r'Methods.*?primary\s+(?:end\s*)?point\s+(?:was\s+|were\s+)?([^.;]+?)(?:\.|;)', 2, 'methods_endpoint'),
        ]
        
        # Single outcome recognition patterns (NEW)
        self.single_outcome_patterns = [
            (r'\bprimary\s+outcome\s+was\s+([^,]+?)(?:\.|,\s*with|$)', 'single_primary'),
            (r'\bprimary\s+endpoint\s+was\s+([^,]+?)(?:\.|,\s*with|$)', 'single_endpoint'),
            (r'\bprimary\s+outcome\s+was\s+(all-cause\s+mortality|death\s+from\s+any\s+cause|overall\s+survival)', 'single_mortality'),
        ]
        
        # Component counting patterns - MORE RESTRICTIVE
        self.component_indicators = [
            # Death/mortality components
            (r'\b(?:death|mortality|fatal)\b', 'mortality'),
            (r'\b(?:cardiovascular|cardiac)\s+death\b', 'cv_death'),
            (r'\ball[-\s]cause\s+(?:death|mortality)\b', 'all_cause_death'),
            
            # Cardiovascular events
            (r'\bmyocardial\s+infarction\b|\bMI\b', 'myocardial_infarction'),
            (r'\b(?:acute\s+)?coronary\s+syndrome\b|\bACS\b', 'acs'),
            (r'\bstroke\b|\bcerebrovascular\s+accident\b|\bCVA\b', 'stroke'),
            (r'\b(?:heart|cardiac)\s+failure\b|\bHF\b', 'heart_failure'),
            
            # Procedures/interventions
            (r'\brevascularization\b', 'revascularization'),
            (r'\bcoronary\s+(?:artery\s+)?bypass\b|\bCABG\b', 'cabg'),
            (r'\b(?:percutaneous\s+)?coronary\s+intervention\b|\bPCI\b', 'pci'),
            
            # Hospitalizations
            (r'\bhospitalization\b|\bhospital\s+admission\b', 'hospitalization'),
            (r'\b(?:cardiovascular|cardiac)[-\s]related\s+hospitalization\b', 'cv_hospitalization'),
            (r'\b(?:heart\s+failure|HF)[-\s]related\s+hospitalization\b', 'hf_hospitalization'),
        ]
        
        # Clinical significance variation indicators
        self.significance_variation_patterns = [
            # Explicit mentions of varying importance
            (r'\b(?:components?\s+)?(?:vary|varied|varying)\s+in\s+(?:clinical\s+)?(?:significance|importance)\b', 3),
            (r'\b(?:different|varying)\s+(?:clinical\s+)?(?:significance|importance|relevance)\b', 2),
            (r'\b(?:more|less)\s+(?:clinically\s+)?(?:significant|important|relevant)\b', 2),
            
            # Hard vs soft endpoint mixing
            (r'\b(?:hard|soft)\s+(?:endpoints?|outcomes?)\b', 2),
            (r'\b(?:mortality|death).*?(?:and|or).*?(?:hospitalization|admission)\b', 2),
            (r'\b(?:fatal|death).*?(?:and|or).*?(?:non[-\s]fatal|nonfatal)\b', 2),
        ]
        
        # Multiple testing/statistical adjustment patterns
        self.statistical_adjustment_patterns = [
            (r'\bmultiple\s+(?:testing|comparisons?)\b', 2),
            (r'\bBonferroni\s+(?:correction|adjustment)\b', 3),
            (r'\balpha\s+(?:adjustment|correction)\b', 2),
            (r'\b(?:adjusted|corrected)\s+(?:p[-\s]values?|significance)\b', 2),
            (r'\b(?:family[-\s]wise|experiment[-\s]wise)\s+error\s+rate\b', 2),
            (r'\bHolm[-\s]Bonferroni\b', 3),
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            CompositeOutcomeCode.NO_COMPOSITE_OUTCOME:
                "Single, well-defined primary outcome provides clear, interpretable results. "
                "This approach minimizes statistical complexity and allows for straightforward "
                "clinical interpretation of treatment effects.",
                
            CompositeOutcomeCode.COMPOSITE_OUTCOME_PRESENT:
                "The use of a composite outcome introduces two key limitations. First, if the "
                "components vary in clinical significance, the overall results become harder to "
                "interpret. Second, composite outcomes are driven by the most frequent event, "
                "which is often the event of least clinical importance.",
                
            CompositeOutcomeCode.OUTCOME_UNCLEAR:
                "Primary outcome not clearly defined or multiple competing outcomes mentioned. "
                "This ambiguity makes it difficult to assess the clinical significance and "
                "interpret the study findings appropriately."
        }
    
    def check_composite_outcome(self, text: str) -> CompositeOutcomeResult:
        """
        Analyze primary outcome to detect composite outcomes - IMPROVED VERSION
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            CompositeOutcomeResult with composite outcome assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                CompositeOutcomeCode.OUTCOME_UNCLEAR,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess outcome structure"
            )
        
        # Extract sections for context-aware analysis
        sections = self._extract_sections(text)
        
        # Extract comprehensive composite outcome features
        features = self._extract_composite_features(text, sections, reasoning)
        
        # Analyze outcome complexity
        complexity_analysis = self._analyze_outcome_complexity(features, reasoning)
        
        # Assess clinical significance variation
        significance_assessment = self._assess_significance_variation(features, reasoning)
        
        # Make composite outcome classification
        return self._make_composite_classification(features, reasoning, significance_assessment)
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections of the abstract for context-aware analysis"""
        sections = {}
        
        # Extract Methods section
        methods_match = re.search(r'Methods.*?(?=Results|Conclusions|$)', text, re.IGNORECASE | re.DOTALL)
        if methods_match:
            sections['methods'] = methods_match.group(0)
        
        # Extract Results section
        results_match = re.search(r'Results.*?(?=Conclusions|$)', text, re.IGNORECASE | re.DOTALL)
        if results_match:
            sections['results'] = results_match.group(0)
        
        # Extract Background/Introduction
        background_match = re.search(r'(?:Background|Introduction).*?(?=Methods|Results|$)', text, re.IGNORECASE | re.DOTALL)
        if background_match:
            sections['background'] = background_match.group(0)
        
        return sections
    
    def _extract_composite_features(self, text: str, sections: Dict[str, str], reasoning: List[str]) -> CompositeOutcomeFeatures:
        """Extract comprehensive composite outcome features - CONTEXT AWARE"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Focus on Methods section for primary outcome analysis
        outcome_analysis_text = sections.get('methods', text)
        methods_section_found = 'methods' in sections
        
        # Extract primary outcome mentions - CONTEXT AWARE
        primary_outcome_mentioned = False
        primary_outcome_text = []
        primary_outcome_count = 0
        outcome_context_analysis = "Not found"
        
        # First check for single outcome patterns
        single_outcome_found = False
        for pattern, context in self.single_outcome_patterns:
            matches = re.finditer(pattern, outcome_analysis_text, re.IGNORECASE)
            for match in matches:
                single_outcome_found = True
                primary_outcome_mentioned = True
                primary_outcome_count += 1
                try:
                    outcome_text = match.group(1).strip()
                    if len(outcome_text) > 5:  # Meaningful outcome description
                        primary_outcome_text.append(outcome_text)
                        outcome_context_analysis = f"Single outcome in {context}"
                        reasoning.append(f"Single primary outcome detected: {outcome_text[:50]}...")
                except (IndexError, AttributeError):
                    continue
        
        # If no single outcome found, look for general patterns
        if not single_outcome_found:
            for pattern, priority, context in self.primary_outcome_patterns:
                matches = re.finditer(pattern, outcome_analysis_text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    primary_outcome_mentioned = True
                    primary_outcome_count += 1
                    try:
                        outcome_text = match.group(1).strip()
                        if len(outcome_text) > 10:  # Meaningful outcome description
                            primary_outcome_text.append(outcome_text)
                            outcome_context_analysis = f"General pattern in {context}"
                    except (IndexError, AttributeError):
                        continue
        
        # Detect composite indicators - ONLY in Methods section
        composite_keywords_found = []
        composite_patterns_detected = []
        
        for pattern, priority, context in self.composite_indicators:
            matches = re.finditer(pattern, outcome_analysis_text, re.IGNORECASE)
            for match in matches:
                composite_keywords_found.append(match.group(0))
                composite_patterns_detected.append(context)
        
        # Count outcome components - ONLY in primary outcome text
        detected_components = []
        component_types = []
        
        # Only analyze components within primary outcome descriptions
        primary_outcome_combined = ' '.join(primary_outcome_text)
        if primary_outcome_combined:
            for pattern, component_type in self.component_indicators:
                matches = re.findall(pattern, primary_outcome_combined, re.IGNORECASE)
                if matches:
                    detected_components.extend(matches)
                    component_types.append(component_type)
        
        component_count_estimate = len(set(component_types))
        
        # Check for clinical significance variation - ONLY in Methods
        clinical_significance_varies = False
        for pattern, priority in self.significance_variation_patterns:
            if re.search(pattern, outcome_analysis_text, re.IGNORECASE):
                clinical_significance_varies = True
                break
        
        # Check for multiple endpoints - BE MORE RESTRICTIVE
        multiple_endpoints_mentioned = False
        # Only count if explicitly mentioned in Methods as PRIMARY outcomes
        if re.search(r'\b(?:multiple|two|three|four)\s+primary\s+(?:outcomes?|endpoints?)\b', 
                    outcome_analysis_text, re.IGNORECASE):
            multiple_endpoints_mentioned = True
        
        # Secondary outcomes - only flag if mentioned in Methods as affecting primary analysis
        secondary_outcomes_present = bool(re.search(
            r'Methods.*?secondary\s+(?:outcomes?|endpoints?)\b',
            text, re.IGNORECASE | re.DOTALL
        ))
        
        endpoint_hierarchy_clear = bool(re.search(
            r'\bprimary\s+(?:outcome|endpoint).*?secondary\s+(?:outcomes?|endpoints?)\b',
            outcome_analysis_text, re.IGNORECASE | re.DOTALL
        ))
        
        # Calculate linguistic patterns - ONLY within primary outcome context
        conjunction_count = 0
        total_words = 0
        
        if primary_outcome_text:
            for outcome_desc in primary_outcome_text:
                conjunction_count += len(re.findall(r'\b(?:and|or)\b', outcome_desc, re.IGNORECASE))
                doc_outcome = self.nlp(outcome_desc)
                total_words += len([t for t in doc_outcome if not t.is_space and not t.is_punct])
        
        conjunction_density = conjunction_count / max(total_words, 1)
        
        # Calculate outcome complexity score
        outcome_complexity_score = self._calculate_complexity_score(
            primary_outcome_text, composite_keywords_found, component_count_estimate
        )
        
        # Check for statistical considerations
        multiple_testing_mentioned = False
        alpha_adjustment_mentioned = False
        
        for pattern, priority in self.statistical_adjustment_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if 'multiple' in pattern or 'family' in pattern or 'experiment' in pattern:
                    multiple_testing_mentioned = True
                if 'alpha' in pattern or 'Bonferroni' in pattern or 'Holm' in pattern:
                    alpha_adjustment_mentioned = True
        
        # Calculate outcome description characteristics
        outcome_description_length = sum(len(desc) for desc in primary_outcome_text)
        outcome_specificity_score = self._calculate_specificity_score(primary_outcome_text)
        
        # Get detected patterns for transparency
        detected_patterns = self._get_detected_patterns(clean_text, composite_patterns_detected)
        
        # Create extracted outcome info
        extracted_outcome_info = []
        if primary_outcome_mentioned:
            extracted_outcome_info.append(f"Primary outcome mentioned: {primary_outcome_count} times")
        if composite_keywords_found:
            extracted_outcome_info.append(f"Composite indicators: {len(set(composite_keywords_found))}")
        if component_count_estimate > 1:
            extracted_outcome_info.append(f"Components in primary outcome: {component_count_estimate}")
        if single_outcome_found:
            extracted_outcome_info.append("Single outcome pattern detected")
        
        # Update reasoning with context-aware findings
        if single_outcome_found:
            reasoning.append("Single, well-defined primary outcome identified")
        elif composite_keywords_found:
            reasoning.append(f"Composite indicators found: {composite_keywords_found[:3]}")
        elif component_count_estimate > 1:
            reasoning.append(f"Multiple outcome components in primary outcome: {component_count_estimate}")
        elif not primary_outcome_mentioned:
            reasoning.append("No clear primary outcome statement found")
        
        return CompositeOutcomeFeatures(
            primary_outcome_mentioned=primary_outcome_mentioned,
            primary_outcome_text=primary_outcome_text,
            primary_outcome_count=primary_outcome_count,
            composite_keywords_found=composite_keywords_found,
            composite_patterns_detected=composite_patterns_detected,
            component_count_estimate=component_count_estimate,
            detected_components=detected_components,
            component_types=component_types,
            clinical_significance_varies=clinical_significance_varies,
            multiple_endpoints_mentioned=multiple_endpoints_mentioned,
            secondary_outcomes_present=secondary_outcomes_present,
            endpoint_hierarchy_clear=endpoint_hierarchy_clear,
            conjunction_density=conjunction_density,
            outcome_complexity_score=outcome_complexity_score,
            multiple_testing_mentioned=multiple_testing_mentioned,
            alpha_adjustment_mentioned=alpha_adjustment_mentioned,
            outcome_description_length=outcome_description_length,
            outcome_specificity_score=outcome_specificity_score,
            methods_section_found=methods_section_found,
            outcome_context_analysis=outcome_context_analysis,
            detected_patterns=detected_patterns,
            extracted_outcome_info=extracted_outcome_info
        )
    
    def _calculate_complexity_score(self, primary_texts: List[str], 
                                  composite_keywords: List[str], 
                                  component_count: int) -> float:
        """Calculate overall outcome complexity score"""
        score = 0.0
        
        # Base score from component count - ONLY if components found in primary outcome
        if component_count >= 4:
            score += 0.4
        elif component_count >= 3:
            score += 0.3
        elif component_count >= 2:
            score += 0.2
        
        # Score from composite keywords
        score += min(len(composite_keywords) * 0.15, 0.4)
        
        # Score from primary outcome text complexity - ONLY conjunctions in outcome text
        for text in primary_texts:
            and_or_count = len(re.findall(r'\b(?:and|or)\b', text, re.IGNORECASE))
            score += min(and_or_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_specificity_score(self, primary_texts: List[str]) -> float:
        """Calculate how specific/detailed outcome descriptions are"""
        if not primary_texts:
            return 0.0
        
        total_specificity = 0.0
        for text in primary_texts:
            # Longer descriptions tend to be more specific
            length_score = min(len(text.split()) / 20.0, 0.5)
            
            # Medical terminology indicates specificity
            medical_terms = len(re.findall(
                r'\b(?:cardiovascular|myocardial|cerebrovascular|thrombotic|ischemic|hemorrhagic)\b',
                text, re.IGNORECASE
            ))
            medical_score = min(medical_terms * 0.1, 0.3)
            
            # Time specifications indicate specificity
            time_specs = len(re.findall(
                r'\b(?:90[-\s]day|30[-\s]day|in[-\s]hospital|during\s+hospitalization)\b',
                text, re.IGNORECASE
            ))
            time_score = min(time_specs * 0.1, 0.2)
            
            total_specificity += length_score + medical_score + time_score
        
        return min(total_specificity / len(primary_texts), 1.0)
    
    def _analyze_outcome_complexity(self, features: CompositeOutcomeFeatures, 
                                  reasoning: List[str]) -> str:
        """Analyze outcome complexity for interpretation assessment"""
        
        complexity_score = features.outcome_complexity_score
        component_count = features.component_count_estimate
        
        if complexity_score >= 0.6:
            analysis = f"HIGH complexity (score: {complexity_score:.2f}) - multiple components with conjunction patterns"
        elif complexity_score >= 0.3:
            analysis = f"MODERATE complexity (score: {complexity_score:.2f}) - some composite indicators"
        elif component_count >= 2:
            analysis = f"MILD complexity - {component_count} components detected in primary outcome"
        else:
            analysis = "LOW complexity - single outcome pattern"
        
        reasoning.append(f"Outcome complexity: {analysis}")
        return analysis
    
    def _assess_significance_variation(self, features: CompositeOutcomeFeatures, 
                                     reasoning: List[str]) -> str:
        """Assess potential for clinical significance variation"""
        
        component_types = set(features.component_types)
        
        # Check for mixing of different clinical significance levels
        has_mortality = any('death' in comp or 'mortality' in comp for comp in component_types)
        has_hospitalization = 'hospitalization' in component_types
        has_procedures = any('pci' in comp or 'cabg' in comp or 'revascularization' in comp 
                           for comp in component_types)
        
        if features.clinical_significance_varies:
            assessment = "EXPLICIT mention of varying clinical significance"
        elif has_mortality and (has_hospitalization or has_procedures):
            assessment = "LIKELY significance variation (mortality mixed with non-fatal events)"
        elif len(component_types) >= 4:
            assessment = "POSSIBLE significance variation (many diverse components)"
        elif len(component_types) >= 2:
            assessment = "POTENTIAL significance variation (multiple components)"
        else:
            assessment = "NO evidence of significance variation"
        
        reasoning.append(f"Clinical significance assessment: {assessment}")
        return assessment
    
    def _get_detected_patterns(self, text: str, composite_patterns: List[str]) -> List[str]:
        """Get list of detected patterns for transparency"""
        detected = []
        
        # Add composite patterns
        for pattern in set(composite_patterns[:4]):  # Top unique patterns
            detected.append(f"composite: {pattern}")
        
        # Add primary outcome patterns
        if re.search(r'\bprimary\s+outcome\b', text):
            detected.append("primary_outcome: mentioned")
        if re.search(r'\bcomposite\b', text):
            detected.append("composite: keyword_found")
        
        return detected[:6]  # Limit for readability
    
    def _make_composite_classification(self, features: CompositeOutcomeFeatures, 
                                     reasoning: List[str], 
                                     significance_assessment: str) -> CompositeOutcomeResult:
        """Make composite outcome classification - IMPROVED LOGIC"""
        
        # IMPROVED COMPOSITE OUTCOME CLASSIFICATION LOGIC
        
        # 1. Clear single outcome patterns (NEW - highest priority)
        if ("Single outcome" in features.outcome_context_analysis and 
            features.component_count_estimate <= 1 and 
            not features.composite_keywords_found):
            
            reasoning.append("Single, well-defined primary outcome with no composite indicators")
            return self._create_result(
                CompositeOutcomeCode.NO_COMPOSITE_OUTCOME,
                0.95,
                "Single primary outcome - no composite structure detected",
                features,
                reasoning,
                "Single outcome provides clear interpretation"
            )
        
        # 2. Explicit composite outcome indicators
        if features.composite_keywords_found:
            reasoning.append(f"Composite outcome detected: {features.composite_keywords_found[:2]}")
            
            # Calculate confidence based on strength of evidence
            confidence = 0.8
            if any('explicit' in pattern for pattern in features.composite_patterns_detected):
                confidence = 0.95
            elif len(features.composite_keywords_found) >= 2:
                confidence = 0.9
            
            return self._create_result(
                CompositeOutcomeCode.COMPOSITE_OUTCOME_PRESENT,
                confidence,
                f"Composite outcome detected: {len(features.composite_keywords_found)} indicators found",
                features,
                reasoning,
                significance_assessment
            )
        
        # 3. Multiple components in PRIMARY OUTCOME (not just anywhere in text)
        elif features.component_count_estimate >= 3 and features.primary_outcome_mentioned:
            reasoning.append(f"Multiple outcome components in primary outcome: {features.component_count_estimate}")
            
            # Require high conjunction density OR explicit multiple endpoints
            if features.conjunction_density > 0.05 or features.multiple_endpoints_mentioned:
                reasoning.append("High conjunction density or multiple endpoints mentioned")
                return self._create_result(
                    CompositeOutcomeCode.COMPOSITE_OUTCOME_PRESENT,
                    0.75,
                    f"Likely composite outcome: {features.component_count_estimate} components with conjunction patterns",
                    features,
                    reasoning,
                    significance_assessment
                )
            else:
                # Lower confidence if no strong conjunction patterns
                return self._create_result(
                    CompositeOutcomeCode.COMPOSITE_OUTCOME_PRESENT,
                    0.6,
                    f"Possible composite outcome: {features.component_count_estimate} components detected",
                    features,
                    reasoning,
                    significance_assessment
                )
        
        # 4. Multiple primary outcomes explicitly mentioned (not secondary outcomes in results)
        elif features.multiple_endpoints_mentioned and features.primary_outcome_count > 1:
            reasoning.append("Multiple primary outcomes explicitly mentioned")
            return self._create_result(
                CompositeOutcomeCode.COMPOSITE_OUTCOME_PRESENT,
                0.7,
                "Multiple primary outcomes suggest composite structure",
                features,
                reasoning,
                significance_assessment
            )
        
        # 5. Clear single primary outcome
        elif (features.primary_outcome_mentioned and 
              features.component_count_estimate <= 1 and 
              not features.multiple_endpoints_mentioned):
            reasoning.append("Single, well-defined primary outcome")
            return self._create_result(
                CompositeOutcomeCode.NO_COMPOSITE_OUTCOME,
                0.85,
                "Single primary outcome - no composite structure detected",
                features,
                reasoning,
                "Single outcome provides clear interpretation"
            )
        
        # 6. Primary outcome present but with some complexity
        elif features.primary_outcome_mentioned:
            reasoning.append("Primary outcome mentioned but assessing complexity")
            
            # Check if complexity is due to composite structure or just detailed description
            if (features.outcome_complexity_score >= 0.4 and 
                features.component_count_estimate >= 2):
                return self._create_result(
                    CompositeOutcomeCode.COMPOSITE_OUTCOME_PRESENT,
                    0.6,
                    "Primary outcome with multiple components - possible composite",
                    features,
                    reasoning,
                    significance_assessment
                )
            else:
                return self._create_result(
                    CompositeOutcomeCode.NO_COMPOSITE_OUTCOME,
                    0.75,
                    "Single primary outcome - complexity from description detail",
                    features,
                    reasoning,
                    "Single outcome structure"
                )
        
        # 7. Default: Outcome unclear
        else:
            reasoning.append("Primary outcome not clearly described")
            return self._create_result(
                CompositeOutcomeCode.OUTCOME_UNCLEAR,
                0.8,
                "Primary outcome not clearly defined",
                features,
                reasoning,
                "Cannot assess outcome structure"
            )
    
    def _create_result(self, code: CompositeOutcomeCode, confidence: float, 
                      message: str, features: CompositeOutcomeFeatures, 
                      reasoning: List[str], interpretation_assessment: str) -> CompositeOutcomeResult:
        """Create a CompositeOutcomeResult object"""
        return CompositeOutcomeResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code],
            interpretation_assessment=interpretation_assessment
        )
    
    def _create_empty_features(self) -> CompositeOutcomeFeatures:
        """Create empty features object for error cases"""
        return CompositeOutcomeFeatures(
            primary_outcome_mentioned=False, primary_outcome_text=[], primary_outcome_count=0,
            composite_keywords_found=[], composite_patterns_detected=[], component_count_estimate=0,
            detected_components=[], component_types=[], clinical_significance_varies=False,
            multiple_endpoints_mentioned=False, secondary_outcomes_present=False, 
            endpoint_hierarchy_clear=False, conjunction_density=0,
            outcome_complexity_score=0, multiple_testing_mentioned=False, alpha_adjustment_mentioned=False,
            outcome_description_length=0, outcome_specificity_score=0,
            methods_section_found=False, outcome_context_analysis="Not analyzed",
            detected_patterns=[], extracted_outcome_info=[]
        )


def run_check(abstract: str):
    """Wrapper method to run composite outcome check"""
    classifier = CompositeOutcomeClassifier()
    result = classifier.check_composite_outcome(abstract)
    return result


# Test with the bloodstream infection abstract
def test_bloodstream_abstract():
    abstract = """ 
'''
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
    """
    
    result = run_check(abstract)
    print(f"Classification: {result.code}")
    print(f"Confidence: {result.confidence}")
    print(f"Message: {result.message}")
    print(f"Primary outcome text: {result.features.primary_outcome_text}")
    print(f"Component count: {result.features.component_count_estimate}")
    print(f"Composite keywords: {result.features.composite_keywords_found}")
    print(f"Reasoning: {result.reasoning}")
    
    return result

# Uncomment to test:

test_result = test_bloodstream_abstract()