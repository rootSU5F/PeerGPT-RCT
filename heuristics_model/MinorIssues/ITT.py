import re
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class ITTAnalysisCode(Enum):
    """ITT analysis classification codes"""
    ITT_ANALYSIS_MENTIONED = 0        # ITT analysis explicitly mentioned
    NON_ITT_ANALYSIS = 1             # Per-protocol or other non-ITT analysis
    ANALYSIS_NOT_MENTIONED = 2       # No analysis approach mentioned (MINOR ISSUE)
    ANALYSIS_UNCLEAR = 3             # Analysis approach mentioned but unclear


@dataclass
class ITTFeatures:
    """Container for ITT analysis features"""
    # Analysis type detection
    itt_explicitly_mentioned: bool           # "intention-to-treat" found
    itt_abbreviation_found: bool             # "ITT" found
    per_protocol_mentioned: bool             # Per-protocol analysis mentioned
    modified_itt_mentioned: bool             # Modified ITT mentioned
    
    # Analysis quality indicators
    analysis_approach_score: int             # Evidence of analysis approach reporting
    itt_score: int                          # Evidence of ITT analysis
    non_itt_score: int                      # Evidence of non-ITT analysis
    
    # Study context
    trial_type_context: str                 # Superiority vs non-inferiority context
    dropout_handling_mentioned: bool        # Dropout handling described
    missing_data_approach: List[str]        # Missing data handling approaches
    
    # Extracted information
    analysis_statements: List[str]           # Raw analysis-related sentences
    analysis_populations: List[str]         # Analysis populations mentioned
    
    # Text characteristics
    statistical_methods_detail: float       # How detailed statistical methods are
    analysis_reporting_quality: str         # Quality of analysis reporting
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_analysis_info: List[str]


@dataclass
class ITTAnalysisResult:
    code: ITTAnalysisCode
    confidence: float
    message: str
    features: ITTFeatures
    reasoning: List[str]
    clinical_implications: str
    analysis_assessment: str


class ITTAnalysisClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_itt_patterns()
        self._setup_clinical_implications()
    
    def _setup_itt_patterns(self):
        """Define patterns for ITT analysis detection - HIGHEST ACCURACY"""
        
        # ITT analysis patterns (POSITIVE indicators)
        self.itt_patterns = [
            # Explicit ITT mentions (highest priority)
            (r'\bintention[-\s]?to[-\s]?treat\b', 3),
            (r'\bintention\s+to\s+treat\b', 3),
            (r'\bITT\s+(?:analysis|approach|principle|population|basis)\b', 3),
            (r'\bITT\b(?:\s+analysis)?', 3),
            
            # ITT analysis descriptions
            (r'\banalysis\s+(?:was\s+)?(?:performed\s+)?(?:on\s+)?(?:an\s+)?intention[-\s]?to[-\s]?treat\s+basis\b', 3),
            (r'\banalyses?\s+(?:were\s+)?(?:conducted\s+)?(?:using\s+)?(?:the\s+)?intention[-\s]?to[-\s]?treat\s+(?:principle|approach|method)\b', 3),
            (r'\bprimary\s+analysis\s+(?:was\s+)?intention[-\s]?to[-\s]?treat\b', 3),
            (r'\bITT\s+analysis\s+(?:was\s+)?(?:performed|conducted|used)\b', 3),
            
            # ITT population mentions
            (r'\bintention[-\s]?to[-\s]?treat\s+population\b', 3),
            (r'\bITT\s+population\b', 3),
            (r'\ball\s+randomized\s+(?:patients?|participants?|subjects?)\s+(?:were\s+)?(?:included\s+)?(?:in\s+the\s+)?analysis\b', 2),
            (r'\ball\s+randomized\s+(?:patients?|participants?|subjects?)\s+analyzed\b', 2),
            
            # Modified ITT mentions
            (r'\bmodified\s+intention[-\s]?to[-\s]?treat\b', 2),
            (r'\bmodified\s+ITT\b', 2),
            (r'\bmITT\b', 2),
            
            # ITT principle descriptions
            (r'\b(?:patients?|participants?|subjects?)\s+(?:were\s+)?analyzed\s+(?:in\s+)?(?:the\s+)?(?:group|arm)\s+(?:to\s+)?which\s+they\s+were\s+(?:originally\s+)?(?:randomized|assigned)\b', 2),
            (r'\banalyzed\s+(?:according\s+to\s+)?(?:original\s+)?(?:randomized\s+)?(?:treatment\s+)?assignment\b', 2),
            (r'\bregardless\s+of\s+(?:treatment\s+)?(?:compliance|adherence|protocol\s+violations)\b', 2),
            
            # Full analysis set mentions
            (r'\bfull\s+analysis\s+set\b', 2),
            (r'\bFAS\b', 1)
        ]
        
        # Non-ITT analysis patterns (NEGATIVE indicators)
        self.non_itt_patterns = [
            # Per-protocol analysis (explicit non-ITT)
            (r'\bper[-\s]?protocol\s+analysis\b', 3),
            (r'\bper[-\s]?protocol\s+population\b', 3),
            (r'\bPP\s+analysis\b', 3),
            (r'\bPP\s+population\b', 3),
            (r'\bprotocol\s+compliant\s+(?:patients?|participants?|subjects?)\b', 2),
            
            # Completers-only analysis
            (r'\bcompleters?[-\s]?only\s+analysis\b', 3),
            (r'\bcompleted\s+(?:the\s+)?(?:study|protocol|treatment)\s+(?:were\s+)?analyzed\b', 2),
            (r'\bonly\s+(?:patients?|participants?|subjects?)\s+(?:who\s+)?completed\s+(?:the\s+)?(?:study|protocol|treatment)\b', 2),
            
            # As-treated analysis
            (r'\bas[-\s]?treated\s+analysis\b', 3),
            (r'\banalyzed\s+(?:according\s+to\s+)?(?:actual\s+)?treatment\s+received\b', 2),
            
            # Evaluable patients analysis
            (r'\bevaluable\s+(?:patients?|participants?|subjects?)\s+(?:only\s+)?(?:were\s+)?analyzed\b', 2),
            (r'\banalysis\s+(?:was\s+)?(?:restricted\s+)?(?:to\s+)?evaluable\s+(?:patients?|participants?|subjects?)\b', 2),
            
            # Exclusion-based analysis
            (r'\b(?:patients?|participants?|subjects?)\s+(?:with\s+)?(?:major\s+)?protocol\s+violations?\s+(?:were\s+)?excluded\s+from\s+analysis\b', 2),
            (r'\bexcluded\s+(?:patients?|participants?|subjects?)\s+(?:who\s+)?(?:did\s+not\s+complete|withdrew|dropped\s+out)\b', 1)
        ]
        
        # Analysis approach indicators (general)
        self.analysis_approach_patterns = [
            # Statistical analysis mentions
            (r'\bstatistical\s+analysis\b', 1),
            (r'\bdata\s+analysis\b', 1),
            (r'\banalysis\s+(?:was\s+)?(?:performed|conducted|carried\s+out)\b', 1),
            (r'\bprimary\s+analysis\b', 2),
            (r'\banalysis\s+(?:population|set|approach|method)\b', 2),
            
            # Analysis populations
            (r'\banalysis\s+population\b', 2),
            (r'\bsafety\s+population\b', 1),
            (r'\befficacy\s+population\b', 1),
            
            # Missing data handling
            (r'\bmissing\s+data\b', 1),
            (r'\bdropout\s+(?:handling|analysis)\b', 1),
            (r'\blast\s+observation\s+carried\s+forward\b', 1),
            (r'\bLOCF\b', 1)
        ]
        
        # Trial type context patterns
        self.trial_type_patterns = [
            # Superiority trials (ITT is ideal)
            (r'\bsuperiority\s+trial\b', 'superiority'),
            (r'\bsuperiority\s+study\b', 'superiority'),
            (r'\bsuperiority\s+design\b', 'superiority'),
            
            # Non-inferiority trials (ITT not ideal)
            (r'\bnon[-\s]?inferiority\s+trial\b', 'non-inferiority'),
            (r'\bnon[-\s]?inferiority\s+study\b', 'non-inferiority'),
            (r'\bnon[-\s]?inferiority\s+design\b', 'non-inferiority'),
            (r'\bnon[-\s]?inferiority\s+(?:margin|analysis)\b', 'non-inferiority'),
            
            # Equivalence trials
            (r'\bequivalence\s+trial\b', 'equivalence'),
            (r'\bequivalence\s+study\b', 'equivalence'),
            (r'\bbioequivalence\s+study\b', 'equivalence')
        ]
        
        # Missing data/dropout handling patterns
        self.missing_data_patterns = [
            (r'\blast\s+observation\s+carried\s+forward\b', 'LOCF'),
            (r'\bLOCF\b', 'LOCF'),
            (r'\bmultiple\s+imputation\b', 'multiple imputation'),
            (r'\bmixed[-\s]?effects?\s+(?:models?|regression)\b', 'mixed effects'),
            (r'\bMMRM\b', 'MMRM'),
            (r'\bmissing\s+at\s+random\b', 'MAR'),
            (r'\bMAR\b', 'MAR'),
            (r'\bmissing\s+(?:completely\s+)?at\s+random\b', 'MCAR'),
            (r'\bsensitivity\s+analysis\b', 'sensitivity analysis')
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            ITTAnalysisCode.ITT_ANALYSIS_MENTIONED: 
                "Intention-to-treat analysis enhances trial validity by including all randomized "
                "participants regardless of treatment compliance or protocol deviations. This "
                "provides a conservative estimate of treatment effect and better reflects "
                "real-world effectiveness.",
                
            ITTAnalysisCode.NON_ITT_ANALYSIS:
                "Non-ITT analysis approach (e.g., per-protocol) may overestimate treatment effects "
                "by excluding non-compliant participants. While appropriate for some research "
                "questions, it may not reflect real-world effectiveness and can introduce bias.",
                
            ITTAnalysisCode.ANALYSIS_NOT_MENTIONED:
                "ITT is the ideal approach for superiority trials but not for non-inferiority trials. "
                "Lack of clarity on the analytic approach limits interpretability. The analysis "
                "method significantly affects result interpretation and should be clearly specified.",
                
            ITTAnalysisCode.ANALYSIS_UNCLEAR:
                "Analysis approach mentioned but unclear. Ambiguous analysis reporting limits "
                "proper interpretation of results and assessment of potential bias. The specific "
                "analysis population and handling of missing data should be clearly defined."
        }
    
    def check_itt_analysis(self, text: str) -> ITTAnalysisResult:
        """
        Check if ITT analysis is mentioned and assess appropriateness
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            ITTAnalysisResult with ITT analysis assessment
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                ITTAnalysisCode.ANALYSIS_NOT_MENTIONED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "Cannot assess analysis approach"
            )
        
        # Extract comprehensive ITT features
        features = self._extract_itt_features(text, reasoning)
        
        # Calculate pattern scores
        itt_score = self._score_patterns(text, self.itt_patterns)
        non_itt_score = self._score_patterns(text, self.non_itt_patterns)
        analysis_approach_score = self._score_patterns(text, self.analysis_approach_patterns)
        
        features.itt_score = itt_score
        features.non_itt_score = non_itt_score
        features.analysis_approach_score = analysis_approach_score
        
        reasoning.append(f"Pattern scores - ITT: {itt_score}, Non-ITT: {non_itt_score}, "
                        f"Analysis approach: {analysis_approach_score}")
        
        # Analyze trial type context
        trial_context = self._analyze_trial_context(text, reasoning)
        features.trial_type_context = trial_context
        
        # Assess analysis approach
        analysis_assessment = self._assess_analysis_approach(features, reasoning)
        
        # Make ITT classification
        return self._make_itt_classification(features, reasoning, analysis_assessment)
    
    def _extract_itt_features(self, text: str, reasoning: List[str]) -> ITTFeatures:
        """Extract comprehensive ITT analysis features"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Check for explicit ITT mentions
        itt_explicitly_mentioned = bool(re.search(
            r'\bintention[-\s]?to[-\s]?treat\b', text, re.IGNORECASE
        ))
        
        itt_abbreviation_found = bool(re.search(
            r'\bITT\b', text
        ))
        
        per_protocol_mentioned = bool(re.search(
            r'\bper[-\s]?protocol\b', text, re.IGNORECASE
        ))
        
        modified_itt_mentioned = bool(re.search(
            r'\b(?:modified\s+(?:intention[-\s]?to[-\s]?treat|ITT)|mITT)\b', text, re.IGNORECASE
        ))
        
        # Check for dropout handling
        dropout_handling_mentioned = bool(re.search(
            r'\b(?:dropout|withdrawal|missing\s+data|protocol\s+violation|non[-\s]?compliance)\b', 
            text, re.IGNORECASE
        ))
        
        # Extract missing data approaches
        missing_data_approach = []
        for pattern, approach in self.missing_data_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                missing_data_approach.append(approach)
        
        # Extract analysis-related sentences
        analysis_statements = self._extract_analysis_statements(text)
        
        # Extract analysis populations mentioned
        analysis_populations = []
        population_patterns = [
            r'\bintention[-\s]?to[-\s]?treat\s+population\b',
            r'\bITT\s+population\b',
            r'\bper[-\s]?protocol\s+population\b',
            r'\bfull\s+analysis\s+set\b',
            r'\bsafety\s+population\b',
            r'\befficacy\s+population\b'
        ]
        
        for pattern in population_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis_populations.extend(matches)
        
        # Calculate statistical methods detail
        statistical_terms = len(re.findall(
            r'\b(?:analysis|statistical|method|approach|population|intent|protocol|randomized)\w*\b', 
            clean_text
        ))
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        statistical_methods_detail = statistical_terms / total_words if total_words > 0 else 0
        
        # Assess analysis reporting quality
        analysis_reporting_quality = self._assess_analysis_reporting_quality(text, itt_explicitly_mentioned, 
                                                                           per_protocol_mentioned, analysis_populations)
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)
        
        # Create extracted analysis info
        extracted_analysis_info = []
        if itt_explicitly_mentioned or itt_abbreviation_found:
            extracted_analysis_info.append("ITT analysis mentioned")
        if per_protocol_mentioned:
            extracted_analysis_info.append("Per-protocol analysis mentioned")
        if modified_itt_mentioned:
            extracted_analysis_info.append("Modified ITT mentioned")
        if missing_data_approach:
            extracted_analysis_info.append(f"Missing data approaches: {', '.join(missing_data_approach)}")
        
        if itt_explicitly_mentioned or per_protocol_mentioned:
            reasoning.append(f"Analysis approach indicators found: ITT={itt_explicitly_mentioned}, PP={per_protocol_mentioned}")
        else:
            reasoning.append("No clear analysis approach indicators found")
        
        return ITTFeatures(
            itt_explicitly_mentioned=itt_explicitly_mentioned,
            itt_abbreviation_found=itt_abbreviation_found,
            per_protocol_mentioned=per_protocol_mentioned,
            modified_itt_mentioned=modified_itt_mentioned,
            analysis_approach_score=0,          # Will be set later
            itt_score=0,
            non_itt_score=0,
            trial_type_context="",              # Will be set later
            dropout_handling_mentioned=dropout_handling_mentioned,
            missing_data_approach=missing_data_approach,
            analysis_statements=analysis_statements,
            analysis_populations=analysis_populations,
            statistical_methods_detail=statistical_methods_detail,
            analysis_reporting_quality=analysis_reporting_quality,
            detected_patterns=detected_patterns,
            extracted_analysis_info=extracted_analysis_info
        )
    
    def _extract_analysis_statements(self, text: str) -> List[str]:
        """Extract sentences containing analysis information"""
        statements = []
        
        # Split into sentences and find analysis-related ones
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if re.search(r'\b(?:analysis|intent|protocol|ITT|randomized|statistical|population)\b', 
                        sentence.lower()):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 15:  # Avoid very short fragments
                    statements.append(clean_sentence[:200])  # Truncate long sentences
        
        return statements[:3]  # Limit to first 3 relevant sentences
    
    def _assess_analysis_reporting_quality(self, text: str, itt_mentioned: bool, 
                                         pp_mentioned: bool, populations: List[str]) -> str:
        """Assess quality of analysis approach reporting"""
        
        if itt_mentioned and populations:
            return "detailed"
        elif itt_mentioned or pp_mentioned:
            return "moderate"
        elif re.search(r'\banalysis\s+(?:was\s+)?(?:performed|conducted)\b', text, re.IGNORECASE):
            return "basic"
        else:
            return "none"
    
    def _analyze_trial_context(self, text: str, reasoning: List[str]) -> str:
        """Analyze trial type context for ITT appropriateness"""
        
        for pattern, trial_type in self.trial_type_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if trial_type == 'superiority':
                    reasoning.append("Superiority trial context - ITT analysis is ideal")
                    return "Superiority trial (ITT ideal)"
                elif trial_type == 'non-inferiority':
                    reasoning.append("Non-inferiority trial context - ITT analysis not ideal")
                    return "Non-inferiority trial (ITT not ideal)"
                elif trial_type == 'equivalence':
                    reasoning.append("Equivalence trial context - per-protocol may be preferred")
                    return "Equivalence trial (PP may be preferred)"
        
        reasoning.append("Trial type context unclear")
        return "Trial type unclear"
    
    def _assess_analysis_approach(self, features: ITTFeatures, reasoning: List[str]) -> str:
        """Assess the analysis approach and its appropriateness"""
        
        itt_mentioned = features.itt_explicitly_mentioned or features.itt_abbreviation_found
        pp_mentioned = features.per_protocol_mentioned
        trial_context = features.trial_type_context
        
        if itt_mentioned and not pp_mentioned:
            if "Superiority" in trial_context:
                assessment = "APPROPRIATE: ITT analysis for superiority trial"
            elif "Non-inferiority" in trial_context:
                assessment = "QUESTIONABLE: ITT analysis for non-inferiority trial (may not be ideal)"
            else:
                assessment = "GOOD: ITT analysis mentioned (generally preferred)"
        elif pp_mentioned and not itt_mentioned:
            if "Non-inferiority" in trial_context or "Equivalence" in trial_context:
                assessment = "APPROPRIATE: Per-protocol analysis for non-inferiority/equivalence trial"
            else:
                assessment = "CONCERNING: Per-protocol analysis (may overestimate effects)"
        elif itt_mentioned and pp_mentioned:
            assessment = "COMPREHENSIVE: Both ITT and per-protocol analyses mentioned"
        else:
            assessment = "UNCLEAR: No clear analysis approach specified"
        
        reasoning.append(f"Analysis assessment: {assessment}")
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
            ('itt', self.itt_patterns),
            ('non_itt', self.non_itt_patterns),
            ('analysis_approach', self.analysis_approach_patterns)
        ]
        
        for category, patterns in all_pattern_sets:
            for pattern, score in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(f"{category}: {pattern[:50]}...")
        
        return detected[:6]  # Limit for readability
    
    def _make_itt_classification(self, features: ITTFeatures, 
                               reasoning: List[str], analysis_assessment: str) -> ITTAnalysisResult:
        """Make ITT analysis classification"""
        
        itt_score = features.itt_score
        non_itt_score = features.non_itt_score
        analysis_approach_score = features.analysis_approach_score
        itt_mentioned = features.itt_explicitly_mentioned or features.itt_abbreviation_found
        pp_mentioned = features.per_protocol_mentioned
        
        # ITT ANALYSIS CLASSIFICATION LOGIC
        
        # 1. Clear ITT analysis mentioned
        if itt_mentioned and itt_score >= 3:
            if non_itt_score > itt_score:
                # Both ITT and non-ITT mentioned, non-ITT stronger
                reasoning.append(f"Both ITT and non-ITT mentioned, non-ITT score higher ({non_itt_score} > {itt_score})")
                return self._create_result(
                    ITTAnalysisCode.NON_ITT_ANALYSIS,
                    0.8,
                    f"Mixed analysis approaches with non-ITT predominant",
                    features,
                    reasoning,
                    analysis_assessment
                )
            else:
                # ITT analysis clearly mentioned
                reasoning.append(f"ITT analysis clearly mentioned (score: {itt_score})")
                return self._create_result(
                    ITTAnalysisCode.ITT_ANALYSIS_MENTIONED,
                    0.95,
                    f"Intention-to-treat analysis mentioned",
                    features,
                    reasoning,
                    analysis_assessment
                )
        
        # 2. Clear non-ITT analysis mentioned
        elif pp_mentioned and non_itt_score >= 3:
            reasoning.append(f"Non-ITT analysis mentioned (score: {non_itt_score})")
            return self._create_result(
                ITTAnalysisCode.NON_ITT_ANALYSIS,
                0.9,
                f"Per-protocol or non-ITT analysis mentioned",
                features,
                reasoning,
                analysis_assessment
            )
        
        # 3. Some ITT indicators but not strong
        elif itt_mentioned or itt_score >= 2:
            reasoning.append(f"Some ITT indicators (mentioned: {itt_mentioned}, score: {itt_score})")
            return self._create_result(
                ITTAnalysisCode.ITT_ANALYSIS_MENTIONED,
                0.7,
                f"ITT analysis likely mentioned",
                features,
                reasoning,
                analysis_assessment
            )
        
        # 4. Some analysis approach mentioned but unclear
        elif analysis_approach_score >= 2 or features.analysis_reporting_quality in ['basic', 'moderate']:
            reasoning.append(f"Some analysis approach mentioned but unclear (score: {analysis_approach_score})")
            return self._create_result(
                ITTAnalysisCode.ANALYSIS_UNCLEAR,
                0.7,
                f"Analysis approach mentioned but unclear",
                features,
                reasoning,
                analysis_assessment
            )
        
        # 5. Default: No analysis approach mentioned - MINOR ISSUE
        else:
            reasoning.append("No clear analysis approach mentioned")
            return self._create_result(
                ITTAnalysisCode.ANALYSIS_NOT_MENTIONED,
                0.9,
                f"Analysis approach not mentioned",
                features,
                reasoning,
                analysis_assessment
            )
    
    def _create_result(self, code: ITTAnalysisCode, confidence: float, 
                      message: str, features: ITTFeatures, 
                      reasoning: List[str], analysis_assessment: str) -> ITTAnalysisResult:
        """Create an ITTAnalysisResult object"""
        return ITTAnalysisResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code],
            analysis_assessment=analysis_assessment
        )
    
    def _create_empty_features(self) -> ITTFeatures:
        """Create empty features object for error cases"""
        return ITTFeatures(
            itt_explicitly_mentioned=False, itt_abbreviation_found=False,
            per_protocol_mentioned=False, modified_itt_mentioned=False,
            analysis_approach_score=0, itt_score=0, non_itt_score=0,
            trial_type_context="", dropout_handling_mentioned=False,
            missing_data_approach=[], analysis_statements=[], analysis_populations=[],
            statistical_methods_detail=0, analysis_reporting_quality="none",
            detected_patterns=[], extracted_analysis_info=[]
        )




def run_check(abstract : str):# just a wrapper method
    classifier = ITTAnalysisClassifier()
    result = classifier.check_itt_analysis(abstract)
    return result