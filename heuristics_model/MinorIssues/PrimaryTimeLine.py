
import re
from unittest import result
import spacy
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np



class TimelineCode(Enum):
    """Primary outcome timeline classification codes"""
    APPROPRIATE_TIMELINE = 0    # >30 days or clearly appropriate timing
    SHORT_TIMELINE = 1         # ≤30 days - Minor issue (context-dependent)
    NOT_MENTIONED = 2          # No timeline specified - Minor issue


@dataclass
class TimelineFeatures:
    """Container for primary outcome timeline features"""
    # Extracted timeline information
    explicit_timeline_days: Optional[float]    # Primary timeline in days
    all_timelines_found: List[float]          # All extracted timelines
    timeline_unit: Optional[str]              # Unit (days, weeks, months, years)
    
    # Pattern scores
    short_timeline_score: int                 # ≤30 days indicators
    appropriate_timeline_score: int           # >30 days indicators
    no_timeline_score: int                   # No timeline mentioned
    
    # Primary outcome detection
    primary_outcome_mentioned: bool           # "primary outcome" explicitly mentioned
    primary_outcome_phrases: List[str]        # Detected primary outcome language
    
    # Clinical context analysis
    condition_type: str                       # Acute vs chronic condition context
    acute_condition_score: int               # Acute setting indicators (stroke, MI, etc.)
    chronic_condition_score: int             # Chronic setting indicators (diabetes, cancer, etc.)
    
    # Timeline appropriateness context
    timeline_justification_mentioned: bool    # Any timing rationale provided
    follow_up_mentions: int                  # Number of follow-up time mentions
    
    # Text characteristics
    temporal_specificity: float              # How specific timeline language is
    outcome_timeline_density: float          # Density of outcome + timeline terms
    
    # Detected patterns
    detected_patterns: List[str]
    extracted_timeline_info: List[str]


@dataclass
class TimelineResult:
    code: TimelineCode
    confidence: float
    message: str
    features: TimelineFeatures
    reasoning: List[str]
    clinical_implications: str
    context_assessment: str


class PrimaryOutcomeTimelineClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_timeline_patterns()
        self._setup_clinical_context_patterns()
        self._setup_clinical_implications()
    
    def _setup_timeline_patterns(self):
        """Define patterns for timeline classification with clinical context"""
        
        # Short timeline patterns (≤30 days) - Context-dependent minor issue
        self.short_timeline_patterns = [
            # Explicit short durations (Strong: 3 points)
            (r'\b(?:[1-9]|[12]\d|30)\s*(?:day|d)s?\b', 3),           # 1-30 days
            (r'\b(?:[1-4])\s*(?:week|wk)s?\b', 3),                   # 1-4 weeks (≤28 days)
            (r'\b(?:1)\s*(?:month|mo)\b', 2),                        # 1 month (~30 days)
            
            # Clinical timing phrases suggesting short assessment (Strong: 3 points)
            (r'\bat\s+(?:discharge|admission|baseline)\b', 3),
            (r'\bimmediate(?:ly)?\s+(?:post|after)\b', 3),
            (r'\b(?:acute|early|immediate)\s+(?:response|outcome|effect)\b', 3),
            (r'\bin-?hospital\s+(?:outcome|mortality|event)\b', 3),
            
            # Short-term assessment phrases (Moderate: 2 points)
            (r'\bwithin\s+(?:[1-9]|[12]\d|30)\s*(?:day|d)s?\b', 2),
            (r'\bafter\s+(?:[1-9]|[12]\d|30)\s*(?:day|d)s?\b', 2),
            (r'\bat\s+(?:[1-4])\s*(?:week|wk)s?\b', 2),
            (r'\bshort-?term\s+(?:outcome|response|effect)\b', 2),
            
            # Hospital stay indicators (Weak: 1 point)
            (r'\bduring\s+(?:hospitalization|admission|stay)\b', 1),
            (r'\bbefore\s+discharge\b', 1),
            (r'\bearly\s+(?:assessment|evaluation|measurement)\b', 1)
        ]
        
        # Appropriate timeline patterns (>30 days)
        self.appropriate_timeline_patterns = [
            # Explicit longer durations (Strong: 3 points)
            (r'\b(?:3[1-9]|\d{2,3})\s*(?:day|d)s?\b', 3),           # >30 days
            (r'\b(?:[2-9]|1[0-9])\s*(?:month|mo)s?\b', 3),          # 2+ months
            (r'\b(?:[1-9])\s*(?:year|yr)s?\b', 3),                  # 1+ years
            (r'\b(?:[5-9]|[1-4]\d)\s*(?:week|wk)s?\b', 3),          # 5+ weeks (>35 days)
            
            # Long-term assessment phrases (Strong: 3 points)
            (r'\blong-?term\s+(?:outcome|follow.?up|survival|effect)\b', 3),
            (r'\b(?:6|12|18|24|36)\s*(?:month|mo)\s+(?:outcome|survival|follow.?up)\b', 3),
            (r'\boverall\s+survival\b', 3),
            (r'\bprogression-?free\s+survival\b', 3),
            (r'\btime\s+to\s+(?:progression|recurrence|death|event)\b', 3),
            
            # Clinical endpoints suggesting longer follow-up (Moderate: 2 points)
            (r'\bmortality\s+(?:at|after)\s+(?:[2-9]|\d{2})\s*(?:month|year)', 2),
            (r'\b(?:disease|cancer|tumor)\s+progression\b', 2),
            (r'\bclinical\s+(?:remission|cure|response)\b', 2),
            (r'\b(?:cardiovascular|cardiac)\s+(?:events?|outcomes?)\b', 2),
            (r'\b(?:sustained|durable)\s+(?:response|remission|effect)\b', 2),
            
            # Follow-up timing indicators (Moderate: 2 points)
            (r'\bfollow.?up\s+(?:at|after|of)\s+(?:[2-9]|\d{2})\s*(?:month|year)', 2),
            (r'\bassessed?\s+(?:at|after)\s+(?:[2-9]|\d{2})\s*(?:month|year)', 2),
            (r'\bmeasured?\s+(?:at|after)\s+(?:[2-9]|\d{2})\s*(?:month|year)', 2),
            
            # General long-term indicators (Weak: 1 point)
            (r'\bchronic\s+(?:outcome|effect|response)\b', 1),
            (r'\bextended\s+(?:follow.?up|observation)\b', 1),
            (r'\blongitudinal\s+(?:outcome|assessment)\b', 1)
        ]
        
        # No timeline mentioned patterns
        self.no_timeline_patterns = [
            # Vague outcome language without timing (Moderate: 2 points)
            (r'\bprimary\s+(?:outcome|endpoint)\s+(?:was|is)\s+(?!.*(?:at|after|within|by))\w+', 2),
            (r'\bmain\s+(?:outcome|endpoint)\s+(?:was|is)\s+(?!.*(?:at|after|within|by))\w+', 2),
            (r'\b(?:efficacy|effectiveness)\s+(?:was\s+)?(?:assessed|measured|evaluated)(?!.*(?:at|after|within|by))', 2),
            
            # Outcome mentions without timing context (Weak: 1 point)
            (r'\bprimary\s+(?:outcome|endpoint)(?!.*(?:at|after|within|by|time|day|week|month|year))', 1),
            (r'\bmain\s+(?:outcome|endpoint)(?!.*(?:at|after|within|by|time|day|week|month|year))', 1),
            (r'\b(?:response|remission|cure)(?!.*(?:at|after|within|by|time|day|week|month|year))', 1)
        ]
        
        # Primary outcome detection patterns
        self.primary_outcome_patterns = [
            (r'\bprimary\s+(?:outcome|endpoint|objective|end.?point)\b', 3),
            (r'\bmain\s+(?:outcome|endpoint|objective)\b', 2),
            (r'\bprincipal\s+(?:outcome|endpoint)\b', 2),
            (r'\bprimary\s+(?:efficacy|safety)\s+(?:outcome|endpoint|measure)\b', 3),
            (r'\bco-?primary\s+(?:outcome|endpoint)\b', 2)
        ]
        
        # Timeline extraction patterns (for numerical analysis)
        self.timeline_extraction_patterns = [
            # Days
            (r'\b(\d+(?:\.\d+)?)\s*(?:day|d)s?\b', 'days'),
            # Weeks  
            (r'\b(\d+(?:\.\d+)?)\s*(?:week|wk)s?\b', 'weeks'),
            # Months
            (r'\b(\d+(?:\.\d+)?)\s*(?:month|mo)s?\b', 'months'),
            # Years
            (r'\b(\d+(?:\.\d+)?)\s*(?:year|yr)s?\b', 'years'),
        ]
    
    def _setup_clinical_context_patterns(self):
        """Define patterns for clinical context analysis"""
        
        # Acute conditions where 30-day outcomes may be appropriate
        self.acute_condition_patterns = [
            # Cardiovascular acute events (Strong: 3 points)
            (r'\b(?:acute\s+)?(?:myocardial\s+infarction|MI|STEMI|NSTEMI)\b', 3),
            (r'\b(?:acute\s+)?(?:stroke|CVA|cerebrovascular\s+accident)\b', 3),
            (r'\b(?:acute\s+)?(?:heart\s+failure|cardiac\s+arrest)\b', 3),
            (r'\bpercutaneous\s+coronary\s+intervention\b', 3),
            (r'\bPCI\b', 2),
            
            # Surgical/procedural contexts (Strong: 3 points)
            (r'\b(?:surgery|surgical|operation|procedure)\b', 2),
            (r'\bpost-?operative\b', 3),
            (r'\bperi-?operative\b', 3),
            (r'\bemergency\s+(?:surgery|procedure|intervention)\b', 3),
            
            # Acute medical conditions (Moderate: 2 points)
            (r'\bacute\s+(?:illness|condition|disease|episode)\b', 2),
            (r'\b(?:sepsis|shock|trauma|pneumonia)\b', 2),
            (r'\bicU\s+(?:admission|stay|patients?)\b', 2),
            (r'\bintensive\s+care\b', 2),
            (r'\bcritical\s+(?:illness|care)\b', 2),
            
            # Emergency/urgent contexts (Weak: 1 point)
            (r'\bemergency\s+(?:department|room|visit)\b', 1),
            (r'\bacute\s+care\b', 1),
            (r'\bhospital\s+admission\b', 1)
        ]
        
        # Chronic conditions where longer follow-up typically needed
        self.chronic_condition_patterns = [
            # Cancer/oncology (Strong: 3 points)
            (r'\b(?:cancer|tumor|tumour|malignancy|neoplasm|oncology)\b', 3),
            (r'\b(?:chemotherapy|radiation|radiotherapy|immunotherapy)\b', 3),
            (r'\b(?:breast|lung|colon|prostate|ovarian|pancreatic)\s+cancer\b', 3),
            (r'\b(?:lymphoma|leukemia|melanoma|sarcoma|carcinoma)\b', 3),
            
            # Chronic diseases (Strong: 3 points)
            (r'\b(?:diabetes|diabetic)\b', 3),
            (r'\b(?:hypertension|chronic\s+kidney\s+disease|CKD)\b', 3),
            (r'\b(?:COPD|chronic\s+obstructive\s+pulmonary)\b', 3),
            (r'\b(?:rheumatoid\s+arthritis|inflammatory\s+bowel\s+disease|IBD)\b', 3),
            (r'\b(?:multiple\s+sclerosis|MS|Parkinson)\b', 3),
            
            # Cardiovascular chronic conditions (Moderate: 2 points)
            (r'\b(?:chronic\s+heart\s+failure|CHF)\b', 2),
            (r'\b(?:coronary\s+artery\s+disease|CAD)\b', 2),
            (r'\b(?:atrial\s+fibrillation|AF)\b', 2),
            (r'\bcardiovascular\s+(?:disease|prevention)\b', 2),
            
            # Chronic treatment contexts (Moderate: 2 points)
            (r'\bchronic\s+(?:treatment|therapy|medication)\b', 2),
            (r'\blong-?term\s+(?:treatment|therapy|management)\b', 2),
            (r'\bmaintenance\s+(?:therapy|treatment)\b', 2),
            
            # General chronic indicators (Weak: 1 point)
            (r'\bchronic\s+(?:condition|disease|illness)\b', 1),
            (r'\bprevention\s+of\b', 1),
            (r'\brisk\s+reduction\b', 1)
        ]
    
    def _setup_clinical_implications(self):
        """Define clinical implications for each classification"""
        self.clinical_implications = {
            TimelineCode.APPROPRIATE_TIMELINE: 
                "Primary outcome timeline appears appropriate (>30 days or suitable for condition type). "
                "Sufficient time for meaningful clinical effects to emerge and be accurately assessed. "
                "Timeline allows for proper evaluation of treatment efficacy and safety.",
                
            TimelineCode.SHORT_TIMELINE:
                "The primary outcome was assessed within a short time frame (≤30 days), which "
                "may be insufficient to observe meaningful clinical effects, particularly for chronic "
                "conditions or interventions with delayed impact. Short timelines risk underestimating "
                "true benefits or harms. Timeline appropriateness should be interpreted in the context "
                "of the disease: while 30-day outcomes may be suitable for acute settings (e.g., stroke, MI), "
                "longer follow-up is often required for interventions in chronic diseases (e.g., diabetes, "
                "cancer, heart failure).",
                
            TimelineCode.NOT_MENTIONED:
                "The timeline for assessing the primary outcome is not clearly stated. In RCTs, the "
                "timing of outcome measurement is critical– assessing outcomes too early may "
                "underestimate treatment effects, while excessively short timelines may miss "
                "delayed benefits or harms. Lack of this information limits interpretability and clinical "
                "relevance."
        }
    
    def check_primary_outcome_timeline(self, text: str) -> TimelineResult:
        """
        Analyze primary outcome timeline with clinical context awareness
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            TimelineResult with context-aware classification
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                TimelineCode.NOT_MENTIONED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "No context available"
            )
        
        # Extract comprehensive features
        features = self._extract_timeline_features(text, reasoning)
        
        # Analyze clinical context
        context_assessment = self._analyze_clinical_context(features, text, reasoning)
        
        # Extract timeline information
        timeline_info = self._extract_timeline_data(text, reasoning)
        features.explicit_timeline_days = timeline_info['primary_timeline']
        features.all_timelines_found = timeline_info['all_timelines']
        features.timeline_unit = timeline_info['primary_unit']
        features.extracted_timeline_info = timeline_info['extracted_strings']
        
        # Calculate pattern scores
        short_score = self._score_patterns(text, self.short_timeline_patterns)
        appropriate_score = self._score_patterns(text, self.appropriate_timeline_patterns)
        no_timeline_score = self._score_patterns(text, self.no_timeline_patterns)
        
        features.short_timeline_score = short_score
        features.appropriate_timeline_score = appropriate_score
        features.no_timeline_score = no_timeline_score
        
        reasoning.append(f"Pattern scores - Short: {short_score}, Appropriate: {appropriate_score}, "
                        f"No timeline: {no_timeline_score}")
        
        # Make context-aware classification
        return self._make_timeline_classification(features, reasoning, context_assessment)
    
    def _extract_timeline_features(self, text: str, reasoning: List[str]) -> TimelineFeatures:
        """Extract comprehensive timeline-related features"""
        
        doc = self.nlp(text)
        clean_text = text.lower().strip()
        
        # Detect primary outcome mentions
        primary_outcome_mentioned = bool(re.search(
            r'\bprimary\s+(?:outcome|endpoint|objective)', clean_text
        ))
        
        # Extract primary outcome phrases
        primary_outcome_phrases = []
        for pattern, score in self.primary_outcome_patterns:
            matches = re.findall(pattern, clean_text, re.IGNORECASE)
            primary_outcome_phrases.extend(matches)
        
        # Analyze clinical context
        acute_score = self._score_patterns(text, self.acute_condition_patterns)
        chronic_score = self._score_patterns(text, self.chronic_condition_patterns)
        
        # Determine condition type
        if acute_score > chronic_score + 2:
            condition_type = "acute"
        elif chronic_score > acute_score + 2:
            condition_type = "chronic"
        elif acute_score > 0 and chronic_score > 0:
            condition_type = "mixed"
        else:
            condition_type = "unclear"
        
        # Check for timeline justification
        timeline_justification = bool(re.search(
            r'\b(?:timing|time|duration|period).*?(?:appropriate|suitable|optimal|critical|important)', 
            clean_text
        ))
        
        # Count follow-up mentions
        follow_up_mentions = len(re.findall(
            r'\bfollow.?up\b', clean_text
        ))
        
        # Calculate temporal specificity
        temporal_terms = len(re.findall(
            r'\b(?:day|week|month|year|time|timing|duration|period|at|after|within|by)\b', 
            clean_text
        ))
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        temporal_specificity = temporal_terms / total_words if total_words > 0 else 0
        
        # Calculate outcome-timeline density
        outcome_timeline_terms = len(re.findall(
            r'\b(?:outcome|endpoint|primary|assess|measure|evaluate).*?(?:day|week|month|year|time|at|after|within)\b|'
            r'\b(?:day|week|month|year|time|at|after|within).*?(?:outcome|endpoint|primary|assess|measure|evaluate)\b', 
            clean_text
        ))
        outcome_timeline_density = outcome_timeline_terms / max(temporal_terms, 1)
        
        # Get detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)
        
        return TimelineFeatures(
            explicit_timeline_days=None,           # Will be set by timeline extraction
            all_timelines_found=[],
            timeline_unit=None,
            short_timeline_score=0,               # Will be set later
            appropriate_timeline_score=0,
            no_timeline_score=0,
            primary_outcome_mentioned=primary_outcome_mentioned,
            primary_outcome_phrases=primary_outcome_phrases[:3],
            condition_type=condition_type,
            acute_condition_score=acute_score,
            chronic_condition_score=chronic_score,
            timeline_justification_mentioned=timeline_justification,
            follow_up_mentions=follow_up_mentions,
            temporal_specificity=temporal_specificity,
            outcome_timeline_density=outcome_timeline_density,
            detected_patterns=detected_patterns,
            extracted_timeline_info=[]
        )
    
    def _extract_timeline_data(self, text: str, reasoning: List[str]) -> Dict:
        """Extract and convert timeline information to days"""
        all_timelines = []
        extracted_strings = []
        primary_timeline = None
        primary_unit = None
        
        # Process each timeline pattern
        for pattern, unit in self.timeline_extraction_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    value = float(match)
                    
                    # Convert to days
                    if unit == 'days':
                        days = value
                    elif unit == 'weeks':
                        days = value * 7
                    elif unit == 'months':
                        days = value * 30.44  # Average days per month
                    elif unit == 'years':
                        days = value * 365.25
                    
                    all_timelines.append(days)
                    extracted_strings.append(f"{match} {unit} ({days:.1f} days)")
                    
                except ValueError:
                    continue
        
        # Find primary timeline (prefer primary outcome context)
        if all_timelines:
            # Look for timeline near "primary outcome" mentions
            primary_outcome_contexts = re.finditer(
                r'\bprimary\s+(?:outcome|endpoint).*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?', 
                text.lower()
            )
            
            for match in primary_outcome_contexts:
                try:
                    value = float(match.group(1))
                    unit = match.group(2)
                    
                    if unit == 'day':
                        primary_timeline = value
                        primary_unit = 'days'
                    elif unit == 'week':
                        primary_timeline = value * 7
                        primary_unit = 'weeks'
                    elif unit == 'month':
                        primary_timeline = value * 30.44
                        primary_unit = 'months'
                    elif unit == 'year':
                        primary_timeline = value * 365.25
                        primary_unit = 'years'
                    break
                except (ValueError, AttributeError):
                    continue
            
            # If no primary outcome context, use the longest timeline as likely primary
            if primary_timeline is None and all_timelines:
                primary_timeline = max(all_timelines)
                # Determine unit for primary timeline
                if primary_timeline <= 7:
                    primary_unit = 'days'
                elif primary_timeline <= 60:
                    primary_unit = 'weeks'
                elif primary_timeline <= 730:
                    primary_unit = 'months'
                else:
                    primary_unit = 'years'
        
        if extracted_strings:
            reasoning.append(f"Extracted timelines: {extracted_strings}")
            if primary_timeline:
                reasoning.append(f"Primary timeline identified: {primary_timeline:.1f} days")
        else:
            reasoning.append("No explicit timelines found")
        
        return {
            'primary_timeline': primary_timeline,
            'all_timelines': all_timelines,
            'primary_unit': primary_unit,
            'extracted_strings': extracted_strings
        }
    
    def _analyze_clinical_context(self, features: TimelineFeatures, text: str, reasoning: List[str]) -> str:
        """Analyze clinical context to determine timeline appropriateness"""
        
        condition_type = features.condition_type
        acute_score = features.acute_condition_score
        chronic_score = features.chronic_condition_score
        
        if condition_type == "acute":
            context = f"Acute clinical context detected (score: {acute_score}). " \
                     f"Short-term outcomes (≤30 days) may be clinically appropriate for acute conditions."
        elif condition_type == "chronic":
            context = f"Chronic clinical context detected (score: {chronic_score}). " \
                     f"Longer follow-up (>30 days) typically needed for chronic conditions."
        elif condition_type == "mixed":
            context = f"Mixed clinical context (acute: {acute_score}, chronic: {chronic_score}). " \
                     f"Timeline appropriateness depends on specific intervention and outcomes."
        else:
            context = f"Clinical context unclear (acute: {acute_score}, chronic: {chronic_score}). " \
                     f"Cannot assess timeline appropriateness without disease context."
        
        reasoning.append(f"Clinical context: {condition_type}")
        return context
    
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
            ('short_timeline', self.short_timeline_patterns),
            ('appropriate_timeline', self.appropriate_timeline_patterns),
            ('no_timeline', self.no_timeline_patterns),
            ('acute_condition', self.acute_condition_patterns),
            ('chronic_condition', self.chronic_condition_patterns)
        ]
        
        for category, patterns in all_pattern_sets:
            for pattern, score in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(f"{category}: {pattern[:40]}...")
        
        return detected[:8]  # Limit for readability
    
    def _make_timeline_classification(self, features: TimelineFeatures, 
                                    reasoning: List[str], context_assessment: str) -> TimelineResult:
        """Make context-aware timeline classification"""
        
        short_score = features.short_timeline_score
        appropriate_score = features.appropriate_timeline_score
        no_timeline_score = features.no_timeline_score
        explicit_timeline = features.explicit_timeline_days
        condition_type = features.condition_type
        
        # CONTEXT-AWARE CLASSIFICATION LOGIC
        
        # 1. Check explicit numerical timeline first (highest confidence)
        if explicit_timeline is not None:
            reasoning.append(f"Using explicit timeline: {explicit_timeline:.1f} days")
            
            if explicit_timeline <= 30:
                # Short timeline - but consider clinical context
                if condition_type == "acute" and features.acute_condition_score >= 3:
                    reasoning.append(f"Short timeline ({explicit_timeline:.1f} days) but acute context - may be appropriate")
                    return self._create_result(
                        TimelineCode.APPROPRIATE_TIMELINE,
                        0.8,  # Lower confidence due to short timeline
                        f"Short timeline ({explicit_timeline:.1f} days) potentially appropriate for acute condition",
                        features,
                        reasoning,
                        context_assessment
                    )
                else:
                    reasoning.append(f"Short timeline ({explicit_timeline:.1f} days) ≤ 30 days - MINOR ISSUE")
                    return self._create_result(
                        TimelineCode.SHORT_TIMELINE,
                        0.9,
                        f"Short primary outcome timeline: {explicit_timeline:.1f} days (≤30 days)",
                        features,
                        reasoning,
                        context_assessment
                    )
            else:  # >30 days
                reasoning.append(f"Appropriate timeline ({explicit_timeline:.1f} days) > 30 days")
                return self._create_result(
                    TimelineCode.APPROPRIATE_TIMELINE,
                    0.95,
                    f"Appropriate timeline: {explicit_timeline:.1f} days (>30 days)",
                    features,
                    reasoning,
                    context_assessment
                )
        
        # 2. No explicit timeline - use pattern analysis
        
        # Check if primary outcome is mentioned at all
        if not features.primary_outcome_mentioned and no_timeline_score < 2:
            reasoning.append("No primary outcome or timeline mentioned")
            return self._create_result(
                TimelineCode.NOT_MENTIONED,
                0.9,
                "Primary outcome timeline not mentioned",
                features,
                reasoning,
                context_assessment
            )
        
        # Strong evidence of no timeline information
        if no_timeline_score >= 3 or (features.primary_outcome_mentioned and no_timeline_score >= 2):
            reasoning.append(f"Primary outcome mentioned but timeline not specified (score: {no_timeline_score})")
            return self._create_result(
                TimelineCode.NOT_MENTIONED,
                0.85,
                "Primary outcome timeline not clearly specified",
                features,
                reasoning,
                context_assessment
            )
        
        # Strong appropriate timeline indicators
        if appropriate_score >= 4 and appropriate_score > short_score + 2:
            reasoning.append(f"Strong appropriate timeline indicators (score: {appropriate_score})")
            return self._create_result(
                TimelineCode.APPROPRIATE_TIMELINE,
                0.8,
                f"Appropriate timeline indicated by patterns (score: {appropriate_score})",
                features,
                reasoning,
                context_assessment
            )
        
        # Strong short timeline indicators - context matters
        if short_score >= 4 and short_score > appropriate_score + 1:
            # Check clinical context for appropriateness
            if condition_type == "acute" and features.acute_condition_score >= 4:
                reasoning.append(f"Short timeline indicators but strong acute context - may be appropriate")
                return self._create_result(
                    TimelineCode.APPROPRIATE_TIMELINE,
                    0.7,
                    f"Short-term assessment potentially appropriate for acute condition (pattern score: {short_score})",
                    features,
                    reasoning,
                    context_assessment
                )
            else:
                reasoning.append(f"Strong short timeline indicators (score: {short_score}) - MINOR ISSUE")
                return self._create_result(
                    TimelineCode.SHORT_TIMELINE,
                    0.8,
                    f"Short timeline indicated by patterns (score: {short_score})",
                    features,
                    reasoning,
                    context_assessment
                )
        
        # Moderate appropriate timeline evidence
        if appropriate_score >= 2 and appropriate_score >= short_score:
            reasoning.append(f"Moderate appropriate timeline evidence (score: {appropriate_score})")
            return self._create_result(
                TimelineCode.APPROPRIATE_TIMELINE,
                0.7,
                f"Likely appropriate timeline (pattern score: {appropriate_score})",
                features,
                reasoning,
                context_assessment
            )
        
        # Moderate short timeline evidence
        if short_score >= 2 and short_score > appropriate_score:
            # Again, consider clinical context
            if condition_type == "acute":
                reasoning.append(f"Short timeline suggested but acute context")
                return self._create_result(
                    TimelineCode.APPROPRIATE_TIMELINE,
                    0.6,
                    f"Short-term assessment may be appropriate for condition type (score: {short_score})",
                    features,
                    reasoning,
                    context_assessment
                )
            else:
                reasoning.append(f"Short timeline suggested (score: {short_score}) - potential issue")
                return self._create_result(
                    TimelineCode.SHORT_TIMELINE,
                    0.7,
                    f"Short timeline suggested (pattern score: {short_score})",
                    features,
                    reasoning,
                    context_assessment
                )
        
        # Default: insufficient information
        reasoning.append("Insufficient timeline information found")
        return self._create_result(
            TimelineCode.NOT_MENTIONED,
            0.75,
            "Primary outcome timeline information insufficient",
            features,
            reasoning,
            context_assessment
        )
    
    def _create_result(self, code: TimelineCode, confidence: float, 
                      message: str, features: TimelineFeatures, 
                      reasoning: List[str], context_assessment: str) -> TimelineResult:
        """Create a TimelineResult object"""
        return TimelineResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            clinical_implications=self.clinical_implications[code],
            context_assessment=context_assessment
        )
    
    def _create_empty_features(self) -> TimelineFeatures:
        """Create empty features object for error cases"""
        return TimelineFeatures(
            explicit_timeline_days=None, all_timelines_found=[], timeline_unit=None,
            short_timeline_score=0, appropriate_timeline_score=0, no_timeline_score=0,
            primary_outcome_mentioned=False, primary_outcome_phrases=[],
            condition_type="unclear", acute_condition_score=0, chronic_condition_score=0,
            timeline_justification_mentioned=False, follow_up_mentions=0,
            temporal_specificity=0, outcome_timeline_density=0,
            detected_patterns=[], extracted_timeline_info=[]
        )

def run_check(abstract : str):# just a wrapper method
    classifier = PrimaryOutcomeTimelineClassifier()
    result = classifier.check_primary_outcome_timeline(abstract)
    return result