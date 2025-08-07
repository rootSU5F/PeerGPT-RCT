import re
import spacy
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class FundingCode(Enum):
    """Simple funding classification codes"""
    FUNDED = 0                  # Any funding source mentioned
    NOT_FUNDED = 1              # No funding information found


@dataclass
class FundingFeatures:
    """Container for funding-related features"""
    # Pattern scores
    funding_mentioned_score: int         # Any funding indicators
    explicit_funding_score: int          # Clear "funded by" statements
    
    # Extracted information
    funding_sources: List[str]           # All detected funding sources
    funding_statements: List[str]        # Raw funding sentences
    
    # Text characteristics
    has_explicit_funding: bool           # Clear funding statement present
    
    # Context filtering results
    false_positive_count: int            # Number of filtered false positives
    
    # Detected patterns
    detected_patterns: List[str]


@dataclass
class FundingResult:
    code: FundingCode
    confidence: float
    message: str
    features: FundingFeatures
    reasoning: List[str]
    funding_source: str                  # Simple "funded by X" or "not funded"

class FundingSourceClassifier:
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP preprocessing"""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model not found. Try: python -m spacy download {spacy_model}")
            
        self._setup_funding_patterns()
        self._setup_context_filters()
    
    def _setup_funding_patterns(self):
        """Define context-aware patterns for funding detection - IMPROVED VERSION"""
        
        # HIGH CONFIDENCE: Explicit funding statements (4-5 points)
        self.explicit_funding_patterns = [
            (r'\bfunded\s+by\s+([^.;,\n]{3,100})', 5, 'funded_by'),
            (r'\bsponsored\s+by\s+([^.;,\n]{3,100})', 5, 'sponsored_by'),
            (r'\bfinanced\s+by\s+([^.;,\n]{3,100})', 4, 'financed_by'),
            (r'\bgranted?\s+by\s+([^.;,\n]{3,100})', 4, 'granted_by'),
            (r'\breceived\s+(?:funding|support|grants?)\s+from\s+([^.;,\n]{3,100})', 5, 'received_funding_from'),
            (r'\bobtained\s+(?:funding|support|grants?)\s+from\s+([^.;,\n]{3,100})', 4, 'obtained_funding_from'),
            (r'\bthis\s+(?:study|work|research)\s+was\s+(?:funded|sponsored|supported)\s+by\s+([^.;,\n]{3,100})', 5, 'study_funded_by'),
        ]
        
        # MEDIUM CONFIDENCE: Funding indicators with context (2-3 points)
        self.contextual_funding_patterns = [
            (r'\bfunding\s*:\s*([^.;,\n]{3,100})', 4, 'funding_colon'),
            (r'\bfunding\s+(?:was\s+)?provided\s+by\s+([^.;,\n]{3,100})', 4, 'funding_provided_by'),
            (r'\bfinancial\s+support\s+(?:was\s+)?provided\s+by\s+([^.;,\n]{3,100})', 4, 'financial_support_by'),
            (r'\b(?:research\s+)?grant\s+(?:number|no\.?|#)\s*([A-Z0-9\-\s]{4,20})', 3, 'grant_number'),
            (r'\bsupported\s+by\s+(?:grants?|funding)\s+from\s+([^.;,\n]{3,100})', 4, 'supported_by_grants'),
        ]
        
        # INSTITUTIONAL: Well-known funding organizations (3 points each)
        self.institutional_patterns = [
            (r'\b(?:NIH|National\s+Institutes?\s+of\s+Health)\b', 3, 'NIH'),
            (r'\b(?:Canadian\s+Institutes?\s+of\s+Health\s+Research|CIHR)\b', 3, 'CIHR'),
            (r'\b(?:NHS|National\s+Health\s+Service)\b', 3, 'NHS'),
            (r'\b(?:NSF|National\s+Science\s+Foundation)\b', 3, 'NSF'),
            (r'\b(?:CDC|Centers?\s+for\s+Disease\s+Control)\b', 3, 'CDC'),
            (r'\b(?:American\s+Heart\s+Association|AHA)\b', 3, 'AHA'),
            (r'\b(?:Gates\s+Foundation|Bill\s+(?:and|&)\s+Melinda\s+Gates\s+Foundation)\b', 3, 'Gates'),
            (r'\b(?:Wellcome\s+Trust|Wellcome)\b', 3, 'Wellcome'),
            (r'\b(?:European\s+Research\s+Council|ERC)\b', 3, 'ERC'),
            (r'\b(?:Medical\s+Research\s+Council|MRC)\b', 3, 'MRC'),
        ]
        
        # PHARMACEUTICAL: Industry funding (3 points each)
        self.pharma_patterns = [
            (r'\b(?:Pfizer|Novartis|Roche|Johnson\s*&\s*Johnson|J&J|Merck|GSK|GlaxoSmithKline|AstraZeneca|Sanofi|Abbott|Bayer|Boehringer\s+Ingelheim|Bristol\s*Myers\s*Squibb|BMS|Eli\s+Lilly|Gilead|Amgen|Biogen)\b', 3, 'pharma_company'),
            (r'\b(?:pharmaceutical|pharma)\s+(?:company|sponsor|industry)\b', 2, 'pharma_industry'),
        ]
        
        # LOW CONFIDENCE: Weak indicators that need context filtering (1-2 points)
        self.weak_patterns = [
            (r'\bfunding\b', 1, 'funding_mention'),  # Only if in funding context
            (r'\bgrants?\b', 1, 'grant_mention'),    # Only if in funding context
            (r'\bfinancial\s+support\b', 2, 'financial_support'),
            (r'\bresearch\s+(?:grant|funding)\b', 2, 'research_funding'),
            (r'\binstitutional\s+(?:grant|funding)\b', 2, 'institutional_funding'),
        ]
        
        # Funding source extraction patterns (for extracting source names)
        self.funding_extraction_patterns = [
            r'\bfunded\s+by\s+([^.;,\n]{5,150})',
            r'\bsponsored\s+by\s+([^.;,\n]{5,150})',
            r'\bsupported\s+by\s+([^.;,\n]{5,150})',
            r'\breceived\s+(?:funding|support|grants?)\s+from\s+([^.;,\n]{5,150})',
            r'\bgranted?\s+by\s+([^.;,\n]{5,150})',
            r'\bfinanced\s+by\s+([^.;,\n]{5,150})',
            r'\bfunding\s*:\s*([^.;,\n]{5,150})',
            r'\bfunding\s+(?:was\s+)?provided\s+by\s+([^.;,\n]{5,150})',
            r'\bfinancial\s+support\s+(?:was\s+)?provided\s+by\s+([^.;,\n]{5,150})',
            r'\bthis\s+(?:study|work|research)\s+was\s+(?:funded|sponsored|supported)\s+by\s+([^.;,\n]{5,150})'
        ]
    
    def _setup_context_filters(self):
        """Set up context filters to eliminate false positives"""
        
        # Patterns that indicate NON-FUNDING context for common trigger words
        self.non_funding_contexts = {
            'support': [
                r'(?:data|results|findings|evidence|studies?|research|analysis|this|these)\s+support',
                r'support\s+(?:the\s+)?(?:use|hypothesis|conclusion|finding|idea|concept|notion|view|theory)',
                r'support\s+that',
                r'support\s+this',
                r'support\s+our',
                r'support\s+a\s+(?:role|benefit|strategy|approach|method)',
                r'do\s+not\s+support',
                r'does\s+not\s+support',
                r'failed\s+to\s+support',
                r'strongly\s+support',
                r'further\s+support',
                r'additional\s+support',
                r'provide\s+support',
                r'to\s+support',
                r'support\s+for\s+(?:the|a|an)',
                r'in\s+support\s+of',
            ],
            'funding': [
                r'funding\s+(?:mechanism|model|structure|approach|strategy|decision)',
                r'healthcare\s+funding',
                r'public\s+funding\s+(?:of|for)',
                r'government\s+funding\s+(?:of|for)',
                r'funding\s+(?:cuts?|reduction|increase|allocation)',
            ],
            'grant': [
                r'grant\s+(?:access|permission|approval)',
                r'grant\s+that',
                r'grant\s+the\s+(?:hypothesis|assumption|premise)',
                r'taking\s+for\s+granted',
                r'granted\s+that',
                r'grant\s+(?:exemption|waiver)',
            ]
        }
        
        # Contexts where funding words ARE likely to be funding-related
        self.funding_contexts = [
            r'(?:funded|sponsored|supported|financed|granted)\s+by',
            r'(?:received|obtained|secured)\s+(?:funding|support|grants?)',
            r'funding\s*:',
            r'grant\s+(?:number|no|#)',
            r'financial\s+(?:support|assistance|backing)',
            r'research\s+(?:funding|grant)',
            r'study\s+(?:funding|grant)',
            r'(?:NIH|NSF|CIHR|NHS|Gates|Wellcome)',
            r'(?:pharmaceutical|pharma)\s+(?:company|sponsor|funding)',
        ]
    
    def check_funding_source(self, text: str) -> FundingResult:
        """
        Context-aware funding detection with false positive filtering
        
        Args:
            text: Input abstract text to analyze
            
        Returns:
            FundingResult with improved FUNDED/NOT_FUNDED classification
        """
        reasoning = []
        
        if not text or not text.strip():
            return self._create_result(
                FundingCode.NOT_FUNDED,
                1.0,
                "Empty or invalid input",
                self._create_empty_features(),
                ["Input is empty or invalid"],
                "not funded"
            )
        
        # Extract features with context awareness
        features = self._extract_funding_features_contextual(text, reasoning)
        
        # Make improved classification
        return self._make_funding_classification(features, reasoning)
    
    def _extract_funding_features_contextual(self, text: str, reasoning: List[str]) -> FundingFeatures:
        """Extract funding features with context-aware filtering"""
        
        clean_text = text.lower().strip()
        
        # Initialize scores
        funding_score = 0
        explicit_score = 0
        false_positive_count = 0
        detected_patterns = []
        
        # 1. HIGH CONFIDENCE: Explicit funding patterns (always count)
        for pattern, points, context in self.explicit_funding_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                explicit_score += points
                funding_score += points
                detected_patterns.append(f"explicit: {context}")
                reasoning.append(f"Explicit funding pattern: {context} (+{points} points)")
        
        # 2. MEDIUM CONFIDENCE: Contextual funding patterns (always count)
        for pattern, points, context in self.contextual_funding_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                funding_score += points
                detected_patterns.append(f"contextual: {context}")
                reasoning.append(f"Contextual funding pattern: {context} (+{points} points)")
        
        # 3. INSTITUTIONAL: Well-known funding organizations (always count)
        for pattern, points, context in self.institutional_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Additional context check: should be in a funding-like sentence
                if self._is_likely_funding_context(text, match.start(), match.end()):
                    funding_score += points
                    detected_patterns.append(f"institutional: {context}")
                    reasoning.append(f"Institutional funder: {context} (+{points} points)")
                else:
                    false_positive_count += 1
                    reasoning.append(f"Institutional mention filtered: {context} (not in funding context)")
        
        # 4. PHARMACEUTICAL: Industry funding (always count)
        for pattern, points, context in self.pharma_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Additional context check for pharma companies (could be mentioned for other reasons)
                if self._is_likely_funding_context(text, match.start(), match.end()):
                    funding_score += points
                    detected_patterns.append(f"pharma: {context}")
                    reasoning.append(f"Pharmaceutical funder: {context} (+{points} points)")
                else:
                    false_positive_count += 1
                    reasoning.append(f"Pharmaceutical mention filtered: {context} (not in funding context)")
        
        # 5. LOW CONFIDENCE: Weak patterns with strict context filtering
        for pattern, points, context in self.weak_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                word = match.group(0).lower()
                
                # Apply context filtering for weak patterns
                if self._is_funding_context_match(text, match.start(), match.end(), word):
                    funding_score += points
                    detected_patterns.append(f"weak: {context}")
                    reasoning.append(f"Weak funding indicator: {context} (+{points} points)")
                else:
                    false_positive_count += 1
                    reasoning.append(f"False positive filtered: '{word}' in non-funding context")
        
        # Extract funding sources
        funding_sources = self._extract_funding_sources_improved(text, reasoning)
        
        # Extract funding statements
        funding_statements = self._extract_funding_statements(text)
        
        # Check for explicit funding language
        has_explicit_funding = explicit_score > 0
        
        return FundingFeatures(
            funding_mentioned_score=funding_score,
            explicit_funding_score=explicit_score,
            funding_sources=funding_sources,
            funding_statements=funding_statements,
            has_explicit_funding=has_explicit_funding,
            false_positive_count=false_positive_count,
            detected_patterns=detected_patterns
        )
    
    def _is_funding_context_match(self, text: str, start: int, end: int, word: str) -> bool:
        """Check if a weak pattern match is in actual funding context"""
        
        # Get surrounding context (100 chars before and after the match)
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()
        
        # Check for non-funding contexts first (higher priority)
        if word in self.non_funding_contexts:
            for non_funding_pattern in self.non_funding_contexts[word]:
                if re.search(non_funding_pattern, context, re.IGNORECASE):
                    return False  # Definitely not funding context
        
        # Check for funding contexts
        for funding_pattern in self.funding_contexts:
            if re.search(funding_pattern, context, re.IGNORECASE):
                return True  # Likely funding context
        
        # Default: if no clear context, don't count weak patterns
        return False
    
    def _is_likely_funding_context(self, text: str, start: int, end: int) -> bool:
        """Check if an institutional/pharma mention is likely in funding context"""
        
        # Get surrounding context (150 chars before and after)
        context_start = max(0, start - 150)
        context_end = min(len(text), end + 150)
        context = text[context_start:context_end].lower()
        
        # Strong funding context indicators
        strong_indicators = [
            r'(?:funded|sponsored|supported|financed|granted)\s+by',
            r'(?:received|obtained|secured)\s+(?:funding|support|grants?)\s+from',
            r'funding\s*:',
            r'grant\s+(?:number|no|#)',
            r'financial\s+(?:support|assistance)',
            r'research\s+(?:funding|grant)',
            r'study\s+(?:funding|grant)',
        ]
        
        for indicator in strong_indicators:
            if re.search(indicator, context, re.IGNORECASE):
                return True
        
        # Weak funding context (still positive but lower confidence)
        weak_indicators = [
            r'funding',
            r'grant',
            r'support',
            r'sponsor',
            r'financial',
        ]
        
        # Need at least 2 weak indicators for context
        weak_count = sum(1 for indicator in weak_indicators 
                        if re.search(indicator, context, re.IGNORECASE))
        
        return weak_count >= 2
    
    def _extract_funding_sources_improved(self, text: str, reasoning: List[str]) -> List[str]:
        """Extract specific funding source names with improved cleaning"""
        sources = []
        
        for pattern in self.funding_extraction_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract the funding source
                    source = match.group(1).strip()
                    
                    # Clean up the extracted source
                    source = self._clean_funding_source(source)
                    
                    # Validate source (not too short, not common false positives)
                    if self._is_valid_funding_source(source) and source not in sources:
                        sources.append(source)
                        reasoning.append(f"Funding source extracted: '{source}'")
                        
                except (IndexError, AttributeError):
                    continue
        
        # Also look for well-known funding agencies in context
        sources.extend(self._extract_known_agencies(text, reasoning))
        
        return list(dict.fromkeys(sources))[:5]  # Remove duplicates, limit to 5
    
    def _clean_funding_source(self, source: str) -> str:
        """Clean and normalize extracted funding source"""
        
        # Remove common trailing phrases
        source = re.sub(r'\s+(?:and\s+others?|et\s+al\.?|etc\.?)$', '', source, flags=re.IGNORECASE)
        source = re.sub(r'\s*[;,].*$', '', source)  # Remove everything after ; or ,
        source = re.sub(r'\s*\([^)]*\)$', '', source)  # Remove trailing parentheses
        source = re.sub(r'\s*\[[^\]]*\]$', '', source)  # Remove trailing brackets
        
        # Remove common non-funding suffixes
        source = re.sub(r'\s+(?:grant|funding|support|program|initiative|foundation|trust|council|agency|institute|center|centre)$', 
                       lambda m: ' ' + m.group(0).strip(), source, flags=re.IGNORECASE)
        
        # Clean whitespace
        source = ' '.join(source.split())
        
        return source.strip()
    
    def _is_valid_funding_source(self, source: str) -> bool:
        """Check if extracted source is a valid funding source"""
        
        # Too short
        if len(source) < 3:
            return False
        
        # Common false positives
        false_positives = [
            'the', 'this', 'that', 'these', 'those', 'and', 'or', 'but', 'with', 'from', 'for',
            'data', 'results', 'findings', 'evidence', 'study', 'research', 'analysis',
            'authors', 'investigators', 'researchers', 'participants', 'patients', 'subjects',
            'use', 'treatment', 'intervention', 'control', 'method', 'approach', 'strategy'
        ]
        
        if source.lower() in false_positives:
            return False
        
        # Too generic
        if re.match(r'^(?:a|an|the)\s+\w+$', source.lower()):
            return False
        
        return True
    
    def _extract_known_agencies(self, text: str, reasoning: List[str]) -> List[str]:
        """Extract well-known funding agencies mentioned in funding context"""
        
        agencies = []
        known_agencies = [
            ('Canadian Institutes of Health Research', 'CIHR'),
            ('National Institutes of Health', 'NIH'), 
            ('National Health Service', 'NHS'),
            ('National Science Foundation', 'NSF'),
            ('American Heart Association', 'AHA'),
            ('Gates Foundation', 'Bill and Melinda Gates Foundation'),
            ('Wellcome Trust', 'Wellcome'),
            ('European Research Council', 'ERC'),
            ('Medical Research Council', 'MRC'),
        ]
        
        for full_name, abbrev in known_agencies:
            # Check for full name or abbreviation
            for name in [full_name, abbrev]:
                if re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE):
                    # Check if it's in funding context
                    matches = list(re.finditer(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
                    for match in matches:
                        if self._is_likely_funding_context(text, match.start(), match.end()):
                            if name not in agencies:
                                agencies.append(name)
                                reasoning.append(f"Known funding agency found: '{name}'")
                            break
        
        return agencies
    
    def _extract_funding_statements(self, text: str) -> List[str]:
        """Extract sentences containing funding information"""
        statements = []
        
        # Split into sentences and find funding-related ones
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            # More specific funding sentence detection
            if re.search(r'\b(?:fund|sponsor|grant|financial\s+support)\w*\s+(?:by|from|provided|received|obtained)\b', 
                        sentence.lower()):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 15:
                    statements.append(clean_sentence)
        
        return statements[:3]  # Limit to first 3 relevant sentences
    
    def _make_funding_classification(self, features: FundingFeatures, reasoning: List[str]) -> FundingResult:
        """Make improved FUNDED/NOT_FUNDED classification with better confidence"""
        
        funding_score = features.funding_mentioned_score
        explicit_score = features.explicit_funding_score
        has_sources = len(features.funding_sources) > 0
        has_explicit = features.has_explicit_funding
        false_positives = features.false_positive_count
        
        reasoning.append(f"Final funding score: {funding_score} (after filtering {false_positives} false positives)")
        
        # IMPROVED BINARY LOGIC WITH CONFIDENCE SCORING
        
        # 1. Strong evidence: Clear funding sources extracted = FUNDED (high confidence)
        if has_sources and explicit_score >= 3:
            funding_source = f"funded by {', '.join(features.funding_sources)}"
            reasoning.append(f"High confidence: Explicit funding sources detected")
            return self._create_result(
                FundingCode.FUNDED,
                0.95,
                f"Funding sources identified: {', '.join(features.funding_sources)}",
                features,
                reasoning,
                funding_source
            )
        
        # 2. Good evidence: Funding sources OR strong explicit language = FUNDED (good confidence)
        elif has_sources or explicit_score >= 5:
            if has_sources:
                funding_source = f"funded by {', '.join(features.funding_sources)}"
                message = f"Funding sources identified: {', '.join(features.funding_sources)}"
            else:
                funding_source = "funded (explicit funding statement found)"
                message = "Strong explicit funding statement found"
            
            reasoning.append(f"Good evidence: Sources={has_sources}, Explicit score={explicit_score}")
            return self._create_result(
                FundingCode.FUNDED,
                0.85,
                message,
                features,
                reasoning,
                funding_source
            )
        
        # 3. Moderate evidence: Multiple funding indicators = FUNDED (moderate confidence)
        elif funding_score >= 6:
            funding_source = "funded (multiple funding indicators detected)"
            reasoning.append(f"Moderate evidence: Multiple funding indicators (score: {funding_score})")
            return self._create_result(
                FundingCode.FUNDED,
                0.75,
                f"Multiple funding indicators detected (score: {funding_score})",
                features,
                reasoning,
                funding_source
            )
        
        # 4. Weak evidence: Some funding mentioned = FUNDED (lower confidence)
        elif funding_score >= 3 and explicit_score >= 3:
            funding_source = "funded (some funding evidence found)"
            reasoning.append(f"Weak evidence: Some funding indicators (score: {funding_score})")
            return self._create_result(
                FundingCode.FUNDED,
                0.65,
                f"Some funding evidence detected",
                features,
                reasoning,
                funding_source
            )
        
        # 5. Minimal evidence: Very weak signals = FUNDED (low confidence)
        elif funding_score >= 2 and false_positives == 0:  # Only if no false positives
            funding_source = "funded (minimal funding evidence)"
            reasoning.append(f"Minimal evidence: Very weak funding signals (score: {funding_score})")
            return self._create_result(
                FundingCode.FUNDED,
                0.55,
                f"Minimal funding evidence detected",
                features,
                reasoning,
                funding_source
            )
        
        # 6. Default: No clear funding evidence = NOT_FUNDED
        else:
            if false_positives > 0:
                reasoning.append(f"Classification: NOT_FUNDED ({false_positives} false positives filtered)")
            else:
                reasoning.append("Classification: NOT_FUNDED (no clear funding evidence)")
                
            return self._create_result(
                FundingCode.NOT_FUNDED,
                0.85,
                f"No funding information detected",
                features,
                reasoning,
                "not funded"
            )
    
    def _create_result(self, code: FundingCode, confidence: float, 
                      message: str, features: FundingFeatures, 
                      reasoning: List[str], funding_source: str) -> FundingResult:
        """Create a FundingResult object"""
        return FundingResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning,
            funding_source=funding_source
        )
    
    def _create_empty_features(self) -> FundingFeatures:
        """Create empty features object for error cases"""
        return FundingFeatures(
            funding_mentioned_score=0, explicit_funding_score=0,
            funding_sources=[], funding_statements=[],
            has_explicit_funding=False, false_positive_count=0,
            detected_patterns=[]
        )


def run_check(abstract: str):
    """Wrapper method to run funding source check - IMPROVED"""
    classifier = FundingSourceClassifier()
    result = classifier.check_funding_source(abstract)
    return result


# Test function for debugging
def test_funding_classifier(abstract_text: str):
    """Test the funding classifier with debug output"""
    print("=" * 60)
    print("TESTING IMPROVED FUNDING CLASSIFIER")
    print("=" * 60)
    
    classifier = FundingSourceClassifier()
    result = classifier.check_funding_source(abstract_text)
    
    print(f"\n🔍 CLASSIFICATION RESULT:")
    print(f"Code: {result.code}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Message: {result.message}")
    print(f"Funding source: {result.funding_source}")
    
    print(f"\n📊 EXTRACTED FEATURES:")
    print(f"Funding score: {result.features.funding_mentioned_score}")
    print(f"Explicit funding score: {result.features.explicit_funding_score}")
    print(f"Funding sources: {result.features.funding_sources}")
    print(f"Has explicit funding: {result.features.has_explicit_funding}")
    print(f"False positives filtered: {result.features.false_positive_count}")
    
    print(f"\n💭 REASONING:")
    for i, reason in enumerate(result.reasoning, 1):
        print(f"{i}. {reason}")
    
    print(f"\n🔬 DETECTED PATTERNS:")
    for pattern in result.features.detected_patterns:
        print(f"- {pattern}")
    
    print(f"\n📋 FUNDING STATEMENTS:")
    for stmt in result.features.funding_statements:
        print(f"- {stmt}")
    
    return result

abst = ''' 
Abstract
Objectives To assess the effectiveness of prone positioning to reduce the risk of death or respiratory failure in non-critically ill patients admitted to hospital with covid-19.

Design Multicentre pragmatic randomised clinical trial.

Setting 15 hospitals in Canada and the United States from May 2020 until May 2021.

Participants Eligible patients had a laboratory confirmed or a clinically highly suspected diagnosis of covid-19, needed supplemental oxygen (up to 50% fraction of inspired oxygen), and were able to independently lie prone with verbal instruction. Of the 570 patients who were assessed for eligibility, 257 were randomised and 248 were included in the analysis.

Intervention Patients were randomised 1:1 to prone positioning (that is, instructing a patient to lie on their stomach while they are in bed) or standard of care (that is, no instruction to adopt prone position).

Main outcome measures The primary outcome was a composite of in-hospital death, mechanical ventilation, or worsening respiratory failure defined as needing at least 60% fraction of inspired oxygen for at least 24 hours. Secondary outcomes included the change in the ratio of oxygen saturation to fraction of inspired oxygen.

Results The trial was stopped early on the basis of futility for the pre-specified primary outcome. The median time from hospital admission until randomisation was 1 day, the median age of patients was 56 (interquartile range 45-65) years, 89 (36%) patients were female, and 222 (90%) were receiving oxygen via nasal prongs at the time of randomisation. The median time spent prone in the first 72 hours was 6 (1.5-12.8) hours in total for the prone arm compared with 0 (0-2) hours in the control arm. The risk of the primary outcome was similar between the prone group (18 (14%) events) and the standard care group (17 (14%) events) (odds ratio 0.92, 95% confidence interval 0.44 to 1.92). The change in the ratio of oxygen saturation to fraction of inspired oxygen after 72 hours was similar for patients randomised to prone positioning and standard of care.

Conclusion Among non-critically ill patients with hypoxaemia who were admitted to hospital with covid-19, a multifaceted intervention to increase prone positioning did not improve outcomes. However, wide confidence intervals preclude definitively ruling out benefit or harm. Adherence to prone positioning was poor, despite multiple efforts to increase it. Subsequent trials of prone positioning should aim to develop strategies to improve adherence to awake prone positioning.

Study registration ClinicalTrials.gov NCT04383613.

'''
test_result = test_funding_classifier(abst)