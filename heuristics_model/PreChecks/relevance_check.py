from ast import pattern
from email import message
from pyexpat import features
import spacy
import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np


class RelevanceCode (Enum):# Enum special class that is used to define constat values
    """Relevance codes """
    RESEARCH_ABSTRACT = 0 
    NON_RESEARCH =1 
    INSUFFICIENT_CONTENT = 2
    NEWS_ARTICLE = 3
    OPINION_EDIT = 4
    REVIWE_SUMMARY = 5
    CASE_REPORT =6 
    CONFRENCE_ABSTRACT =7 
    BOOK_CHAPTER =8
    UKNOWN =9


@dataclass
class HeuristicFeatures:
    """Container for all heuristic features"""
    # Content metrics
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    
    # Research indicators
    study_design_score: int
    methodology_score: int
    statistical_score: int
    structure_score: int
    
    # Non-research indicators  
    news_score: int
    opinion_score: int
    review_score: int
    case_report_score: int
    
    # Text characteristics
    numerical_density: float
    technical_terms_ratio: float
    passive_voice_count: int
    
    # Detected patterns
    detected_patterns: List[str]


@dataclass #decorator just for creating auto functions
class RelevanceResult :
    code : RelevanceCode
    confidence : float # 0-1 level
    message :str
    features: HeuristicFeatures
    reasoning : List[str]

class HeuristicAbstractRelevanceChecker:
    # creating the constructor 
    def __init__(self , spacy_model : str = "en_core_web_sm" ):
        """ initializing with spaCy model for NLP preprocessing"""
        try : 
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print("spacy model not found ")
            print(f"try download it by python -m spacy download {spacy_model}")
        
        self._setup_hueristic_patterns()
        self._setup_hueristic_threshholds()
    def _setup_hueristic_patterns(self):
        
        """Define all of the patterns for classification
            to understand the scores read the md file in the docs file  -relevanceScroing-
        """
        #list of tuples of the patternes and regex that we may use 
        # im going to devide it into 3 levels from strong indicators 3 points to weak 1 point
        self.study_design_patterns = [
            #stron            
            (r'\b(?:randomized|randomised)\s+(?:controlled\s+)?trial\b', 3),
            (r'\bdouble-blind\b', 3),
            (r'\bplacebo-controlled\b', 3),
            (r'\brct\b', 3),
            #phases from i-iii or 1,2,3,4
            (r'\bphase\s+(?:i{1,3}|[1-4])\b', 3),


            # Moderate indicators (2 points each)
            (r'\bsingle-blind\b', 2),
            (r'\bcross-?over\b', 2),
            (r'\bmulti-?cent(?:er|re)\b', 2),
            (r'\bprospective\s+(?:study|cohort)\b', 2),
            (r'\bretro?spective\s+(?:study|analysis)\b', 2),

                        
            # Weak indicators (1 point each)
            (r'\bcohort\s+study\b', 1),
            (r'\bcase-control\s+study\b', 1),
            (r'\bpilot\s+study\b', 1),
            (r'\bfeasibility\s+study\b', 1)
        ]

        self.methodology_patterns = [
            # Strong methodology (3 points each)
            (r'\brandom(?:ly\s+)?(?:assigned|allocated)\b', 3),
            (r'\bintention-to-treat\b', 3),
            (r'\bper-protocol\s+analysis\b', 3),
            
            # Moderate methodology (2 points each)
            (r'\bstratified\s+randomization\b', 2),
            (r'\bblinded?\s+(?:assessment|evaluation)\b', 2),
            (r'\bunblinded?\b', 2),
            (r'\bopen-label\b', 2),
            
            # Basic methodology (1 point each)
            (r'\bbaseline\s+characteristics\b', 1),
            (r'\bfollow-up\s+(?:period|duration)\b', 1),
            (r'\benrollment\b', 1),
            (r'\bscreening\b', 1),
            (r'\beligibility\s+criteria\b', 1)
        ]


        self.statistical_pattern = [

            # Statistical analysis (2-3 points each)
            (r'p\s*[<>=]\s*0\.\d+', 3),
            (r'\b95%\s*(?:confidence\s+interval|ci)\b', 3),
            (r'\bhazard\s+ratio\b', 3),
            (r'\bodds\s+ratio\b', 3),
            (r'\brisk\s+ratio\b', 3),
            (r'\bmean\s+difference\b', 2),
            (r'\bstatistical\s+(?:analysis|significance)\b', 2),
            (r'\bpower\s+(?:analysis|calculation)\b', 2),
            (r'\bsample\s+size\s+calculation\b', 2),
            
            # Sample size reporting (1-2 points)
            (r'\bn\s*=\s*\d+', 2),
            (r'\b\d+\s+(?:participants?|patients?|subjects?)\b', 1)
       
        ]

        self.structure_patterns = [
            # Structured abstract sections (2 points each)
            (r'\b(?:background|introduction)\s*:', 2),
            (r'\b(?:objective|aim)s?\s*:', 2),
            (r'\bmethods?\s*:', 2),
            (r'\bresults?\s*:', 2),
            (r'\bconclusions?\s*:', 2),
            
            # Content flow indicators (1 point each)
            (r'\bprimary\s+(?:endpoint|outcome)\b', 1),
            (r'\bsecondary\s+(?:endpoint|outcome)\b', 1),
            (r'\bmain\s+(?:finding|result)\b', 1),
            (r'\bclinical\s+(?:significance|implication)\b', 1)
        ]

        self.news_patterns = [
            # Strong news indicators (3 points each) 
            (r'\b(?:reports?|announces?)\s+(?:show|indicate|find)\b', 3),
            (r'\baccording\s+to\s+(?:a\s+)?(?:new\s+)?study\b', 3),
            (r'\bresearchers?\s+(?:say|report|told\s+reporters)\b', 3),
            (r'\bpublished\s+(?:today|yesterday|this\s+week)\b', 3),
            (r'\bnews\s+release\b', 3),
            (r'\bpress\s+release\b', 3),

            # Moderate news indicators (2 points each)
            (r'\bscientists?\s+(?:say|found|discovered)\b', 2),
            (r'\ba\s+new\s+study\s+(?:shows|suggests|finds)\b', 2),
            (r'\bbreaking\s+(?:news|story)\b', 2),
            (r'\bexclusive\s+interview\b', 2)
        ]

        self.opinion_patterns = [
            # Strong opinion indicators (3 points each)
            (r'\bi\s+(?:believe|think|argue)\b', 3),
            (r'\bin\s+my\s+opinion\b', 3),
            (r'\bwe\s+(?:should|must|need\s+to)\b', 3),
            (r'\beditorial\b', 3),
            (r'\bcommentary\b', 3),
            
            # Moderate opinion indicators (2 points each)
            (r'\bperspective\b', 2),
            (r'\bviewpoint\b', 2),
            (r'\bit\s+is\s+(?:important|crucial|essential)\s+that\b', 2),
            (r'\bwe\s+advocate\b', 2)
        ]

        self.review_patterns = [
            # Strong review indicators (3 points each)
            (r'\bsystematic\s+review\b', 3),
            (r'\bmeta-analysis\b', 3),
            (r'\bthis\s+review\b', 3),
            (r'\bliterature\s+review\b', 3),
            
            # Moderate review indicators (2 points each)
            (r'\brecent\s+studies?\s+have\s+shown\b', 2),
            (r'\bseveral\s+studies?\s+(?:have\s+)?(?:shown|demonstrated)\b', 2),
            (r'\bmultiple\s+studies?\b', 2),
            (r'\bsummary\s+of\s+(?:evidence|literature)\b', 2),
            (r'\boverview\s+of\b', 2)
        ]


        self.case_report_patterns = [
            # Strong case report indicators (3 points each)
            (r'\bcase\s+report\b', 3),
            (r'\bcase\s+series\b', 3),
            (r'\bwe\s+(?:present|report)\s+(?:a\s+)?case\b', 3),
            
            # Moderate case indicators (2 points each)
            (r'\ba\s+\d+-year-old\s+(?:man|woman|male|female|patient)\b', 2),
            (r'\bpatient\s+presented\s+with\b', 2),
            (r'\bclinical\s+case\b', 2),
            (r'\bcase\s+(?:study|description)\b', 2)
        ]

    def _setup_hueristic_threshholds(self):
        # HARD FILTERS :]
        self.min_word_count = 75
        self.min_sentence_count = 4
        self.min_avg_sentence_length = 10


        self.research_thresholds ={
            'strong_research' : 12,
            'moderate_research' : 8,
            'weak_research' : 5
        }

        self.non_research_thresholds = {
            'strong_non_research': 8,   
            'moderate_non_research': 5, 
            'weak_non_research': 3      
        }
        
        self.confidence_params ={
            'high_confidence': 0.9,
            'medium_confidence': 0.75,
            'low_confidence': 0.6
        }

    def check_relevance(self , text: str) ->RelevanceResult:
        """
        Main heuristic relevance checking function
        
        Args:
            text: Input text to analyze
            
        Returns:
            RelevanceResult with heuristic classification
        """
        reasoning =[]
        if not text or not text.strip():
            return self._create_result(
                RelevanceCode.INSUFFICIENT_CONTENT ,
                1.0,
                "Empty or invalid input" ,
                self._create_empty_features(),
                ["Input is empty or invalid"]
            )
        features = self._extract_heuristic_features(text, reasoning)
        length_result = self._check_content_length(features, reasoning)
        if length_result:
            return length_result
        
        research_score = self._calculate_research_score(features ,reasoning)
        non_research_score = self._calculate_non_research_score(features , reasoning)
        
        
        return self._make_classification_decision(
            research_score, non_research_score, features, reasoning
        )

    def _extract_heuristic_features(self , text:str , reasoning:List[str])->HeuristicFeatures:
        """Extract all heuristic features from text"""
        doc = self.nlp(text)
        clean_text = text.lower().strip()

        sentence = list(doc.sents)
        sentence_count = len(sentence)

        word_count = len([t for t in doc if not t.is_space and not t.is_punct])#number of tokens not words ;]
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        reasoning.append(f"Basic metrics: {word_count} words, {sentence_count} sentences")


        # score each catagory ! 
        study_design_score = self._score_patterns(clean_text, self.study_design_patterns)
        methodology_score =self._score_patterns(clean_text , self.methodology_patterns)
        statistical_score = self._score_patterns(clean_text , self.statistical_pattern)
        structure_score = self._score_patterns(clean_text, self.structure_patterns)

        news_score = self._score_patterns(clean_text, self.news_patterns)
        opinion_score = self._score_patterns(clean_text, self.opinion_patterns)
        review_score = self._score_patterns(clean_text, self.review_patterns)
        case_report_score = self._score_patterns(clean_text, self.case_report_patterns)

        # Additional characteristics
        numerical_density = self._calculate_numerical_density(doc)
        technical_terms_ratio = self._calculate_technical_ratio(doc)
        passive_voice_count = self._count_passive_voice(doc)

        # Collect detected patterns
        detected_patterns = self._get_detected_patterns(clean_text)


        return HeuristicFeatures(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            study_design_score=study_design_score,
            methodology_score=methodology_score,
            statistical_score=statistical_score,
            structure_score=structure_score,
            news_score=news_score,
            opinion_score=opinion_score,
            review_score=review_score,
            case_report_score=case_report_score,
            numerical_density=numerical_density,
            technical_terms_ratio=technical_terms_ratio,
            passive_voice_count=passive_voice_count,
            detected_patterns=detected_patterns
        )

    def _score_patterns(self, text: str, patterns: List[Tuple[str, int]]) -> int:
        """Score text against a list of patterns"""
        total_score = 0
        for pattern, points in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                # Cap points per pattern to avoid overwhelming
                pattern_score = min(matches * points, points * 2)
                total_score += pattern_score
        return total_score




    def _calculate_numerical_density(self, doc) -> float:
        """Calculate density of numerical content"""
        num_tokens = len([t for t in doc if t.like_num or re.match(r'\d', t.text)])
        total_tokens = len([t for t in doc if not t.is_space])
        return num_tokens / total_tokens if total_tokens > 0 else 0

    def _calculate_technical_ratio(self, doc) -> float:#words like stabilization development are considered technical word
        """Calculate ratio of technical/scientific terms"""
        technical_count = 0
        total_words = 0
        
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                total_words += 1
                # Technical indicators: long words, scientific suffixes
                if (len(token.text) > 8 or 
                    token.text.endswith(('tion', 'sion', 'ment', 'sis', 'ity', 'ism'))):
                    technical_count += 1
            return technical_count / total_words if total_words > 0 else 0.0


    def _count_passive_voice(self, doc) -> int:#words that have something in passive is more likely research

        """Count passive voice constructions"""
        passive_count = 0
        for token in doc:
            if (token.dep_ == 'auxpass' or 
                (token.tag_ == 'VBN' and any(child.lemma_ in ['be', 'was', 'were'] 
                                           for child in token.children))):
                passive_count += 1
        return passive_count
    def _get_detected_patterns(self , text:str) ->List[str]:
        """Get list of all detected patterns for transparency"""
        detected =[]

        all_patterns_sets =[
                        ('study_design', self.study_design_patterns),
            ('methodology', self.methodology_patterns),
            ('statistical', self.statistical_pattern),
            ('structure', self.structure_patterns),
            ('news', self.news_patterns),
            ('opinion', self.opinion_patterns),
            ('review', self.review_patterns),
            ('case_report', self.case_report_patterns)
        ]

        for category , patterns in all_patterns_sets:
            for pattern , score in patterns:
                if re.search(pattern , text ,re.IGNORECASE):
                    detected.append(f"{category}_{pattern[:20]}...")
        return detected[:10]  # Limit for readability
    def _check_content_length(self , features : HeuristicFeatures , reasoning:List[str])-> RelevanceResult:
        if features.word_count < self.min_word_count:
            reasoning.append(f"Too short: {features.word_count} < {self.min_word_count} words")
            return self._create_result(
                RelevanceCode.INSUFFICIENT_CONTENT,
                0.95,
                f"Content too short: {features.word_count} words",
                features,
                reasoning
            )
        if features.sentence_count < self.min_avg_sentence_length:
            reasoning.append(f"Too few sentences: {features.sentence_count} < {self.min_sentence_count}")
            return self._create_result(
                RelevanceCode.INSUFFICIENT_CONTENT,
                0.9,
                f"Too few sentences: {features.sentence_count}",
                features,
                reasoning
            )
        if features.avg_sentence_length < self.min_avg_sentence_length:
            reasoning.append(f"Sentences too short: avg {features.avg_sentence_length:.1f} words")
            return self._create_result(
                RelevanceCode.INSUFFICIENT_CONTENT,
                0.8,
                f"Sentences too short: avg {features.avg_sentence_length:.1f} words",
                features,
                reasoning
            )
        return None
    def _calculate_research_score(self , features:HeuristicFeatures , reasoning:List[str]):
        """Calculate total research evidence score"""
        research_score = (
            features.study_design_score +
            features.methodology_score +
            features.statistical_score +
            features.structure_score
        )
        if features.numerical_density > 0.1:
            research_score += 2
            reasoning.append(f"High numerical density: {features.numerical_density:.2f}")
            
        if features.technical_terms_ratio > 0.3:
            research_score += 2
            reasoning.append(f"High technical term ratio: {features.technical_terms_ratio:.2f}")
            
        if features.passive_voice_count >= 2:
            research_score += 1
            reasoning.append(f"Passive voice usage: {features.passive_voice_count}")
        
        reasoning.append(f"Research evidence score: {research_score}")
        return research_score
    def _calculate_non_research_score(self , features:HeuristicFeatures , reasoning:List[str])->int:
        "calculate non research score"
        non_research_score = (
            features.news_score +
            features.opinion_score +
            features.review_score +
            features.case_report_score
        )
        reasoning.append(f"Non-research evidence score: {non_research_score}")
        return non_research_score

    #Making the final decision
    def _make_classification_decision(self , research_score:int , non_research_score:int , features:HeuristicFeatures , reasoning:List[str])->RelevanceResult:
        "Make the final result"
        score_diff = abs(research_score - non_research_score)
        if score_diff >= 8:
            confidence = self.confidence_params['high_confidence']
        elif score_diff >= 4:
            confidence = self.confidence_params['medium_confidence']
        else:
            confidence = self.confidence_params['low_confidence']
        #classification logic
        if (research_score >= self.research_thresholds['strong_research'] and 
            research_score > non_research_score + 3):
            reasoning.append(f"Strong research evidence: {research_score} >> {non_research_score}")
            return self._create_result(
                RelevanceCode.RESEARCH_ABSTRACT,
                confidence,
                f"Research abstract detected (score: +{research_score}, -{non_research_score})",
                features,
                reasoning
            )
        elif (non_research_score >= self.non_research_thresholds['strong_non_research'] and
              non_research_score > research_score + 2):
            
            # Determine specific non-research type
            if features.news_score >= 5:
                code = RelevanceCode.NEWS_ARTICLE
                msg = "News article detected"
            elif features.opinion_score >= 5:
                code = RelevanceCode.OPINION_EDITORIAL
                msg = "Opinion/editorial content detected"
            elif features.review_score >= 5:
                code = RelevanceCode.REVIEW_SUMMARY
                msg = "Literature review detected"
            elif features.case_report_score >= 5:
                code = RelevanceCode.CASE_REPORT
                msg = "Case report detected"
            else:
                code = RelevanceCode.NON_RESEARCH
                msg = "Non-research content detected"
            
            reasoning.append(f"Strong non-research evidence: {non_research_score} >> {research_score}")
            return self._create_result(
                code,
                confidence,
                f"{msg} (score: +{research_score}, -{non_research_score})",
                features,
                reasoning
            )
            
        elif research_score >= self.research_thresholds['moderate_research']:
            reasoning.append(f"Moderate research evidence: {research_score}")
            return self._create_result(
                RelevanceCode.RESEARCH_ABSTRACT,
                confidence * 0.8,  # Lower confidence for borderline cases
                f"Likely research abstract (score: +{research_score}, -{non_research_score})",
                features,
                reasoning
            )
            
        else:
            reasoning.append(f"Ambiguous classification: research={research_score}, non-research={non_research_score}")
            return self._create_result(
                RelevanceCode.AMBIGUOUS,
                1.0 - confidence,  # Low confidence for ambiguous
                f"Unclear classification (score: +{research_score}, -{non_research_score})",
                features,
                reasoning
            )
    
    def _create_result(self, code: RelevanceCode, confidence: float, message: str,
                      features: HeuristicFeatures, reasoning: List[str]) -> RelevanceResult:
        """Create a RelevanceResult object"""
        return RelevanceResult(
            code=code,
            confidence=confidence,
            message=message,
            features=features,
            reasoning=reasoning
        )
    def _create_empty_features(self) -> HeuristicFeatures:
        """Create empty features object for error cases"""
        return HeuristicFeatures(
            word_count=0, sentence_count=0, avg_sentence_length=0,
            study_design_score=0, methodology_score=0, statistical_score=0, structure_score=0,
            news_score=0, opinion_score=0, review_score=0, case_report_score=0,
            numerical_density=0, technical_terms_ratio=0, passive_voice_count=0,
            detected_patterns=[]
        )
















def run_check(abstract : str):# just a wrapper method
    classifier = HeuristicAbstractRelevanceChecker()
    result = classifier.check_relevance(abstract)
    return result