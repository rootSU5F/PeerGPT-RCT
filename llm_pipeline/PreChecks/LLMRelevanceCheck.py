import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAbstractClassifier:
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """
        Initialize the research abstract classifier.
        
        Args:
            api_key: Groq API key (optional if set as env var)
            model_name: Model to use for classification
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        elif not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0,  # Consistent results
            max_tokens=10   # Short response
        )
        
        # Improved prompt with more specific criteria
        self.prompt = PromptTemplate.from_template("""
You are a scientific reviewer. Determine whether the following abstract describes a research study.

Definitions:
- RESEARCH: Original empirical study with:
  * Clear methodology (experimental design, data collection methods)
  * Defined participants/subjects/sample
  * Data analysis and results presentation
  * Evidence-based conclusions
  * Examples: RCTs, cohort studies, case-control studies, experimental studies

- NON-RESEARCH: 
  * Literature reviews, systematic reviews, meta-analyses
  * Editorials, commentaries, opinions
  * Guidelines, recommendations
  * Case reports (single cases without systematic analysis)
  * Theoretical papers without empirical data

Abstract:
{abstract}

Analyze the abstract and answer with exactly one word: "research" or "non-research".
""")
        
        # Create chain
        self.chain = self.prompt | self.llm
    
    def is_research_abstract(self, abstract: str) -> bool:
        """
        Classify whether an abstract describes research.
        
        Args:
            abstract: The abstract text to classify
            
        Returns:
            bool: True if research, False if non-research
        """
        try:
            # Clean the abstract
            abstract = abstract.strip()
            if not abstract:
                logger.warning("Empty abstract provided")
                return False
            
            # Get prediction
            result = self.chain.invoke({'abstract': abstract})
            answer = result.content.strip().lower()
            
            # More robust parsing
            if 'research' in answer and 'non-research' not in answer:
                return True
            elif 'non-research' in answer:
                return False
            else:
                logger.warning(f"Unexpected response: {answer}")
                # Fallback: look for research indicators in response
                return 'research' in answer
                
        except Exception as e:
            logger.error(f"Error classifying abstract: {e}")
            return False
    
    def batch_classify(self, abstracts: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple abstracts at once.
        
        Args:
            abstracts: List of abstract texts
            
        Returns:
            List of dictionaries with results and metadata
        """
        results = []
        for i, abstract in enumerate(abstracts):
            try:
                is_research = self.is_research_abstract(abstract)
                results.append({
                    'index': i,
                    'is_research': is_research,
                    'abstract_length': len(abstract),
                    'preview': abstract[:100] + "..." if len(abstract) > 100 else abstract
                })
            except Exception as e:
                logger.error(f"Error processing abstract {i}: {e}")
                results.append({
                    'index': i,
                    'is_research': None,
                    'error': str(e),
                    'abstract_length': len(abstract) if abstract else 0
                })
        return results
    
    def get_classification_with_confidence(self, abstract: str) -> Dict[str, any]:
        """
        Get classification with additional metadata.
        
        Args:
            abstract: The abstract text to classify
            
        Returns:
            Dictionary with classification and metadata
        """
        try:
            result = self.chain.invoke({'abstract': abstract})
            raw_response = result.content.strip().lower()
            
            # Determine classification
            is_research = self.is_research_abstract(abstract)
            
            # Simple confidence estimation based on response clarity
            confidence = "high" if raw_response in ["research", "non-research"] else "low"
            
            return {
                'is_research': is_research,
                'raw_response': raw_response,
                'confidence': confidence,
                'abstract_length': len(abstract),
                'has_methods_section': 'method' in abstract.lower(),
                'has_results_section': 'result' in abstract.lower(),
                'has_participants': any(word in abstract.lower() for word in ['participant', 'subject', 'patient'])
            }
            
        except Exception as e:
            logger.error(f"Error in detailed classification: {e}")
            return {
                'is_research': None,
                'error': str(e),
                'abstract_length': len(abstract) if abstract else 0
            }

# Initialize classifier
def create_classifier():
    """Factory function to create classifier with error handling."""
    try:
        #os.environ["GROQ_API_KEY"] = "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"
        
        api_key = os.environ.get("GROQ_API_KEY")
        return ResearchAbstractClassifier(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        return None

# Example usage
def main():
    classifier = create_classifier()
    if not classifier:
        print("Failed to initialize classifier")
        return
    
    # Your test abstract
    abstract1 = """
Abstract
Background: Among infants with isolated cleft palate, whether primary surgery at 6 months of age is more beneficial than surgery at 12 months of age with respect to speech outcomes, hearing outcomes, dentofacial development, and safety is unknown.

Methods: We randomly assigned infants with nonsyndromic isolated cleft palate, in a 1:1 ratio, to undergo standardized primary surgery at 6 months of age (6-month group) or at 12 months of age (12-month group) for closure of the cleft. Standardized assessments of quality-checked video and audio recordings at 1, 3, and 5 years of age were performed independently by speech and language therapists who were unaware of the trial-group assignments. The primary outcome was velopharyngeal insufficiency at 5 years of age, defined as a velopharyngeal composite summary score of at least 4 (scores range from 0 to 6, with higher scores indicating greater severity). Secondary outcomes included speech development, postoperative complications, hearing sensitivity, dentofacial development, and growth.

Results: We randomly assigned 558 infants at 23 centers across Europe and South America to undergo surgery at 6 months of age (281 infants) or at 12 months of age (277 infants). Speech recordings from 235 infants (83.6%) in the 6-month group and 226 (81.6%) in the 12-month group were analyzable. Insufficient velopharyngeal function at 5 years of age was observed in 21 of 235 infants (8.9%) in the 6-month group as compared with 34 of 226 (15.0%) in the 12-month group (risk ratio, 0.59; 95% confidence interval, 0.36 to 0.99; P = 0.04). Postoperative complications were infrequent and similar in the 6-month and 12-month groups. Four serious adverse events were reported (three in the 6-month group and one in the 12-month group) and had resolved at follow-up.

Conclusions: Medically fit infants who underwent primary surgery for isolated cleft palate in adequately resourced settings at 6 months of age were less likely to have velopharyngeal insufficiency at the age of 5 years than those who had surgery at 12 months of age. (Funded by the National Institute of Dental and Craniofacial Research; TOPS ClinicalTrials.gov number, NCT00993551.).
"""
    
    # Test cases
    test_abstracts = [
        abstract1,
        """
        This review examines current literature on machine learning applications in healthcare.
        We searched PubMed and identified 150 relevant papers published between 2020-2024.
        Common applications include diagnostic imaging, drug discovery, and patient monitoring.
        """,
        """
        Editorial: The future of personalized medicine looks promising but faces significant
        challenges including data privacy, algorithmic bias, and healthcare accessibility.
        Policymakers must address these issues proactively.
        """
    ]
    
    print("=== Simple Classification ===")
    for i, abstract in enumerate(test_abstracts):
        result = classifier.is_research_abstract(abstract)
        print(f"Abstract {i+1}: {'RESEARCH' if result else 'NON-RESEARCH'}")
    
    print("\n=== Detailed Classification ===")
    for i, abstract in enumerate(test_abstracts):
        result = classifier.get_classification_with_confidence(abstract)
        print(f"\nAbstract {i+1}:")
        print(f"  Classification: {'RESEARCH' if result.get('is_research') else 'NON-RESEARCH'}")
        print(f"  Confidence: {result.get('confidence', 'unknown')}")
        print(f"  Length: {result.get('abstract_length', 0)} characters")
        print(f"  Has methods: {result.get('has_methods_section', False)}")
        print(f"  Has results: {result.get('has_results_section', False)}")
    
if __name__ == "__main__":
    main()