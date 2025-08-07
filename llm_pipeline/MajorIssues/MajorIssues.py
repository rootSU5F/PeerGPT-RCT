"""
LLM pipeline for identifying major issues in clinical trial abstracts.
Following Tamara's approach: high-level prompt without specifying what issues to look for.
"""
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import List, Dict
import logging
from dotenv import load_dotenv

load_dotenv()  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MajorIssues:
    """
    LLM pipeline for evaluating clinical trial internal validity and external generalizability.
    Uses high-level prompt approach as recommended by Tamara.
    """

    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """
        Initialize the Major Issues classifier.
        
        Args:
            api_key: Groq API key (optional if set as environment variable)
            model_name: Model to use for evaluation
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        elif not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0,  # Keep deterministic for consistency
            max_tokens=2000  # Enough for detailed analysis
        )
        
        # Tamara's recommended high-level prompt with medical context
        self.prompt = PromptTemplate.from_template("""
You are a clinical epidemiologist evaluating randomized controlled trials for evidence-based medicine. 

Your task: Please evaluate the internal validity and external generalizability of the following randomized controlled trial. Consider how the study design, execution, and findings would impact clinical decision-making and patient care.

Abstract:
{abstract}

Please provide a structured evaluation covering both internal validity and external generalizability concerns.
""")
        
        # Create chain
        self.chain = self.prompt | self.llm

    def identify_major_issues(self, abstract: str) -> str:
        """
        Identify major issues in the abstract using LLM.
        
        Args:
            abstract: Clinical trial abstract text
            
        Returns:
            Evaluation string or None if error
        """
        try: 
            # Input validation
            abstract = abstract.strip()
            if not abstract:
                logger.warning("Empty abstract provided")
                return None

            # Get LLM evaluation
            result = self.chain.invoke({"abstract": abstract})
            evaluation = result.content.strip()
            
            logger.info("Successfully evaluated abstract")
            return evaluation
        
        except Exception as e:
            error_msg = f"Error identifying major issues: {str(e)}"
            logger.error(error_msg)
            return None

    def batch_evaluate(self, abstracts: List[str]) -> List[Dict[str, any]]:
        """
        Evaluate multiple abstracts in batch.
        
        Args:
            abstracts: List of abstract strings
            
        Returns:
            List of dictionaries with results and metadata
        """
        results = []
        for i, abstract in enumerate(abstracts):
            logger.info(f"Evaluating abstract {i+1}/{len(abstracts)}")
            evaluation = self.identify_major_issues(abstract)
            
            results.append({
                'index': i,
                'evaluation': evaluation,
                'abstract_length': len(abstract),
                'preview': abstract[:100] + "..." if len(abstract) > 100 else abstract,
                'success': evaluation is not None
            })
        
        return results


def create_major_issues_checker(api_key: str = None):
    """
    Factory function to create MajorIssues classifier.
    
    Args:
        api_key: Optional API key (will use environment variable if not provided)
        
    Returns:
        MajorIssues instance or None if initialization fails
    """
    try:
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        
        return MajorIssues(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize MajorIssues classifier: {str(e)}")
        return None


def main():
    """Example usage of the Major Issues pipeline"""
    
    # Initialize checker
    checker = create_major_issues_checker()
    if not checker:
        print("Failed to initialize MajorIssues checker")
        return
    
    # Sample abstracts for testing
    test_abstracts = [
        """
Abstract
Background: Among infants with isolated cleft palate, whether primary surgery at 6 months of age is more beneficial than surgery at 12 months of age with respect to speech outcomes, hearing outcomes, dentofacial development, and safety is unknown.

Methods: We randomly assigned infants with nonsyndromic isolated cleft palate, in a 1:1 ratio, to undergo standardized primary surgery at 6 months of age (6-month group) or at 12 months of age (12-month group) for closure of the cleft. Standardized assessments of quality-checked video and audio recordings at 1, 3, and 5 years of age were performed independently by speech and language therapists who were unaware of the trial-group assignments. The primary outcome was velopharyngeal insufficiency at 5 years of age, defined as a velopharyngeal composite summary score of at least 4 (scores range from 0 to 6, with higher scores indicating greater severity). Secondary outcomes included speech development, postoperative complications, hearing sensitivity, dentofacial development, and growth.

Results: We randomly assigned 558 infants at 23 centers across Europe and South America to undergo surgery at 6 months of age (281 infants) or at 12 months of age (277 infants). Speech recordings from 235 infants (83.6%) in the 6-month group and 226 (81.6%) in the 12-month group were analyzable. Insufficient velopharyngeal function at 5 years of age was observed in 21 of 235 infants (8.9%) in the 6-month group as compared with 34 of 226 (15.0%) in the 12-month group (risk ratio, 0.59; 95% confidence interval, 0.36 to 0.99; P = 0.04). Postoperative complications were infrequent and similar in the 6-month and 12-month groups. Four serious adverse events were reported (three in the 6-month group and one in the 12-month group) and had resolved at follow-up.

Conclusions: Medically fit infants who underwent primary surgery for isolated cleft palate in adequately resourced settings at 6 months of age were less likely to have velopharyngeal insufficiency at the age of 5 years than those who had surgery at 12 months of age. (Funded by the National Institute of Dental and Craniofacial Research; TOPS ClinicalTrials.gov number, NCT00993551.).
        """
    ]
    
    print("=== SINGLE ABSTRACT EVALUATION ===")
    evaluation = checker.identify_major_issues(test_abstracts[0])
    
    if evaluation:
        print("EVALUATION RESULTS:")
        print("=" * 50)
        print(evaluation)
    else:
        print("Evaluation failed")
    
    print("\n=== BATCH EVALUATION ===")
    batch_results = checker.batch_evaluate(test_abstracts)
    
    for result in batch_results:
        print(f"\nAbstract {result['index'] + 1}:")
        print(f"  Success: {result['success']}")
        print(f"  Length: {result['abstract_length']} characters")
        print(f"  Preview: {result['preview']}")
        if result['evaluation']:
            print(f"  Evaluation: {result['evaluation'][:200]}...")


if __name__ == "__main__":
    main()

