import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import Dict
import logging
from dotenv import load_dotenv

load_dotenv()  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCTClassifier:
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """
        Initialize the RCT vs Non-RCT classifier (LLM-only).
        
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
        
        # LLM-only prompt for RCT identification
        self.prompt = PromptTemplate.from_template("""
You are an expert in clinical research methodology. Determine whether the following abstract describes a Randomized Controlled Trial (RCT) or a Non-RCT study.

DEFINITIONS:

RCT (Randomized Controlled Trial):
- Participants are RANDOMLY ASSIGNED/ALLOCATED to different groups
- Has control group(s) and intervention group(s)
- Key phrases: "randomly assigned", "randomized", "random allocation", "randomization", "RCT"
- Includes parallel RCTs, crossover RCTs, cluster RCTs, factorial RCTs

NON-RCT:
- Observational studies: cohort, case-control, cross-sectional
- Quasi-experimental studies (non-random assignment)
- Case series, case reports
- Systematic reviews, meta-analyses
- Laboratory studies, animal studies
- Surveys, qualitative studies
- Before-after studies without randomization

CRITICAL: Focus on whether participants were RANDOMLY ASSIGNED to interventions, not just whether the study mentions "random" in other contexts (like random sampling).

Abstract:
{abstract}

Based on the study design described, answer with exactly one word: "RCT" or "NON-RCT".
""")
        
        # Create chain
        self.chain = self.prompt | self.llm
    
    def is_rct(self, abstract: str) -> bool:
        """
        Classify whether an abstract describes an RCT using LLM only.
        
        Args:
            abstract: The abstract text to classify
            
        Returns:
            bool: True if RCT, False if Non-RCT
        """
        try:
            # Clean the abstract
            abstract = abstract.strip()
            if not abstract:
                logger.warning("Empty abstract provided")
                return False
            
            # Get LLM prediction
            result = self.chain.invoke({"abstract": abstract})
            answer = result.content.strip().upper()
            
            # Parse response - LLM only, no fallback
            if 'RCT' in answer and 'NON-RCT' not in answer:
                return True
            elif 'NON-RCT' in answer:
                return False
            else:
                logger.warning(f"Unexpected LLM response: {answer}")
                # Return False for unclear responses (conservative approach)
                return False
                
        except Exception as e:
            logger.error(f"Error with LLM classification: {e}")
            return False
    


# Initialize classifier
def create_rct_classifier():
    """Factory function to create RCT classifier."""
    try:
        #os.environ["GROQ_API_KEY"] = "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"
        api_key = os.environ.get("GROQ_API_KEY")
        return RCTClassifier(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize RCT classifier: {e}")
        return None

# Example usage
def main():
    classifier = create_rct_classifier()
    if not classifier:
        print("Failed to initialize RCT classifier")
        return
    
    # Test abstracts
    test_abstracts = [
        # Clear RCT
        """
        Background: Among infants with isolated cleft palate, whether primary surgery at 6 months of age is more beneficial than surgery at 12 months of age with respect to speech outcomes, hearing outcomes, dentofacial development, and safety is unknown.
        Methods: We randomly assigned infants with nonsyndromic isolated cleft palate, in a 1:1 ratio, to undergo standardized primary surgery at 6 months of age (6-month group) or at 12 months of age (12-month group) for closure of the cleft.
        Results: We randomly assigned 558 infants at 23 centers across Europe and South America.
        """,
        
        # Cohort study (Non-RCT)
        """
        Background: Long-term effects of childhood obesity on cardiovascular health are not well understood.
        Methods: We followed 1,200 children for 20 years in a prospective cohort study. Participants were recruited from schools and assessed annually for BMI, blood pressure, and lipid profiles.
        Results: Children with obesity had 2.3 times higher risk of developing hypertension in adulthood.
        """,
        
        # Case-control study (Non-RCT)
        """
        Objective: To investigate risk factors for pancreatic cancer.
        Methods: We conducted a case-control study comparing 300 pancreatic cancer patients with 600 matched controls. Data on smoking, diet, and family history were collected.
        Results: Smoking was associated with increased risk (OR=3.2, 95% CI: 2.1-4.8).
        """,
        
        # Crossover RCT
        """
        Background: Comparing two insulin regimens in type 1 diabetes.
        Methods: In this randomized crossover trial, 45 patients were randomly allocated to receive either insulin A or insulin B for 3 months, followed by a washout period, then crossed over to the alternative treatment.
        Results: Both treatments showed similar efficacy.
        """
    ]
    
    print("=== LLM-Only RCT vs Non-RCT Classification ===\n")
    
    for i, abstract in enumerate(test_abstracts):
        print(f"--- Abstract {i+1} ---")
        
        # Simple classification
        is_rct = classifier.is_rct(abstract)
        print(f"Classification: {'RCT' if is_rct else 'NON-RCT'}")
        print(f"Preview: {abstract[:80]}...\n")

if __name__ == "__main__":
    main()