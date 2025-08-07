import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import logging
from dotenv import load_dotenv

load_dotenv()  


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCTSummarizer:
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """
        Initialize the RCT abstract summarizer.
        
        Args:
            api_key: Groq API key (optional if set as env var)
            model_name: Model to use for summarization
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        elif not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.3,  # Slightly creative for varied phrasing
            max_tokens=250    # Ensure under 200 words
        )
        
        # Structured summarization prompt
        self.prompt = PromptTemplate.from_template("""
You are an expert medical editor. Create a concise, structured summary of this RCT abstract for editorial review purposes.

Requirements:
- Maximum 200 words
- Clear, professional language
- Structured format with key sections
- Focus on actionable findings
- Highlight clinical significance

Structure your summary as follows:
**Objective:** [Brief study aim]
**Design:** [Study type, duration, setting]
**Participants:** [Sample size, population, key characteristics]
**Intervention:** [What was tested vs control]
**Primary Outcome:** [Main endpoint measured]
**Key Results:** [Most important findings with key statistics]
**Clinical Significance:** [Practical implications for practice]

Abstract:
{abstract}

Summary:
""")
        
        # Create chain
        self.chain = self.prompt | self.llm
    
    def summarize_abstract(self, abstract: str) -> str:
        """
        Generate a structured summary of an RCT abstract.
        
        Args:
            abstract: The abstract text to summarize
            
        Returns:
            str: Structured summary under 200 words
        """
        try:
            # Clean the abstract
            abstract = abstract.strip()
            if not abstract:
                logger.warning("Empty abstract provided")
                return "Error: Empty abstract provided"
            
            # Generate summary
            result = self.chain.invoke({"abstract": abstract})
            summary = result.content.strip()
            
            # Word count check
            word_count = len(summary.split())
            if word_count > 200:
                logger.warning(f"Summary exceeds 200 words ({word_count} words)")
            
            return summary
                
        except Exception as e:
            logger.error(f"Error summarizing abstract: {e}")
            return f"Error: Failed to generate summary - {str(e)}"
    
    def get_word_count(self, text: str) -> int:
        """Get word count of text."""
        return len(text.split())

# Initialize summarizer
def create_summarizer():
    """Factory function to create summarizer."""
    try:
        #os.environ["GROQ_API_KEY"] = "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"
        api_key = os.environ.get('GROQ_API_KEY')
        return RCTSummarizer(api_key= api_key)
    except Exception as e:
        logger.error(f"Failed to initialize summarizer: {e}")
        return None

# Example usage
def main():
    summarizer = create_summarizer()
    if not summarizer:
        print("Failed to initialize summarizer")
        return
    
    # Test abstract - the cleft palate RCT
    test_abstract = """
Abstract
Background: Among infants with isolated cleft palate, whether primary surgery at 6 months of age is more beneficial than surgery at 12 months of age with respect to speech outcomes, hearing outcomes, dentofacial development, and safety is unknown.

Methods: We randomly assigned infants with nonsyndromic isolated cleft palate, in a 1:1 ratio, to undergo standardized primary surgery at 6 months of age (6-month group) or at 12 months of age (12-month group) for closure of the cleft. Standardized assessments of quality-checked video and audio recordings at 1, 3, and 5 years of age were performed independently by speech and language therapists who were unaware of the trial-group assignments. The primary outcome was velopharyngeal insufficiency at 5 years of age, defined as a velopharyngeal composite summary score of at least 4 (scores range from 0 to 6, with higher scores indicating greater severity). Secondary outcomes included speech development, postoperative complications, hearing sensitivity, dentofacial development, and growth.

Results: We randomly assigned 558 infants at 23 centers across Europe and South America to undergo surgery at 6 months of age (281 infants) or at 12 months of age (277 infants). Speech recordings from 235 infants (83.6%) in the 6-month group and 226 (81.6%) in the 12-month group were analyzable. Insufficient velopharyngeal function at 5 years of age was observed in 21 of 235 infants (8.9%) in the 6-month group as compared with 34 of 226 (15.0%) in the 12-month group (risk ratio, 0.59; 95% confidence interval, 0.36 to 0.99; P = 0.04). Postoperative complications were infrequent and similar in the 6-month and 12-month groups. Four serious adverse events were reported (three in the 6-month group and one in the 12-month group) and had resolved at follow-up.

Conclusions: Medically fit infants who underwent primary surgery for isolated cleft palate in adequately resourced settings at 6 months of age were less likely to have velopharyngeal insufficiency at the age of 5 years than those who had surgery at 12 months of age. (Funded by the National Institute of Dental and Craniofacial Research; TOPS ClinicalTrials.gov number, NCT00993551.).
"""
    
    print("=== RCT Abstract Summarizer ===\n")
    print("Original Abstract:")
    print("-" * 50)
    print(test_abstract)
    
    print(f"\nOriginal word count: {summarizer.get_word_count(test_abstract)} words")
    
    print("\n" + "=" * 50)
    print("STRUCTURED SUMMARY:")
    print("=" * 50)
    
    summary = summarizer.summarize_abstract(test_abstract)
    
    print(summary)
    
    print(f"\nSummary word count: {summarizer.get_word_count(summary)} words")
    
    # Test with another type of study
    test_abstract2 = """
Background: The effectiveness of mindfulness-based stress reduction (MBSR) for chronic pain management remains unclear.
Methods: We conducted a randomized controlled trial comparing 8-week MBSR program (n=75) versus usual care (n=73) in adults with chronic low back pain. Primary outcome was pain intensity measured by Visual Analog Scale at 8 weeks.
Results: MBSR group showed significantly greater reduction in pain scores compared to control (mean difference -2.1, 95% CI: -3.2 to -1.0, p<0.001). Secondary outcomes including disability and quality of life also improved significantly.
Conclusions: MBSR is effective for reducing chronic low back pain intensity and improving functional outcomes.
"""
    
    print("\n" + "=" * 70)
    print("SECOND EXAMPLE:")
    print("=" * 70)
    
    summary2 = summarizer.summarize_abstract(test_abstract2)
    print(summary2)
    print(f"\nSummary word count: {summarizer.get_word_count(summary2)} words")

if __name__ == "__main__":
    main()