import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCTBiasDetector:
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """
        Initialize the RCT bias detector.
        
        Args:
            api_key: Groq API key (optional if set as env var)
            model_name: Model to use for bias detection
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        elif not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=500
        )
        
        # Define heuristic messages (kept for reference but not used in this simple approach)
        self.heuristic_messages = {
            "small_sample": "This was a small study, which increases the possibility that results are due to chance alone and that not all baseline confounders were balanced.",
            "relatively_small_sample": "This was a relatively small study, which limits the ability to detect a small effect size.",
            "low_events": "The relatively low event rate increases the possibility that results are due to chance and typically leads to wide confidence intervals.",
            "composite_outcome": "The use of a composite outcome introduces two key limitations. First, if the components vary in clinical significance, the overall results become harder to interpret. Second, composite outcomes are driven by the most frequent event, which is often the event of least clinical importance.",
            "no_placebo_standard_care": "The abstract indicated this trial compared a drug to standard of care. The lack of a placebo introduces the potential for time-varying confounding and outcome adjudication bias.",
            "no_placebo_mentioned": "The abstract doesn't mention the use of a placebo. If the trial lacked a placebo, this introduces the potential for time-varying confounding and outcome adjudication bias.",
            "no_blinding_mentioned": "The abstract doesn't mention blinding. The lack of blinding introduces the potential for time-varying confounding and outcome adjudication bias.",
            "not_blinded": "The lack of blinding introduces the potential for time-varying confounding and outcome adjudication bias."
        }
        
        # Bias detection prompt
        self.prompt = PromptTemplate.from_template("""
You are an expert clinical trial methodologist. Analyze this RCT abstract for specific methodological issues.

ABSTRACT:
{abstract}

Check for these conditions and return the EXACT corresponding message if found:

1. SAMPLE SIZE:
   - IF sample size < 500 → "This was a small study, which increases the possibility that results are due to chance alone and that not all baseline confounders were balanced."
   - IF sample size 500-1000 → "This was a relatively small study, which limits the ability to detect a small effect size."

2. EVENT RATE:
   - IF < 30 events in either treatment group → "The relatively low event rate increases the possibility that results are due to chance and typically leads to wide confidence intervals."

3. COMPOSITE OUTCOME:
   - IF primary outcome is composite → "The use of a composite outcome introduces two key limitations. First, if the components vary in clinical significance, the overall results become harder to interpret. Second, composite outcomes are driven by the most frequent event, which is often the event of least clinical importance."

4. PLACEBO CONTROL:
   - IF drug vs standard care → "The abstract indicated this trial compared a drug to standard of care. The lack of a placebo introduces the potential for time-varying confounding and outcome adjudication bias."
   - IF no mention of placebo → "The abstract doesn't mention the use of a placebo. If the trial lacked a placebo, this introduces the potential for time-varying confounding and outcome adjudication bias."

5. BLINDING:
   - IF not blinded/open-label → "The lack of blinding introduces the potential for time-varying confounding and outcome adjudication bias."
   - IF no mention of blinding → "The abstract doesn't mention blinding. The lack of blinding introduces the potential for time-varying confounding and outcome adjudication bias."

Return only the applicable messages, one per line. If no issues found, return "No methodological issues detected."
""")
        
        # Create chain
        self.chain = self.prompt | self.llm
    
    def detect_bias(self, abstract: str) -> List[str]:
        """
        Detect bias and methodological issues in an RCT abstract.
        
        Args:
            abstract: The abstract text to analyze
            
        Returns:
            List of bias messages
        """
        try:
            # Clean the abstract
            abstract = abstract.strip()
            if not abstract:
                logger.warning("Empty abstract provided")
                return ["Error: Empty abstract provided"]
            
            # Generate analysis
            result = self.chain.invoke({"abstract": abstract})
            response_text = result.content.strip()
            
            # Split response into individual messages
            if response_text == "No methodological issues detected.":
                return []
            
            # Split by lines and clean up
            messages = [msg.strip() for msg in response_text.split('\n') if msg.strip()]
            return messages
                
        except Exception as e:
            logger.error(f"Error detecting bias: {e}")
            return [f"Error: Failed to analyze abstract - {str(e)}"]
    
    def format_results(self, messages: List[str]) -> str:
        """
        Format analysis results for display.
        
        Args:
            messages: List of bias messages
            
        Returns:
            Formatted string of results
        """
        if not messages:
            return "No methodological issues detected."
        
        if len(messages) == 1 and messages[0].startswith("Error:"):
            return messages[0]
        
        result_lines = [f"Detected {len(messages)} methodological issue(s):", ""]
        
        for i, message in enumerate(messages, 1):
            result_lines.append(f"{i}. {message}")
            result_lines.append("")
        
        return "\n".join(result_lines)

# Initialize detector
def create_bias_detector():
    """Factory function to create bias detector."""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        return RCTBiasDetector(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize bias detector: {e}")
        return None

# Example usage
def main():
    detector = create_bias_detector()
    if not detector:
        print("Failed to initialize bias detector")
        return
    
    # Test abstract with multiple potential issues
    test_abstract = """
Background: The effectiveness of a new antihypertensive drug versus standard care in reducing cardiovascular events is unknown.

Methods: We conducted an open-label randomized trial comparing the new drug (n=85) versus standard care (n=92) in patients with hypertension. The primary composite outcome included myocardial infarction, stroke, and cardiovascular death measured over 12 months.

Results: The primary outcome occurred in 8 patients in the treatment group and 15 patients in the control group (hazard ratio 0.52, 95% CI 0.22-1.23, p=0.14).

Conclusions: The new drug showed a trend toward reducing cardiovascular events but did not reach statistical significance.
"""
    
    print("=== RCT Bias Detector ===\n")
    print("Abstract to analyze:")
    print("-" * 50)
    print(test_abstract)
    
    print("\n" + "=" * 50)
    print("BIAS ANALYSIS RESULTS:")
    print("=" * 50)
    
    # Detect bias
    messages = detector.detect_bias(test_abstract)
    
    # Display formatted results
    formatted_results = detector.format_results(messages)
    print(formatted_results)
    
    # Show individual messages
    print("\n" + "=" * 50)
    print("INDIVIDUAL BIAS MESSAGES:")
    print("=" * 50)
    for i, message in enumerate(messages, 1):
        print(f"{i}. {message}")
        print()

if __name__ == "__main__":
    main()