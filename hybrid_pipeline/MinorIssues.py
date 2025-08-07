import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCTMinorIssuesDetector:
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """
        Initialize the RCT minor issues detector.
        
        Args:
            api_key: Groq API key (optional if set as env var)
            model_name: Model to use for minor issues detection
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        elif not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=700
        )
        
        # Define minor issue heuristics
        self.minor_issues = {
            "short_followup": {
                "cutpoint": "Less than 1 year",
                "message": "The follow-up duration is under one year, which may be insufficient to capture meaningful clinical outcomes, especially for chronic conditions or interventions with delayed effects. This limits the ability to assess sustained efficacy, adverse events, and long-term risk."
            },
            "followup_not_provided": {
                "cutpoint": "Not provided",
                "message": "The duration of follow-up was not provided in the abstract. For studies less than a year, important clinical outcomes may not have had sufficient time to develop, limiting the ability to draw meaningful conclusions about long-term risks or benefits."
            },
            "single_center_assumed": {
                "cutpoint": "Not mentioned [assume single centre]",
                "message": "The abstract lacked details on whether the study was single-centered or multicentered. This often means the study was single-centered. Single-centre studies, even RCTs, are often hard to replicate."
            },
            "outcome_timeline_not_mentioned": {
                "cutpoint": "Not mentioned",
                "message": "The timeline for assessing the primary outcome is not clearly stated. In RCTs, the timing of outcome measurement is critical– assessing outcomes too early may underestimate treatment effects, while excessively short timelines may miss delayed benefits or harms. Lack of this information limits interpretability and clinical relevance."
            },
            "short_outcome_timeline": {
                "cutpoint": "< 30 days",
                "message": "The primary outcome was assessed within a short time frame (≤30 days), which may be insufficient to observe meaningful clinical effects, particularly for chronic conditions or interventions with delayed impact. Short timelines risk underestimating true benefits or harms. Timeline appropriateness should be interpreted in the context of the disease: while 30-day outcomes may be suitable for acute settings (e.g., stroke, MI), longer follow-up is often required for interventions in chronic diseases (e.g., diabetes, cancer, heart failure)."
            },
            "industry_funding": {
                "cutpoint": "funded by [Company Name] or sponsored by [PharmaCo]",
                "message": "Industry-sponsored trials are more likely to emphasize benefits and underestimate harms."
            },
            "age_under_70": {
                "cutpoint": "Age < 70",
                "message": "The mean or median age of participants is under 70 years, which may limit the generalizability of findings to older adults."
            },
            "low_female_participation": {
                "cutpoint": "Sex < 50% F",
                "message": "This study population includes fewer than 50% females, limiting the applicability of results across sexes."
            },
            "itt_not_mentioned": {
                "cutpoint": "Not mentioned",
                "message": "ITT is the ideal approach for superiority trials but not for non-inferiority trials. Lack of clarity on the analytic approach limits interpretability."
            }
        }
        
        # Minor issues detection prompt
        self.prompt = PromptTemplate.from_template("""
You are an expert clinical trial methodologist focused on identifying MINOR methodological issues in RCT abstracts.

ABSTRACT:
{abstract}

Analyze for these MINOR ISSUES and return the EXACT corresponding message if the condition is met:

1. FOLLOW-UP DURATION:
   - IF follow-up duration < 1 year → "The follow-up duration is under one year, which may be insufficient to capture meaningful clinical outcomes, especially for chronic conditions or interventions with delayed effects. This limits the ability to assess sustained efficacy, adverse events, and long-term risk."
   - IF no follow-up duration mentioned → "The duration of follow-up was not provided in the abstract. For studies less than a year, important clinical outcomes may not have had sufficient time to develop, limiting the ability to draw meaningful conclusions about long-term risks or benefits."

2. STUDY CENTER TYPE:
   - IF no mention of "multicenter" OR "multicentre" OR "single center" → "The abstract lacked details on whether the study was single-centered or multicentered. This often means the study was single-centered. Single-centre studies, even RCTs, are often hard to replicate."

3. PRIMARY OUTCOME ASSESSMENT TIMELINE:
   - IF no specific timeline mentioned for when primary outcome was assessed → "The timeline for assessing the primary outcome is not clearly stated. In RCTs, the timing of outcome measurement is critical– assessing outcomes too early may underestimate treatment effects, while excessively short timelines may miss delayed benefits or harms. Lack of this information limits interpretability and clinical relevance."
   - IF primary outcome assessed within ≤30 days → "The primary outcome was assessed within a short time frame (≤30 days), which may be insufficient to observe meaningful clinical effects, particularly for chronic conditions or interventions with delayed impact. Short timelines risk underestimating true benefits or harms. Timeline appropriateness should be interpreted in the context of the disease: while 30-day outcomes may be suitable for acute settings (e.g., stroke, MI), longer follow-up is often required for interventions in chronic diseases (e.g., diabetes, cancer, heart failure)."

4. FUNDING SOURCE:
   - IF mentions industry/company funding or sponsorship → "Industry-sponsored trials are more likely to emphasize benefits and underestimate harms."

5. GENERALIZABILITY CONCERNS:
   - IF mean/median age < 70 years → "The mean or median age of participants is under 70 years, which may limit the generalizability of findings to older adults."
   - IF < 50% female participants → "This study population includes fewer than 50% females, limiting the applicability of results across sexes."

6. STATISTICAL ANALYSIS:
   - IF no mention of "intention-to-treat" OR "ITT" analysis → "ITT is the ideal approach for superiority trials but not for non-inferiority trials. Lack of clarity on the analytic approach limits interpretability."

Return only the applicable messages, one per line. If no minor issues are found, return "No minor methodological issues detected."
""")
        
        # Create chain
        self.chain = self.prompt | self.llm
    
    def detect_minor_issues(self, abstract: str) -> List[str]:
        """
        Detect minor methodological issues in an RCT abstract.
        
        Args:
            abstract: The abstract text to analyze
            
        Returns:
            List of minor issue messages
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
            
            # Handle no issues case
            if response_text == "No minor methodological issues detected.":
                return []
            
            # Split by lines and clean up
            messages = [msg.strip() for msg in response_text.split('\n') if msg.strip()]
            
            # Filter out any non-message lines (should be substantial text)
            filtered_messages = []
            for msg in messages:
                if len(msg) > 30:  # Minor issue messages are long
                    filtered_messages.append(msg)
            
            return filtered_messages
                
        except Exception as e:
            logger.error(f"Error detecting minor issues: {e}")
            return [f"Error: Failed to analyze abstract - {str(e)}"]
    
    def categorize_issues(self, messages: List[str]) -> Dict[str, List[str]]:
        """
        Categorize detected issues by type.
        
        Args:
            messages: List of issue messages
            
        Returns:
            Dictionary with categorized issues
        """
        categories = {
            "Follow-up Duration": [],
            "Study Design": [],
            "Outcome Assessment": [],
            "Funding": [],
            "Generalizability": [],
            "Statistical Analysis": []
        }
        
        for msg in messages:
            if "follow-up" in msg.lower():
                categories["Follow-up Duration"].append(msg)
            elif "single-centered" in msg.lower():
                categories["Study Design"].append(msg)
            elif "timeline" in msg.lower() and "outcome" in msg.lower():
                categories["Outcome Assessment"].append(msg)
            elif "industry" in msg.lower() or "sponsored" in msg.lower():
                categories["Funding"].append(msg)
            elif "age" in msg.lower() or "female" in msg.lower():
                categories["Generalizability"].append(msg)
            elif "ITT" in msg or "analytic approach" in msg.lower():
                categories["Statistical Analysis"].append(msg)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def generate_summary_report(self, abstract: str, messages: List[str]) -> str:
        """
        Generate a comprehensive minor issues report.
        
        Args:
            abstract: Original abstract text
            messages: Detected issue messages
            
        Returns:
            Formatted report string
        """
        categorized = self.categorize_issues(messages)
        
        report = f"""
📋 === RCT MINOR METHODOLOGICAL ISSUES REPORT ===

ABSTRACT EXCERPT:
{'-' * 60}
{abstract[:250]}{'...' if len(abstract) > 250 else ''}

SUMMARY:
{'-' * 60}
• Total Minor Issues: {len(messages)}
• Categories Affected: {len(categorized)}
• Issue Types: {', '.join(categorized.keys()) if categorized else 'None'}

DETAILED FINDINGS:
{'-' * 60}
"""
        
        if not messages:
            report += "✅ No minor methodological issues detected.\n"
        elif len(messages) == 1 and messages[0].startswith("Error:"):
            report += f"❌ {messages[0]}\n"
        else:
            issue_counter = 1
            for category, issues in categorized.items():
                report += f"\n📌 {category.upper()}:\n"
                for issue in issues:
                    report += f"   {issue_counter}. {issue}\n\n"
                    issue_counter += 1
        
        report += f"\n{'='*60}\n"
        report += f"Analysis completed - {len(self.minor_issues)} heuristics checked\n"
        
        return report
    
    def get_issue_statistics(self, messages: List[str]) -> Dict[str, any]:
        """
        Get statistical summary of detected issues.
        
        Args:
            messages: List of issue messages
            
        Returns:
            Dictionary with statistics
        """
        if not messages or (len(messages) == 1 and messages[0].startswith("Error:")):
            return {
                "total_issues": 0,
                "has_errors": len(messages) > 0 and messages[0].startswith("Error:"),
                "categories": {},
                "most_common_category": None
            }
        
        categorized = self.categorize_issues(messages)
        category_counts = {k: len(v) for k, v in categorized.items()}
        most_common = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
        
        return {
            "total_issues": len(messages),
            "has_errors": False,
            "categories": category_counts,
            "most_common_category": most_common
        }

# Factory function
def create_minor_issues_detector():
    """Factory function to create minor issues detector."""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        return RCTMinorIssuesDetector(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize minor issues detector: {e}")
        return None

# Example usage and testing
def main():
    detector = create_minor_issues_detector()
    if not detector:
        print("❌ Failed to initialize minor issues detector")
        return
    
    # Test abstract with several minor issues
    test_abstract = """
Background: We evaluated the efficacy of a new cardiac medication in patients with heart failure.

Methods: This randomized trial enrolled 150 patients (mean age 65 years, 40% female) with heart failure. Patients received either the new medication or standard therapy. The primary outcome was improvement in ejection fraction. The study was sponsored by CardioPharm Inc.

Results: At 3 weeks, the treatment group showed significant improvement in ejection fraction compared to control (p=0.02).

Conclusions: The new medication appears effective for heart failure treatment in the short term.
"""
    
    print("🔍 === RCT MINOR METHODOLOGICAL ISSUES DETECTOR ===\n")
    
    # Detect minor issues
    issues = detector.detect_minor_issues(test_abstract)
    
    # Generate comprehensive report
    report = detector.generate_summary_report(test_abstract, issues)
    print(report)
    
    # Show statistics
    stats = detector.get_issue_statistics(issues)
    print("📊 ISSUE STATISTICS:")
    print("-" * 30)
    print(f"Total Issues: {stats['total_issues']}")
    print(f"Most Common Category: {stats['most_common_category']}")
    print(f"Category Breakdown: {stats['categories']}")
    
    # Test with a higher quality abstract
    quality_abstract = """
Background: We investigated the long-term effects of antihypertensive therapy.

Methods: This double-blind, placebo-controlled, multicenter trial included 2,500 patients (mean age 72 years, 52% female) followed for 3 years. The primary outcome was cardiovascular events assessed at 36 months. Analysis was performed using intention-to-treat principles. This study was funded by the National Heart Institute.

Results: Significant reduction in cardiovascular events was observed (HR 0.75, 95% CI 0.65-0.87, p<0.001).

Conclusions: Long-term antihypertensive therapy reduces cardiovascular risk.
"""
    
    print("\n" + "="*70)
    print("🧪 TESTING WITH HIGHER QUALITY ABSTRACT:")
    print("="*70)
    
    quality_issues = detector.detect_minor_issues(quality_abstract)
    quality_report = detector.generate_summary_report(quality_abstract, quality_issues)
    print(quality_report)

if __name__ == "__main__":
    main()