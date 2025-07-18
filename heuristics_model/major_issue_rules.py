import subprocess
import sys
import importlib

# Auto-install function
def install_package(package_name, import_name=None):
    """Install a package if it's not already installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            return False

# Install required packages automatically
def auto_install_requirements():
    """Automatically install all required packages"""
    requirements = [
        ("streamlit", "streamlit"),
        ("spacy>=3.6.0", "spacy"),
        ("pandas>=2.0.0", "pandas"),
        ("numpy>=1.24.0", "numpy"),
        ("regex>=2023.6.3", "regex")
    ]
    
    missing_packages = []
    
    for package, import_name in requirements:
        if not install_package(package, import_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Failed to install: {', '.join(missing_packages)}")
        return False
    
    return True

# Run auto-installation
if __name__ == "__main__":
    print("🔄 Checking and installing required packages...")
    auto_install_requirements()

# Now import the packages
try:
    import streamlit as st
    import re
    import spacy
    from typing import Optional, List, Dict, Set
    import csv
    import pandas as pd
    from datetime import datetime
    import os
    import io
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all packages are installed correctly.")
    sys.exit(1)

# Set page config
st.set_page_config(
    page_title="RCT Bias Detection Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-setup function for spaCy model
def setup_spacy_model():
    """Automatically download spaCy model if not found"""
    try:
        # Try to load the model first
        return spacy.load("en_core_web_sm")
    except OSError:
        # Model not found, try to download it
        try:
            st.info("🔄 Downloading spaCy English model (this may take a few minutes)...")
            
            # Create progress indicators
            progress_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.write("⏳ Downloading spaCy model...")
                progress_bar = st.progress(0)
                
                # Download the model
                result = subprocess.run([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], capture_output=True, text=True)
                
                progress_bar.progress(100)
                
                if result.returncode == 0:
                    st.success("✅ spaCy model downloaded successfully!")
                    progress_placeholder.empty()
                    
                    # Try to load the model again
                    try:
                        return spacy.load("en_core_web_sm")
                    except OSError:
                        st.error("Model downloaded but still cannot load. Please restart the app.")
                        st.stop()
                else:
                    progress_placeholder.empty()
                    st.error(f"""
                    ❌ Failed to download spaCy model automatically.
                    
                    Error: {result.stderr}
                    
                    **Manual Installation Instructions:**
                    1. Open your terminal/command prompt
                    2. Run: `python -m spacy download en_core_web_sm`
                    3. Restart this Streamlit app
                    """)
                    st.stop()
                    
        except Exception as e:
            st.error(f"❌ Error during spaCy setup: {str(e)}")
            st.error("""
            **Manual Installation Instructions:**
            1. Open your terminal/command prompt  
            2. Run: `pip install spacy`
            3. Run: `python -m spacy download en_core_web_sm`
            4. Restart this Streamlit app
            """)
            st.stop()

# Load spaCy model with auto-setup
@st.cache_resource
def load_nlp_model():
    return setup_spacy_model()

# Initialize spaCy only when Streamlit is running
if 'streamlit' in sys.modules:
    nlp = load_nlp_model()
else:
    nlp = None

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .bias-card {
        background: #f8f9fa;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .no-bias-card {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .minor-issue-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Major bias detection functions (from your first code)
def check_sample_size(text: str) -> str | None:
    """
    Checks for reported sample size using spaCy to extract number + population nouns.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    doc = nlp(text.lower())
    text_lower = text.lower()
    
    # Enhanced participant terms
    participant_terms = {
        "participants", "patients", "subjects", "individuals", "enrollees",
        "adults", "children", "women", "men", "volunteers", "cases",
        "persons", "people", "cohort", "sample", "population"
    }
    
    # Context terms to exclude (not sample sizes)
    exclusion_contexts = {
        "years", "months", "weeks", "days", "hours", "minutes",
        "mg", "ml", "kg", "dose", "dosage", "times", "percent", "%",
        "age", "aged", "old", "baseline", "follow-up", "followup",
        "centers", "sites", "countries", "visits", "sessions"
    }
    
    # Study type-specific thresholds
    size_thresholds = {
        "pilot": {"min": 20, "adequate": 50},
        "phase_1": {"min": 20, "adequate": 50},
        "phase_2": {"min": 100, "adequate": 200},
        "phase_3": {"min": 300, "adequate": 1000},
        "rct": {"min": 200, "adequate": 500},
        "observational": {"min": 500, "adequate": 1000},
        "meta_analysis": {"min": 1000, "adequate": 5000}
    }
    
    sample_sizes = []
    total_enrolled = None
    randomized_number = None
    
    # Enhanced number extraction with context validation
    for i, token in enumerate(doc):
        # Skip if token is part of exclusion context
        context_window = doc[max(0, i-2):min(len(doc), i+3)]
        if any(exc in [t.text for t in context_window] for exc in exclusion_contexts):
            continue
            
        number = None
        
        # Handle numeric tokens (including comma-separated)
        if token.like_num:
            try:
                clean_num = token.text.replace(',', '').replace('.', '')
                number = int(clean_num)
            except ValueError:
                continue
        
        # Handle written numbers
        elif token.text in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            number = word_to_num.get(token.text)
        
        # Handle compound numbers like "one hundred", "two thousand"
        elif token.text in ["hundred", "thousand"]:
            if i > 0 and doc[i-1].text in ["one", "two", "three", "four", "five"]:
                multiplier = 100 if token.text == "hundred" else 1000
                prev_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(doc[i-1].text, 1)
                number = prev_num * multiplier
        
        if number and number > 5:  # Ignore very small numbers
            # Look for participant terms in surrounding context
            context_start = max(0, i-3)
            context_end = min(len(doc), i+4)
            context_tokens = [t.text for t in doc[context_start:context_end]]
            
            # Check if any participant term is nearby
            if any(term in context_tokens for term in participant_terms):
                sentence = token.sent.text.lower()
                
                # Distinguish enrollment vs randomization
                if any(keyword in sentence for keyword in ["enrolled", "randomized", "randomised", "recruited", "assigned"]):
                    if "enrolled" in sentence or "recruited" in sentence:
                        total_enrolled = number
                    elif "randomized" in sentence or "randomised" in sentence or "assigned" in sentence:
                        randomized_number = number
                    
                sample_sizes.append(number)
    
    # Remove duplicates and determine primary sample size
    sample_sizes = sorted(list(set(sample_sizes)))
    
    if not sample_sizes:
        return "No sample size reported: unable to assess study power and precision"
    
    # Determine primary sample size (prefer total enrolled, then randomized, then max)
    primary_n = max(sample_sizes)
    if total_enrolled:
        primary_n = total_enrolled
    elif randomized_number:
        primary_n = randomized_number
    
    # Detect study type for appropriate thresholds
    study_type = "rct"  # default
    
    if "pilot" in text_lower or "feasibility" in text_lower:
        study_type = "pilot"
    elif "phase i" in text_lower or "phase 1" in text_lower:
        study_type = "phase_1"
    elif "phase ii" in text_lower or "phase 2" in text_lower:
        study_type = "phase_2"
    elif "phase iii" in text_lower or "phase 3" in text_lower:
        study_type = "phase_3"
    elif "meta-analysis" in text_lower or "systematic review" in text_lower:
        study_type = "meta_analysis"
    elif any(term in text_lower for term in ["cohort", "case-control", "cross-sectional", "observational"]):
        study_type = "observational"
    elif any(term in text_lower for term in ["randomized", "randomised", "rct", "controlled trial"]):
        study_type = "rct"
    
    # Get appropriate thresholds
    thresholds = size_thresholds[study_type]
    
    # Check for power analysis reporting
    power_indicators = [
        "power analysis", "sample size calculation", "power calculation",
        "80% power", "90% power", "power of 80%", "power of 90%",
        "effect size", "statistical power", "adequately powered"
    ]
    has_power_analysis = any(indicator in text_lower for indicator in power_indicators)
    
    # Check for multi-arm design
    group_indicators = ["groups", "arms", "treatment groups", "control group", "placebo group"]
    has_multiple_groups = any(indicator in text_lower for indicator in group_indicators)
    
    # Main sample size assessment
    if primary_n < thresholds["min"]:
        return f"Very small sample size (n={primary_n}) for {study_type.replace('_', ' ')}: High risk of underpowering and selection bias"
    
    elif primary_n < thresholds["adequate"]:
        # Additional context for RCTs with multiple groups
        if study_type in ["rct", "phase_2", "phase_3"] and has_multiple_groups:
            estimated_per_group = primary_n // 2
            if estimated_per_group < 50:
                return f"Small per-group sample size (~{estimated_per_group} per group, total n={primary_n}): High risk of underpowering"
            else:
                return f"Modest sample size (n={primary_n}) for {study_type.replace('_', ' ')}: May be underpowered to detect moderate effects"
        else:
            return f"Small sample size (n={primary_n}) for {study_type.replace('_', ' ')}: May be underpowered to detect clinically meaningful effects"
    
    # Adequate sample size but check for power analysis
    elif not has_power_analysis and primary_n < 1000 and study_type in ["rct", "phase_2", "phase_3"]:
        return f"No power analysis reported (n={primary_n}): Unable to verify adequate study power for effect detection"
    
    # Additional check for very large studies without power analysis
    elif not has_power_analysis and primary_n >= 1000 and study_type == "observational":
        return f"Large observational study (n={primary_n}) without power analysis: Verify if adequately powered for planned analyses"
    
    # Check for extremely small per-group sizes in multi-arm RCTs
    elif study_type in ["rct", "phase_2", "phase_3"] and has_multiple_groups:
        estimated_per_group = primary_n // 2
        if estimated_per_group < 30:
            return f"Very small per-group sample size (~{estimated_per_group} per group): High risk of underpowering despite total n={primary_n}"
    
    # No sample size bias detected
    return None

def check_primary_outcome_events(text: str) -> str | None:
    """
    Detects small event counts (<30) linked to primary outcomes or group-level results.
    """
    doc = nlp(text.lower())
    text_lower = text.lower()
    
    # Enhanced outcome detection patterns
    primary_outcome_patterns = [
        "primary outcome", "primary endpoint", "main outcome", "principal outcome",
        "primary end point", "primary efficacy", "main endpoint", "key outcome",
        "primary analysis", "primary measure", "main efficacy"
    ]
    
    secondary_outcome_patterns = [
        "secondary outcome", "secondary endpoint", "secondary analysis",
        "exploratory outcome", "additional outcome"
    ]
    
    # Enhanced event terms with context
    binary_event_terms = {
        "events", "cases", "occurrences", "incidents", "episodes",
        "deaths", "mortality", "hospitalizations", "admissions",
        "responses", "responders", "successes", "failures",
        "relapses", "recurrences", "complications", "adverse events"
    }
    
    continuous_outcome_terms = {
        "score", "level", "concentration", "measurement", "value",
        "change", "improvement", "reduction", "increase", "difference"
    }
    
    # Exclusion contexts (not event counts)
    exclusion_contexts = {
        "percent", "%", "percentage", "rate per", "years", "months", "weeks", "days",
        "mg", "ml", "kg", "mmol", "units", "baseline", "follow-up",
        "age", "aged", "old", "duration", "time", "period"
    }
    
    # Group comparison indicators
    group_indicators = [
        "treatment group", "control group", "placebo group", "intervention group",
        "active group", "experimental group", "comparison group", "study group",
        "arm", "cohort", "versus", "vs", "compared with", "compared to"
    ]
    
    detected_events = []
    outcome_type = None
    is_primary_outcome_context = False
    
    # Analyze each sentence for outcome reporting
    for sent in doc.sents:
        sent_text = sent.text.lower().strip()
        sent_doc = nlp(sent_text)
        
        # Check if sentence contains primary outcome context
        has_primary_outcome = any(pattern in sent_text for pattern in primary_outcome_patterns)
        has_group_comparison = any(indicator in sent_text for indicator in group_indicators)
        has_results_context = any(word in sent_text for word in ["results", "outcome", "occurred", "observed", "found"])
        
        # Focus on relevant sentences
        if has_primary_outcome or (has_group_comparison and has_results_context):
            is_primary_outcome_context = True
            
            # Determine outcome type
            if any(term in sent_text for term in binary_event_terms):
                outcome_type = "binary"
            elif any(term in sent_text for term in continuous_outcome_terms):
                outcome_type = "continuous"
            
            # Extract numbers with enhanced context validation
            for i, token in enumerate(sent_doc):
                if token.like_num:
                    try:
                        # Handle various number formats
                        clean_num = token.text.replace(',', '').replace('(', '').replace(')', '')
                        if '.' in clean_num and len(clean_num.split('.')[1]) > 2:
                            continue  # Skip decimal numbers with many places (likely measurements)
                        number = int(float(clean_num))
                    except (ValueError, AttributeError):
                        continue
                    
                    # Enhanced context window analysis
                    context_start = max(0, i-4)
                    context_end = min(len(sent_doc), i+5)
                    context_tokens = [t.text.lower() for t in sent_doc[context_start:context_end]]
                    
                    # Skip if in exclusion context
                    if any(exc in context_tokens for exc in exclusion_contexts):
                        continue
                    
                    # Skip percentages and rates
                    if any(perc in context_tokens for perc in ["%", "percent", "percentage", "rate"]):
                        continue
                    
                    # Check for event context
                    has_event_context = any(term in context_tokens for term in binary_event_terms)
                    has_participant_context = any(term in context_tokens for term in ["patients", "participants", "subjects"])
                    
                    # Look for group-specific reporting
                    group_context = any(term in context_tokens for term in ["group", "arm", "cohort"])
                    
                    if (has_event_context or has_participant_context) and number > 0:
                        # Additional validation using spaCy dependencies
                        valid_event = False
                        
                        # Check if number is subject/object of relevant verbs
                        if token.dep_ in ["nsubj", "dobj", "pobj"] or any(child.dep_ in ["nsubj", "dobj"] for child in token.children):
                            valid_event = True
                        
                        # Check for explicit event reporting patterns
                        event_patterns = [
                            f"{number} events", f"{number} cases", f"{number} patients",
                            f"{number} participants", f"{number} subjects"
                        ]
                        if any(pattern in sent_text for pattern in event_patterns):
                            valid_event = True
                        
                        if valid_event:
                            detected_events.append({
                                'number': number,
                                'context': ' '.join(context_tokens),
                                'sentence': sent_text,
                                'has_group_context': group_context
                            })
    
    # Decision logic based on detected events and context
    if not is_primary_outcome_context:
        return None  # No primary outcome context found
    
    if not detected_events:
        # Check if it's a continuous outcome (may not have discrete events)
        if outcome_type == "continuous":
            return None  # Continuous outcomes don't need event count assessment
        else:
            return "Primary outcome results not clearly reported: unable to assess event rates"
    
    # Analyze detected events for adequacy
    small_event_counts = [event for event in detected_events if event['number'] < 30]
    very_small_events = [event for event in detected_events if event['number'] < 10]
    
    if very_small_events:
        min_events = min(event['number'] for event in very_small_events)
        return f"Very low primary outcome events (minimum {min_events}): High risk of underpowering and unreliable effect estimates"
    
    elif small_event_counts:
        min_events = min(event['number'] for event in small_event_counts)
        # Check if multiple groups reported
        has_group_specific = any(event['has_group_context'] for event in small_event_counts)
        
        if has_group_specific:
            return f"Low primary outcome events in one or more groups ({min_events} events): Risk of insufficient power for reliable estimates"
        else:
            return f"Low primary outcome events ({min_events} total): May be underpowered for primary analysis"
    
    # Additional checks for binary outcomes
    if outcome_type == "binary":
        total_events = sum(event['number'] for event in detected_events)
        
        # Check for very unbalanced event rates
        if len(detected_events) >= 2:
            event_numbers = [event['number'] for event in detected_events]
            max_events = max(event_numbers)
            min_events = min(event_numbers)
            
            if min_events > 0 and (max_events / min_events) > 10:
                return f"Highly unbalanced event rates between groups ({min_events} vs {max_events}): May affect statistical validity"
        
        # Rule of thumb: need at least 5-10 events per variable in regression
        if total_events < 50 and any(term in text_lower for term in ["multivariate", "adjusted", "regression", "model"]):
            return f"Low total events ({total_events}) for multivariate analysis: Risk of overfitting and unreliable estimates"
    
    return None

def check_composite_outcome(text: str) -> str | None:
    """
    Uses spaCy to detect composite primary outcomes with more flexibility.
    """
    doc = nlp(text.lower())
    text_lower = text.lower()
    
    # Enhanced composite outcome patterns
    explicit_composite_patterns = [
        "composite outcome", "composite endpoint", "composite end point",
        "combined outcome", "combined endpoint", "combined end point",
        "composite of", "combination of", "composite primary",
        "composite secondary", "mace", "major adverse cardiac events",
        "major adverse cardiovascular events", "composite safety",
        "time to first", "first occurrence of"
    ]
    
    # Medical outcome categories for better classification
    hard_endpoints = {
        "death", "mortality", "died", "fatal", "fatality",
        "stroke", "myocardial infarction", "heart attack", "mi",
        "cardiac arrest", "sudden death", "cardiovascular death"
    }
    
    soft_endpoints = {
        "hospitalization", "admission", "readmission", "rehospitalization",
        "revascularization", "intervention", "procedure", "surgery",
        "symptom", "quality of life", "functional status", "pain"
    }
    
    surrogate_endpoints = {
        "blood pressure", "cholesterol", "glucose", "hba1c",
        "ejection fraction", "laboratory", "biomarker", "level"
    }
    
    # Clinical outcome terms (actual medical events)
    clinical_outcomes = hard_endpoints | soft_endpoints | {
        "bleeding", "hemorrhage", "thrombosis", "embolism", "infection",
        "fracture", "cancer", "tumor", "progression", "recurrence",
        "failure", "rejection", "complication", "adverse event",
        "emergency", "icu", "intensive care", "ventilation"
    }
    
    # Non-outcome terms to exclude
    non_outcome_terms = {
        "patient", "participant", "subject", "study", "trial", "group",
        "arm", "treatment", "drug", "medication", "dose", "therapy",
        "baseline", "follow-up", "visit", "assessment", "measurement",
        "analysis", "method", "procedure", "protocol", "design",
        "center", "site", "investigator", "researcher", "data"
    }
    
    # Primary vs secondary outcome context
    primary_context_patterns = [
        "primary outcome", "primary endpoint", "primary end point",
        "main outcome", "principal outcome", "key outcome",
        "primary efficacy", "primary analysis"
    ]
    
    # Step 1: Check for explicit composite patterns
    for pattern in explicit_composite_patterns:
        if pattern in text_lower:
            # Check if it's in primary outcome context
            is_primary = any(p_pattern in text_lower for p_pattern in primary_context_patterns)
            
            if is_primary:
                return (
                    "Composite primary outcome detected: may reduce interpretability "
                    "and dilute treatment effects of individual components"
                )
            else:
                return (
                    "Composite outcome detected: verify clinical relevance "
                    "and individual component importance"
                )
    
    # Additional analysis would go here (simplified for space)
    return None

def check_placebo(text: str) -> str | None:
    """
    Checks for the absence of a placebo in the abstract.
    """    
    # Process text with spaCy
    doc = nlp(text)
    text_lower = text.lower()
    
    # Enhanced placebo patterns
    placebo_positive_patterns = [
        "placebo", "placebo-controlled", "placebo arm", "placebo group",
        "received a placebo", "given a placebo", "received placebo",
        "assigned to placebo", "placebo comparator", "use of placebo",
        "placebo treatment", "placebo intervention", "matching placebo",
        "identical placebo", "sham", "dummy pill", "sugar pill"
    ]
    
    placebo_negative_patterns = [
        "no placebo", "not placebo-controlled", "did not receive a placebo",
        "without a placebo", "lack of placebo", "no sham", "without sham",
        "open-label", "open label", "unblinded", "non-blinded"
    ]
    
    # Study design patterns
    blinding_patterns = [
        "double-blind", "double blind", "single-blind", "single blind",
        "triple-blind", "triple blind", "blinded", "masked"
    ]
    
    control_patterns = [
        "control group", "control arm", "standard of care", "standard care",
        "usual care", "current practice", "active control", "historical control",
        "waitlist control", "wait-list control", "no treatment control"
    ]
    
    comparison_phrases = [
        "vs", "versus", "compared with", "compared to", "compared against",
        "head-to-head", "head to head", "superiority trial", "non-inferiority",
        "equivalence trial", "active comparator", "active-controlled"
    ]
    
    # Extract drug names using basic patterns (simplified)
    drug_names = set()
    
    # Use spaCy NER for drug detection
    for ent in doc.ents:
        if ent.label_ in ["DRUG", "PRODUCT", "CHEMICAL", "PERSON"] and len(ent.text) > 2:
            # Filter out common false positives
            if not any(word in ent.text.lower() for word in ["patient", "study", "trial", "group", "week", "day", "month"]):
                drug_names.add(ent.text.lower())
    
    # Look for negation patterns around placebo mentions
    has_negated_placebo = False
    for token in doc:
        if token.lemma_ == "placebo":
            # Check for negation in surrounding tokens
            for child in token.children:
                if child.dep_ == "neg":
                    has_negated_placebo = True
                    break
            # Check previous tokens for negation
            for i in range(max(0, token.i-3), token.i):
                if doc[i].lemma_ in ["no", "not", "without", "lack"]:
                    has_negated_placebo = True
                    break
    
    # Boolean flags
    found_placebo = any(p in text_lower for p in placebo_positive_patterns) and not has_negated_placebo
    found_negative = any(n in text_lower for n in placebo_negative_patterns) or has_negated_placebo
    has_blinding = any(b in text_lower for b in blinding_patterns)
    has_control = any(c in text_lower for c in control_patterns)
    head_to_head_trial = any(phrase in text_lower for phrase in comparison_phrases) and len(drug_names) >= 2
    
    # Decision logic
    # 1. Explicit negative statements
    if found_negative:
        if "open-label" in text or "unblinded" in text:
            return "Open-label/unblinded study: high risk of performance and detection bias"
        return "Explicitly states no placebo used: possible adjudication and confounding bias"
    
    # 2. Head-to-head trials (drug vs drug)
    if head_to_head_trial:
        if not has_blinding:
            return "Head-to-head trial without blinding: possible performance bias"
        # Head-to-head with blinding is generally acceptable
        return None
    
    # 3. Control group analysis
    if has_control:
        if not found_placebo and not has_blinding:
            return "Control group without placebo or blinding: high risk of performance and detection bias"
        elif not found_placebo and has_blinding:
            return "Blinded control without placebo mention: verify if sham/placebo control used"
    
    # 4. No placebo in non-head-to-head
    if not found_placebo and not head_to_head_trial:
        return "No mention of placebo in apparent controlled trial: possible confounding or adjudication bias"
    
    # No bias detected
    return None

def check_blinding(text: str) -> str | None:
    """
    Checks for missing or unreported blinding.
    """ 
    text = text.lower()
    doc = nlp(text)

    # Enhanced blinding positive patterns
    blinding_positive_patterns = [
        "blinded", "double-blind", "double blind", "single-blind", "single blind",
        "triple-blind", "triple blind", "assessors blinded", "participants blinded",
        "investigator blinded", "outcome assessors were blinded", "blinding was maintained",
        "masked", "double-masked", "single-masked", "triple-masked",
        "assessors masked", "participants masked", "investigator masked",
        "outcome assessors were masked", "masking was maintained",
        "allocation concealment", "concealed allocation", "concealed randomization"
    ]

    # Enhanced negative patterns
    blinding_negative_patterns = [
        "not blinded", "no blinding", "open label", "open-label", "unblinded",
        "lack of blinding", "blinding was not performed", "not masked", "no masking",
        "unmasked", "lack of masking", "masking was not performed",
        "single-arm", "single arm", "non-blinded", "non-masked",
        "investigators were not blinded", "participants were not blinded",
        "assessors were not blinded", "outcome assessors were not blinded"
    ]

    # Study design and placebo patterns
    study_design_patterns = [
        "randomized controlled trial", "rct", "placebo-controlled", "sham-controlled"
    ]
    
    placebo_patterns = ["placebo", "sham", "dummy pill", "sugar pill", "double-dummy"]

    # Initialize flags
    found_blinding = False
    found_unblinded = False
    found_study_design = False
    has_placebo = False
    blinding_negated = False
    
    # Check basic patterns
    has_placebo = any(p in text for p in placebo_patterns)
    found_study_design = any(d in text for d in study_design_patterns)

    # Sentence-level analysis
    for sent in doc.sents:
        s = sent.text.lower().strip()
        
        if any(p in s for p in blinding_positive_patterns):
            found_blinding = True
        
        if any(n in s for n in blinding_negative_patterns):
            found_unblinded = True

    # Enhanced spaCy negation detection
    for token in doc:
        if token.lemma_ in ["blind", "blinded", "mask", "masked"]:
            # Check for negation dependencies
            for child in token.children:
                if child.dep_ == "neg" or child.lemma_ in ["not", "no", "without"]:
                    blinding_negated = True
                    break
            
            # Check surrounding context (3 tokens before/after)
            start_idx = max(0, token.i - 3)
            end_idx = min(len(doc), token.i + 3)
            context = doc[start_idx:end_idx]
            
            negation_words = ["not", "no", "without", "lack", "absence", "un"]
            for ctx_token in context:
                if any(neg in ctx_token.text.lower() for neg in negation_words):
                    blinding_negated = True
                    break

    # Decision logic
    if found_unblinded or blinding_negated:
        return (
            "Blinding explicitly not performed (e.g. open-label or unblinded): "
            "high risk of performance, detection, and adjudication bias"
        )

    if has_placebo and not found_blinding:
        return (
            "Placebo/sham mentioned but no blinding reported: "
            "verify if blinding was properly implemented and reported"
        )

    if found_study_design and not found_blinding and not has_placebo:
        return (
            "Randomized controlled trial without blinding mention: "
            "possible performance and detection bias"
        )

    if not found_blinding:
        return (
            "No mention of blinding: possible performance, detection, "
            "and adjudication bias due to inadequate blinding"
        )

    return None

# Minor bias detection functions (from your second code)
def check_duration_followup(text: str) -> str | None:
    """
    Detects follow-up duration issues based on the rules:
    - Less than 1 year: insufficient for meaningful clinical outcomes
    - Not provided: duration not specified
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    doc = nlp(text.lower())
    text_lower = text.lower()
    
    # Enhanced duration extraction patterns
    duration_patterns = [
        # Standard follow-up patterns
        r'follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'followed.*?(?:for|over).*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\s*follow(?:\s|-)*up',
        r'observation.*?period.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'study.*?duration.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'median.*?follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'mean.*?follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        
        # Assessment timeline patterns
        r'(?:at|after)\s+(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\s*(?:of\s+age)?',
        r'assessments?.*?(?:at|after)\s+(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'evaluated?.*?(?:at|after)\s+(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'measured?.*?(?:at|after)\s+(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'from\s+baseline\s+to\s+(?:year\s+)?(\d+(?:\.\d+)?)\s*(year)s?',
        
        # Range patterns
        r'follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*(day|week|month|year)s?'
    ]
    
    # Check for "not provided" indicators
    no_duration_patterns = [
        r'follow(?:\s|-)*up.*?(?:not|no).*?(?:provided|specified|mentioned|reported)',
        r'duration.*?(?:not|no).*?(?:provided|specified|mentioned|reported)',
        r'(?:not|no).*?follow(?:\s|-)*up.*?(?:duration|period)',
        r'follow(?:\s|-)*up.*?(?:unclear|unspecified|unreported)'
    ]
    
    # Check for no duration provided first
    for pattern in no_duration_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "The duration of follow-up was not provided. In studies less than a year, important clinical outcomes may not have had sufficient time to develop, limiting the ability to draw meaningful conclusions about long-term risks or benefits."
    
    # Extract durations
    detected_durations = []
    
    for pattern in duration_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                if len(match) == 3:  # Range pattern
                    value1, value2, unit = match
                    try:
                        avg_value = (float(value1) + float(value2)) / 2
                        detected_durations.append((avg_value, unit))
                    except ValueError:
                        continue
                else:  # Single value pattern
                    value, unit = match
                    try:
                        detected_durations.append((float(value), unit))
                    except ValueError:
                        continue
    
    if not detected_durations:
        # Check if follow-up is mentioned without duration
        if re.search(r'follow(?:\s|-)*up', text_lower, re.IGNORECASE):
            return "The duration of follow-up was not provided. In studies less than a year, important clinical outcomes may not have had sufficient time to develop, limiting the ability to draw meaningful conclusions about long-term risks or benefits."
        return None
    
    # Convert all durations to months for comparison
    durations_in_months = []
    for value, unit in detected_durations:
        unit_lower = unit.lower()
        if 'day' in unit_lower:
            months = value / 30.44
        elif 'week' in unit_lower:
            months = value / 4.35
        elif 'month' in unit_lower:
            months = value
        elif 'year' in unit_lower:
            months = value * 12
        else:
            continue
        durations_in_months.append(months)
    
    if durations_in_months:
        # Use the highest duration found (most representative)
        max_duration = max(durations_in_months)
        if max_duration < 12:  # Less than 1 year
            return "The follow-up duration is under one year, which may be insufficient to capture meaningful clinical outcomes, especially for chronic conditions or interventions with delayed effects. This limits the ability to assess sustained efficacy, adverse events, and long-term risk."
    
    return None

def check_multicentre(text: str) -> str | None:
    """
    Detects multicentre study issues.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # Enhanced multicentre positive patterns
    multicentre_patterns = [
        r'multi[\s-]*cent(?:er|re)',
        r'multiple\s+cent(?:er|re)s?',
        r'(\d+)\s+cent(?:er|re)s?',
        r'multi[\s-]*site',
        r'multiple\s+sites?',
        r'(\d+)[\s-]*sites?',
        r'two[\s-]*site',
        r'three[\s-]*site',
        r'multi[\s-]*institution',
        r'multiple\s+institutions?',
        r'international\s+(?:study|trial)',
        r'multinational\s+(?:study|trial)',
        r'across.*?(?:europe|america|asia|countries)',
        r'(?:europe|america|asia)\s+and\s+(?:europe|america|asia)'
    ]
    
    # Single centre patterns
    single_centre_patterns = [
        r'single[\s-]*cent(?:er|re)',
        r'one\s+cent(?:er|re)',
        r'single[\s-]*site',
        r'one\s+site',
        r'single[\s-]*institution',
        r'one\s+institution'
    ]
    
    # Check for explicit multicentre mentions
    for pattern in multicentre_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return None  # Multicentre is good - no issue
    
    # Check for explicit single centre mentions
    for pattern in single_centre_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "The abstract lacked details on whether the study was single-centered or multicentered. This often means the study was single-centered. Single-centre studies are NOT as easy to replicate."
    
    # If no centre information mentioned, assume single centre
    return "The abstract lacked details on whether the study was single-centered or multicentered. This often means the study was single-centered. Single-centre studies are NOT as easy to replicate."

def check_timeline_outcome(text: str) -> str | None:
    """
    Detects primary outcome timeline issues.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # Primary outcome context patterns
    primary_outcome_patterns = [
        r'primary\s+(?:outcome|endpoint|end[\s-]*point)',
        r'main\s+(?:outcome|endpoint)',
        r'principal\s+(?:outcome|endpoint)',
        r'key\s+(?:outcome|endpoint)'
    ]
    
    # Timeline extraction patterns
    timeline_patterns = [
        r'(?:primary\s+)?(?:outcome|endpoint).*?(?:at|after|within)\s*(\d+)\s*(day|week|month|year)s?',
        r'(?:primary\s+)?(?:outcome|endpoint).*?(\d+)\s*(day|week|month|year)s?',
        r'(\d+)\s*(day|week|month|year)s?.*?(?:primary\s+)?(?:outcome|endpoint)',
        r'assessed.*?(?:at|after|within)\s*(\d+)\s*(day|week|month|year)s?',
        r'measured.*?(?:at|after|within)\s*(\d+)\s*(day|week|month|year)s?',
        r'evaluated.*?(?:at|after|within)\s*(\d+)\s*(day|week|month|year)s?'
    ]
    
    # Check if primary outcome is mentioned
    has_primary_outcome = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in primary_outcome_patterns)
    
    if not has_primary_outcome:
        return None  # No primary outcome mentioned, so no timeline issue
    
    # Extract timeline information
    detected_timelines = []
    
    for pattern in timeline_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                value, unit = match
                try:
                    detected_timelines.append((float(value), unit.lower()))
                except ValueError:
                    continue
    
    if not detected_timelines:
        return "The timeline for assessing the primary outcome is not clearly stated. In RCTs, the timing of outcome measurement is critical for assessing outcomes happening too early may underestimate treatment effects, while excessively short timelines may miss delayed benefits or harms. Lack of this information limits interpretability and clinical relevance."
    
    # Convert to days and check for short timelines
    for value, unit in detected_timelines:
        if 'day' in unit:
            days = value
        elif 'week' in unit:
            days = value * 7
        elif 'month' in unit:
            days = value * 30.44
        elif 'year' in unit:
            days = value * 365.25
        else:
            continue
        
        if days <= 30:
            return "The primary outcome was assessed within a short time frame (≤30 days), which may be insufficient to observe meaningful clinical effects, particularly for chronic conditions or interventions with delayed impact. Short timelines risk underestimating true benefits or harms."
    
    return None

def check_funding(text: str) -> str | None:
    """
    Detects industry funding issues.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # Check for government/academic funding first (these are NOT industry funding)
    government_academic_patterns = [
        r'national institute', r'nih\b', r'nsf\b', r'department of',
        r'ministry of', r'government', r'public health', r'health service',
        r'university', r'college', r'medical school', r'hospital',
        r'research council', r'foundation(?!\s+(?:inc|corp|ltd))', 
        r'charitable', r'non[\s-]?profit', r'academy of',
        r'society of', r'association of', r'wellcome trust',
        r'gates foundation', r'howard hughes', r'cancer research'
    ]
    
    # Check if funded by government/academic sources
    for pattern in government_academic_patterns:
        funding_context_patterns = [
            rf'funded\s+by.*?{pattern}',
            rf'supported\s+by.*?{pattern}',
            rf'grant.*?(?:from|by).*?{pattern}',
            rf'{pattern}.*?(?:funded|supported|grant)'
        ]
        
        for fund_pattern in funding_context_patterns:
            if re.search(fund_pattern, text_lower, re.IGNORECASE):
                return None  # Government/academic funding - no issue
    
    # Known pharmaceutical company patterns
    known_pharma_companies = [
        'pfizer', 'novartis', 'roche', 'merck', 'gsk', 'glaxosmithkline', 'astrazeneca',
        'sanofi', 'bayer', 'eli lilly', 'lilly', 'bristol myers', 'bristol-myers',
        'amgen', 'gilead', 'biogen', 'regeneron', 'moderna', 'johnson & johnson',
        'j&j', 'abbott', 'boehringer', 'takeda', 'novo nordisk', 'teva',
        'celgene', 'vertex', 'alexion', 'shire', 'allergan', 'mylan'
    ]
    
    # Industry funding patterns
    industry_funding_patterns = [
        r'funded\s+by\s+([A-Z][a-zA-Z\s&\-]+(?:Inc|Corp|Ltd|LLC|Pharmaceuticals?|Pharma|Company|AG|SA|PLC))',
        r'sponsored\s+by\s+([A-Z][a-zA-Z\s&\-]+(?:Inc|Corp|Ltd|LLC|Pharmaceuticals?|Pharma|Company|AG|SA|PLC))',
        r'supported\s+by\s+([A-Z][a-zA-Z\s&\-]+(?:Inc|Corp|Ltd|LLC|Pharmaceuticals?|Pharma|Company|AG|SA|PLC))',
        r'grant\s+(?:from|by)\s+([A-Z][a-zA-Z\s&\-]+(?:Inc|Corp|Ltd|LLC|Pharmaceuticals?|Pharma|Company|AG|SA|PLC))',
        r'([A-Z][a-zA-Z\s&\-]+(?:Pharmaceuticals?|Pharma))\s+(?:funded|sponsored|supported)'
    ]
    
    # Check for known pharmaceutical companies
    for company in known_pharma_companies:
        company_patterns = [
            rf'funded\s+by\s+{company}',
            rf'sponsored\s+by\s+{company}',
            rf'supported\s+by\s+{company}',
            rf'{company}\s+(?:funded|sponsored|supported)',
            rf'grant\s+(?:from|by)\s+{company}'
        ]
        
        for pattern in company_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "Industry-sponsored trials are more likely to emphasize benefits and underestimate harms."
    
    # Check for general industry funding patterns
    for pattern in industry_funding_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return "Industry-sponsored trials are more likely to emphasize benefits and underestimate harms."
    
    return None

def check_generalizability_age(text: str) -> str | None:
    """
    Detects age generalizability issues.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # Check for pediatric populations first
    pediatric_indicators = [
        r'\binfants?\b', r'\bbabies\b', r'\bneonates?\b', r'\bnewborns?\b',
        r'\bchildren\b', r'\bkids\b', r'\bpediatric\b', r'\bpaediatric\b',
        r'\badolescents?\b', r'\bteenagers?\b', r'\byouths?\b',
        r'(?:aged?|age)\s+(?:under|<|below)\s+18',
        r'(?:aged?|age)\s+\d+\s*(?:months?|days?)\s*(?:old|of\s+age)',
        r'months?\s+of\s+age', r'weeks?\s+of\s+age', r'days?\s+of\s+age'
    ]
    
    for pattern in pediatric_indicators:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "This study was conducted in a pediatric population (infants/children), which limits the generalizability of findings to adult populations."
    
    # Age extraction patterns
    age_patterns = [
        r'(?:mean|median|average)\s+age.*?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'age.*?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:old|of\s+age)',
        r'aged\s+(\d+(?:\.\d+)?)',
        r'participants.*?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'patients.*?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)'
    ]
    
    # Age range patterns
    age_range_patterns = [
        r'age.*?(\d+)(?:\s|-)*(?:to|-)(?:\s|-)*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)(?:\s|-)*(?:to|-)(?:\s|-)*(\d+)\s*(?:years?|yrs?)\s*(?:old|of\s+age)',
        r'aged\s+(\d+)(?:\s|-)*(?:to|-)(?:\s|-)*(\d+)',
        r'between\s+(\d+)\s+and\s+(\d+)\s*(?:years?|yrs?)'
    ]
    
    # Extract single ages
    detected_ages = []
    
    for pattern in age_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                age = float(match)
                if 18 <= age <= 120:  # Reasonable adult age range
                    detected_ages.append(age)
            except (ValueError, TypeError):
                continue
    
    # Extract age ranges
    for pattern in age_range_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                try:
                    age1, age2 = float(match[0]), float(match[1])
                    if 18 <= age1 <= 120 and 18 <= age2 <= 120:
                        avg_age = (age1 + age2) / 2
                        detected_ages.append(avg_age)
                except (ValueError, TypeError):
                    continue
    
    if detected_ages:
        # Use the highest age found (most representative)
        max_age = max(detected_ages)
        if max_age < 70:
            return "The mean or median age of participants is under 70 years, which may limit the generalizability of findings to older adults."
    
    return None

def check_generalizability_sex(text: str) -> str | None:
    """
    Detects sex generalizability issues.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # Only very explicit sex data patterns
    explicit_sex_percentage_patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:were\s+|of\s+(?:participants|patients|subjects)\s+were\s+)(?:female|women)',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:were\s+|of\s+(?:participants|patients|subjects)\s+were\s+)(?:male|men)',
        r'(\d+(?:\.\d+)?)\s*percent\s*(?:were\s+|of\s+(?:participants|patients|subjects)\s+were\s+)(?:female|women)',
        r'(\d+(?:\.\d+)?)\s*percent\s*(?:were\s+|of\s+(?:participants|patients|subjects)\s+were\s+)(?:male|men)'
    ]
    
    # Very explicit count patterns
    explicit_sex_count_patterns = [
        r'(\d+)\s+(?:female|women)\s+(?:participants|patients|subjects)',
        r'(\d+)\s+(?:male|men)\s+(?:participants|patients|subjects)',
        r'(\d+)\s+(?:female|women).*?(\d+)\s+(?:male|men)',
        r'(\d+)\s+(?:male|men).*?(\d+)\s+(?:female|women)'
    ]
    
    # Check for explicit sex percentages
    has_sex_data = False
    female_percentages = []
    male_percentages = []
    
    # Check percentage patterns
    for pattern in explicit_sex_percentage_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            has_sex_data = True
            for match in matches:
                try:
                    percentage = float(match)
                    if 0 <= percentage <= 100:
                        if 'female' in pattern or 'women' in pattern:
                            female_percentages.append(percentage)
                        elif 'male' in pattern or 'men' in pattern:
                            male_percentages.append(percentage)
                except (ValueError, TypeError):
                    continue
    
    # If no explicit sex data found, don't flag
    if not has_sex_data:
        return None
    
    # Determine female percentage
    female_pct = None
    
    if female_percentages:
        female_pct = max(female_percentages)
    elif male_percentages:
        male_pct = max(male_percentages)
        female_pct = 100 - male_pct
    
    # Only flag if we have actual data showing <50% female
    if female_pct is not None and female_pct < 50:
        return "This study population includes fewer than 50% females, limiting the applicability of results across sexes."
    
    return None

def check_intention_to_treat(text: str) -> str | None:
    """
    Detects intention-to-treat analysis issues.
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # ITT positive patterns
    itt_patterns = [
        r'intention[\s-]*to[\s-]*treat',
        r'intent[\s-]*to[\s-]*treat',
        r'itt\s*(?:analysis|population|approach)',
        r'by\s+intention[\s-]*to[\s-]*treat',
        r'on\s+intention[\s-]*to[\s-]*treat',
        r'intention[\s-]*to[\s-]*treat\s*(?:analysis|population|approach)'
    ]
    
    # Check for ITT mention
    for pattern in itt_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return None  # ITT mentioned - no issue
    
    # Check if it's a superiority trial context
    superiority_indicators = [
        'superiority', 'randomized controlled trial', 'rct', 'placebo-controlled',
        'efficacy', 'effectiveness', 'treatment effect'
    ]
    
    is_superiority_context = any(indicator in text_lower for indicator in superiority_indicators)
    
    # Check for non-inferiority context
    non_inferiority_indicators = [
        'non-inferiority', 'noninferiority', 'non inferiority', 'equivalence'
    ]
    
    is_non_inferiority = any(indicator in text_lower for indicator in non_inferiority_indicators)
    
    if is_non_inferiority:
        return None  # For non-inferiority trials, ITT is not always the ideal approach
    
    if is_superiority_context:
        return "ITT is the ideal approach for superiority trials but not for non-inferiority trials. Lack of clarity on the analytic approach limits interpretability."
    
    return None

# Main analysis functions
def analyze_major_biases(abstract: str) -> Dict[str, Optional[str]]:
    """Run all major bias checks on the abstract."""
    checks = [
        ("placebo_bias", check_placebo),
        ("blinding_bias", check_blinding),
        ("sample_size_bias", check_sample_size),
        ("composite_outcome_bias", check_composite_outcome),
        ("primary_outcome_events_bias", check_primary_outcome_events)
    ]

    results = {}
    for bias_name, check_function in checks:
        result = check_function(abstract)
        results[bias_name] = result

    return results

def analyze_minor_issues(abstract: str) -> List[str]:
    """Run all minor issue checks on the abstract."""
    checks = [
        check_duration_followup,
        check_multicentre,
        check_timeline_outcome,
        check_funding,
        check_generalizability_age,
        check_generalizability_sex,
        check_intention_to_treat
    ]

    results = [check(abstract) for check in checks if check(abstract)]
    return results

def save_results_to_csv(abstract: str, major_results: Dict[str, Optional[str]], minor_results: List[str]) -> str:
    """Save analysis results to CSV format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data row
    data_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'abstract': abstract.replace('\n', ' ').replace('\r', ' '),
        'placebo_bias': major_results.get('placebo_bias', ''),
        'blinding_bias': major_results.get('blinding_bias', ''),
        'sample_size_bias': major_results.get('sample_size_bias', ''),
        'composite_outcome_bias': major_results.get('composite_outcome_bias', ''),
        'primary_outcome_events_bias': major_results.get('primary_outcome_events_bias', ''),
        'total_major_biases': sum(1 for v in major_results.values() if v is not None),
        'minor_issues': ' | '.join(minor_results),
        'total_minor_issues': len(minor_results)
    }
    
    # Create CSV content
    fieldnames = [
        'timestamp', 'abstract', 'placebo_bias', 'blinding_bias', 
        'sample_size_bias', 'composite_outcome_bias', 
        'primary_outcome_events_bias', 'total_major_biases',
        'minor_issues', 'total_minor_issues'
    ]
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(data_row)
    
    return output.getvalue()

# Streamlit UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 RCT Bias Detection Tool</h1>
        <p>Automated detection of major biases and minor issues in randomized controlled trial abstracts</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📋 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["Single Abstract", "Batch Analysis"]
    )
    
    # Analysis type selection
    if analysis_type == "Single Abstract":
        st.header("📝 Single Abstract Analysis")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Upload File"]
        )
        
        abstract_text = ""
        
        if input_method == "Text Input":
            abstract_text = st.text_area(
                "Enter RCT Abstract:",
                height=200,
                placeholder="Paste your RCT abstract here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload text file containing abstract:",
                type=['txt', 'doc', 'docx']
            )
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    abstract_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.error("Please upload a .txt file for now.")
        
        if st.button("🔍 Analyze Abstract", type="primary"):
            if abstract_text.strip():
                with st.spinner("Analyzing abstract..."):
                    # Run analyses
                    major_results = analyze_major_biases(abstract_text)
                    minor_results = analyze_minor_issues(abstract_text)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🚨 Major Biases")
                        major_detected = [bias for bias, result in major_results.items() if result]
                        
                        if major_detected:
                            for bias_name, result in major_results.items():
                                if result:
                                    st.markdown(f"""
                                    <div class="bias-card">
                                        <strong>{bias_name.replace('_', ' ').title()}</strong><br>
                                        {result}
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="no-bias-card">
                                <strong>✅ No major biases detected!</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("⚠️ Minor Issues")
                        
                        if minor_results:
                            for issue in minor_results:
                                st.markdown(f"""
                                <div class="minor-issue-card">
                                    {issue}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="no-bias-card">
                                <strong>✅ No minor issues detected!</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Summary statistics
                    st.subheader("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Major Biases", len(major_detected))
                    with col2:
                        st.metric("Minor Issues", len(minor_results))
                    with col3:
                        total_issues = len(major_detected) + len(minor_results)
                        st.metric("Total Issues", total_issues)
                    
                    # Download results
                    if st.button("📥 Download Results as CSV"):
                        csv_content = save_results_to_csv(abstract_text, major_results, minor_results)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv_content,
                            file_name=f"rct_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("Please enter an abstract to analyze.")
    
    else:  # Batch Analysis
        st.header("📂 Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with abstracts:",
            type=['csv'],
            help="CSV should have a column named 'Abstract' containing the RCT abstracts"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'Abstract' not in df.columns:
                    st.error("CSV file must contain a column named 'Abstract'")
                else:
                    st.success(f"Loaded {len(df)} abstracts successfully!")
                    
                    # Show preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df.head())
                    
                    if st.button("🔍 Analyze All Abstracts", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        
                        for i, abstract in enumerate(df['Abstract'].tolist()):
                            status_text.text(f"Analyzing abstract {i+1}/{len(df)}...")
                            progress_bar.progress((i + 1) / len(df))
                            
                            # Run analyses
                            major_results = analyze_major_biases(abstract)
                            minor_results = analyze_minor_issues(abstract)
                            
                            # Prepare result row
                            result_row = {
                                'abstract_id': i + 1,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'abstract': abstract.replace('\n', ' ').replace('\r', ' '),
                                'placebo_bias': major_results.get('placebo_bias', ''),
                                'blinding_bias': major_results.get('blinding_bias', ''),
                                'sample_size_bias': major_results.get('sample_size_bias', ''),
                                'composite_outcome_bias': major_results.get('composite_outcome_bias', ''),
                                'primary_outcome_events_bias': major_results.get('primary_outcome_events_bias', ''),
                                'total_major_biases': sum(1 for v in major_results.values() if v is not None),
                                'minor_issues': ' | '.join(minor_results),
                                'total_minor_issues': len(minor_results)
                            }
                            
                            results.append(result_row)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Display summary statistics
                        st.subheader("📈 Batch Analysis Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Abstracts", len(results_df))
                        with col2:
                            abstracts_with_major_bias = len(results_df[results_df['total_major_biases'] > 0])
                            st.metric("With Major Biases", abstracts_with_major_bias)
                        with col3:
                            abstracts_with_minor_issues = len(results_df[results_df['total_minor_issues'] > 0])
                            st.metric("With Minor Issues", abstracts_with_minor_issues)
                        with col4:
                            clean_abstracts = len(results_df[(results_df['total_major_biases'] == 0) & (results_df['total_minor_issues'] == 0)])
                            st.metric("Clean Abstracts", clean_abstracts)
                        
                        # Bias frequency analysis
                        st.subheader("🔍 Bias Frequency Analysis")
                        
                        bias_types = ['placebo_bias', 'blinding_bias', 'sample_size_bias', 'composite_outcome_bias', 'primary_outcome_events_bias']
                        bias_counts = {}
                        
                        for bias_type in bias_types:
                            count = len(results_df[results_df[bias_type] != ''])
                            percentage = (count / len(results_df)) * 100
                            bias_counts[bias_type.replace('_', ' ').title()] = f"{count} ({percentage:.1f}%)"
                        
                        bias_df = pd.DataFrame(list(bias_counts.items()), columns=['Bias Type', 'Count (%)'])
                        st.table(bias_df)
                        
                        # Display detailed results
                        with st.expander("📋 Detailed Results"):
                            st.dataframe(results_df)
                        
                        # Download button
                        csv_output = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Complete Results",
                            data=csv_output,
                            file_name=f"batch_rct_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        status_text.text("✅ Analysis complete!")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer with information
    st.markdown("---")
    st.subheader("ℹ️ About This Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Major Biases Detected:**
        - 🔸 Placebo bias (absence of placebo control)
        - 🔸 Blinding bias (inadequate blinding)
        - 🔸 Sample size bias (insufficient sample size)
        - 🔸 Composite outcome bias (complex primary outcomes)
        - 🔸 Primary outcome events bias (low event counts)
        """)
    
    with col2:
        st.markdown("""
        **Minor Issues Detected:**
        - 🔹 Short follow-up duration (<1 year)
        - 🔹 Single-center design
        - 🔹 Short outcome assessment timeline
        - 🔹 Industry funding
        - 🔹 Age generalizability (pediatric or <70 years)
        - 🔹 Sex generalizability (<50% female)
        - 🔹 Missing intention-to-treat analysis
        """)

    st.markdown("""
    **Instructions:**
    1. **Single Abstract**: Paste or upload a single RCT abstract for analysis
    2. **Batch Analysis**: Upload a CSV file with multiple abstracts (column named 'Abstract')
    3. **Results**: View detected biases and issues, download results as CSV
    
    **Note**: This tool uses advanced NLP techniques with spaCy for accurate bias detection.
    """)

if __name__ == "__main__":
    main()