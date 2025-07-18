#re is library for regular expressions re
import re
from unittest import result
import spacy
from typing import Optional, List, Dict, Set
import csv
import pandas as pd
from datetime import datetime
import os

#defining some models we are going to use 
nlp = spacy.load("en_core_web_sm")

def mainFunction(abstract: str, save_to_csv: bool = True, csv_filename: str = None):
    """
    Runs all major issue checks on the abstract and prints formatted output.
    Optionally saves results to CSV file.
    
    Args:
        abstract: The RCT abstract text
        save_to_csv: Whether to save results to CSV file
        csv_filename: Custom CSV filename (optional)
    """
    print(abstract + "\n")
    
    # Run all bias checks
    checks = [
        ("placebo_bias", check_placebo),
        ("blinding_bias", check_blinding),
        ("sample_size_bias", check_sample_size),
        ("composite_outcome_bias", check_composite_outcome),
        ("primary_outcome_events_bias", check_primary_outcome_events)
    ]

    # Store results in dictionary
    results = {}
    detected_issues = []
    
    for bias_name, check_function in checks:
        result = check_function(abstract)
        results[bias_name] = result
        if result:
            detected_issues.append(result)

    # Print results
    if detected_issues:
        print("\nDetected Major Issues:")
        for issue in detected_issues:
            print(f"- {issue}")
    else:
        print("✅ No major issues detected.")
    
    # Save to CSV if requested
    if save_to_csv:
        save_results_to_csv(abstract, results, csv_filename)
        
    return results

def save_results_to_csv(abstract: str, results: Dict[str, Optional[str]], csv_filename: str = None):
    """
    Save abstract and bias detection results to CSV file.
    
    Args:
        abstract: The RCT abstract text
        results: Dictionary with bias detection results
        csv_filename: Custom CSV filename (optional)
    """
    # Generate filename if not provided
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"rct_bias_results_{timestamp}.csv"
    
    # Prepare data row
    data_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'abstract': abstract.replace('\n', ' ').replace('\r', ' '),  # Clean newlines
        'placebo_bias': results.get('placebo_bias', ''),
        'blinding_bias': results.get('blinding_bias', ''),
        'sample_size_bias': results.get('sample_size_bias', ''),
        'composite_outcome_bias': results.get('composite_outcome_bias', ''),
        'primary_outcome_events_bias': results.get('primary_outcome_events_bias', ''),
        'total_biases_detected': sum(1 for v in results.values() if v is not None)
    }
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_filename)
    
    # Write to CSV
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'timestamp', 'abstract', 'placebo_bias', 'blinding_bias', 
            'sample_size_bias', 'composite_outcome_bias', 
            'primary_outcome_events_bias', 'total_biases_detected'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(data_row)
    
    print(f"\n📊 Results saved to: {csv_filename}")

def analyze_multiple_abstracts(abstracts: List[str], csv_filename: str = "batch_rct_analysis.csv"):
    """
    Analyze multiple abstracts and save all results to one CSV file.
    
    Args:
        abstracts: List of abstract texts
        csv_filename: CSV filename for batch results
    """
    print(f"🔬 Analyzing {len(abstracts)} abstracts...")
    
    all_results = []
    
    for i, abstract in enumerate(abstracts, 1):
        print(f"\n{'='*60}")
        print(f"ANALYZING ABSTRACT {i}/{len(abstracts)}")
        print(f"{'='*60}")
        
        # Analyze abstract (don't save individual CSV)
        results = mainFunction(abstract, save_to_csv=False)
        
        # Prepare data for batch CSV
        data_row = {
            'abstract_id': i,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'abstract': abstract.replace('\n', ' ').replace('\r', ' '),
            'placebo_bias': results.get('placebo_bias', ''),
            'blinding_bias': results.get('blinding_bias', ''),
            'sample_size_bias': results.get('sample_size_bias', ''),
            'composite_outcome_bias': results.get('composite_outcome_bias', ''),
            'primary_outcome_events_bias': results.get('primary_outcome_events_bias', ''),
            'total_biases_detected': sum(1 for v in results.values() if v is not None)
        }
        
        all_results.append(data_row)
    
    # Save all results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"\n🎉 Batch analysis complete!")
    print(f"📊 All results saved to: {csv_filename}")
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"Total abstracts analyzed: {len(abstracts)}")
    print(f"Abstracts with no bias: {len(df[df['total_biases_detected'] == 0])}")
    print(f"Abstracts with bias detected: {len(df[df['total_biases_detected'] > 0])}")
    
    # Bias type frequency
    bias_types = ['placebo_bias', 'blinding_bias', 'sample_size_bias', 'composite_outcome_bias', 'primary_outcome_events_bias']
    for bias_type in bias_types:
        count = len(df[df[bias_type] != ''])
        percentage = (count / len(df)) * 100
        print(f"{bias_type}: {count} ({percentage:.1f}%)")

def load_abstracts_from_file(filename: str) -> List[str]:
    """
    Load abstracts from a text file (one abstract per line or separated by empty lines).
    
    Args:
        filename: Path to text file containing abstracts
        
    Returns:
        List of abstract strings
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines (assuming abstracts are separated by empty lines)
    abstracts = [abstract.strip() for abstract in content.split('\n\n') if abstract.strip()]
    
    return abstracts

# Your existing functions (keeping them exactly the same)
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
    Uses spaCy for sentence-level analysis with contextual filtering.
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
    
    return None  # Adequate event reporting

def check_composite_outcome(text: str) -> str | None:
    """
    Uses spaCy to detect composite primary outcomes with more flexibility.
    Looks for:
    - Known phrases (e.g., 'composite outcome')
    - Lists of 3+ outcomes joined by 'and'/'or' in the same sentence 
    - we used 3 here becuase 2 in most of the cases will result in the thing such as : stroke or death for example
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
    
    secondary_context_patterns = [
        "secondary outcome", "secondary endpoint", "secondary end point",
        "exploratory outcome", "additional outcome", "safety outcome"
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
    
    # Step 2: Enhanced detection of outcome lists using spaCy
    potential_composites = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        sent_doc = nlp(sent_text)
        
        # Check if sentence is about outcomes
        has_outcome_context = (
            any(pattern in sent_text for pattern in primary_context_patterns + secondary_context_patterns) or
            any(word in sent_text for word in ["outcome", "endpoint", "event", "occurrence"])
        )
        
        if not has_outcome_context:
            continue
        
        # Look for conjunctions and lists
        if (" and " in sent_text or " or " in sent_text) and "," in sent_text:
            
            # Extract potential outcome terms
            outcome_terms = []
            conjunction_count = 0
            
            for token in sent_doc:
                # Count conjunctions
                if token.text in ["and", "or"]:
                    conjunction_count += 1
                
                # Identify medical outcome terms
                if (token.pos_ in ["NOUN", "PROPN"] and 
                    len(token.text) > 2 and 
                    token.text not in non_outcome_terms):
                    
                    # Check if it's a medical outcome
                    is_medical_outcome = (
                        token.text in clinical_outcomes or
                        any(clinical_term in token.text for clinical_term in clinical_outcomes) or
                        # Check surrounding context for medical terms
                        any(clinical_term in [t.text for t in sent_doc[max(0, token.i-2):token.i+3]] 
                            for clinical_term in clinical_outcomes)
                    )
                    
                    if is_medical_outcome:
                        outcome_terms.append(token.text)
            
            # Remove duplicates and filter
            unique_outcomes = list(set(outcome_terms))
            
            # Enhanced validation criteria
            if len(unique_outcomes) >= 3 and conjunction_count >= 1:
                
                # Additional validation using medical knowledge
                hard_endpoint_count = sum(1 for outcome in unique_outcomes if outcome in hard_endpoints)
                soft_endpoint_count = sum(1 for outcome in unique_outcomes if outcome in soft_endpoints)
                surrogate_count = sum(1 for outcome in unique_outcomes if outcome in surrogate_endpoints)
                
                # Classify composite type and severity
                total_outcomes = len(unique_outcomes)
                
                potential_composites.append({
                    'outcomes': unique_outcomes,
                    'hard_endpoints': hard_endpoint_count,
                    'soft_endpoints': soft_endpoint_count,
                    'surrogates': surrogate_count,
                    'total': total_outcomes,
                    'sentence': sent_text,
                    'conjunctions': conjunction_count
                })
    
    # Step 3: Analyze detected composites
    if potential_composites:
        # Find the most significant composite (prioritize primary outcome context)
        primary_composite = None
        secondary_composite = None
        
        for composite in potential_composites:
            is_primary_context = any(pattern in composite['sentence'] 
                                   for pattern in primary_context_patterns)
            
            if is_primary_context:
                primary_composite = composite
                break
            else:
                secondary_composite = composite
        
        main_composite = primary_composite or secondary_composite
        
        if main_composite:
            total_outcomes = main_composite['total']
            hard_count = main_composite['hard_endpoints']
            soft_count = main_composite['soft_endpoints']
            surrogate_count = main_composite['surrogates']
            
            # Generate specific warnings based on composite characteristics
            if primary_composite:
                if hard_count >= 2:
                    return (
                        f"Composite primary outcome with {total_outcomes} components including hard endpoints: "
                        "may dilute effects of clinically important outcomes"
                    )
                elif soft_count >= 2 and hard_count == 0:
                    return (
                        f"Composite primary outcome with {total_outcomes} soft endpoints: "
                        "limited clinical significance and interpretability concerns"
                    )
                elif surrogate_count >= 2:
                    return (
                        f"Composite primary outcome includes {surrogate_count} surrogate endpoints: "
                        "questionable clinical relevance"
                    )
                else:
                    return (
                        f"Composite primary outcome with {total_outcomes} components: "
                        "may reduce interpretability of treatment effects"
                    )
            else:
                return (
                    f"Composite outcome detected with {total_outcomes} components: "
                    "verify clinical meaningfulness of individual components"
                )
    
    # Step 4: Special patterns for common composite outcomes
    common_composites = {
        "mace": ["death", "myocardial infarction", "stroke"],
        "cardiovascular": ["death", "stroke", "heart attack"],
        "safety": ["adverse", "serious", "death"]
    }
    
    for composite_name, components in common_composites.items():
        if (composite_name in text_lower and 
            sum(1 for comp in components if comp in text_lower) >= 2):
            
            return (
                f"Standard composite outcome ({composite_name.upper()}) detected: "
                "verify appropriate component weighting and clinical relevance"
            )
    
    return None

def check_placebo(text: str) -> str | None:
    """
    Checks for the absence of a placebo in the abstract using:
    - Keyword variation detection
    - Negative phrasing detection
    - Contextual co-occurrence of terms
    - Sentence-level parsing via spaCy
    """    
    # Process text with spaCy
    doc = nlp(text)
    text_lower = text.lower()
    
    # Enhanced placebo patterns using spaCy's pattern matching capabilities
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
    
    # Extract drug names using spaCy NER + pattern matching
    drug_names = set()
    
    # Use spaCy NER for drug detection
    for ent in doc.ents:
        if ent.label_ in ["DRUG", "PRODUCT", "CHEMICAL", "PERSON"] and len(ent.text) > 2:
            # Filter out common false positives
            if not any(word in ent.text.lower() for word in ["patient", "study", "trial", "group", "week", "day", "month"]):
                drug_names.add(ent.text.lower())
    
    # Add pattern for generic drug names like "Drug A", "Drug B", "Treatment A", etc.
    generic_drug_patterns = [
        r'\b(?:drug|treatment|therapy|intervention|medication)\s+[a-z]\b',
        r'\b[a-z]\s+(?:drug|treatment|therapy|intervention|medication)\b',
        r'\b(?:arm|group)\s+[a-z]\b'
    ]
    
    for pattern in generic_drug_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            drug_names.add(match.strip())
    
    # Use spaCy's token analysis for drug suffixes
    drug_suffixes = [
        r'\w+mab\b', r'\w+nib\b', r'\w+zumab\b', r'\w+cillin\b',
        r'\w+mycin\b', r'\w+prazole\b', r'\w+statin\b', r'\w+olol\b',
        r'\w+pril\b', r'\w+sartan\b'
    ]
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
            for pattern in drug_suffixes:
                if re.match(pattern, token.text.lower()):
                    drug_names.add(token.text.lower())
    
    # Enhanced detection for capitalized drug references
    drug_reference_patterns = [
        r'\b[A-Z][a-z]*\s+[A-Z]\b',  # "Drug A", "Treatment B"
        r'\b[A-Z]\s+(?:vs|versus)\s+[A-Z]\b',  # "A vs B"
        r'\b(?:drug|treatment)\s+[A-Z]\b'  # "drug A"
    ]
    
    for pattern in drug_reference_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if not any(word in match.lower() for word in ["patient", "study", "trial", "group"]):
                drug_names.add(match.lower())
    
    # Use spaCy for better pattern matching
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
    
    # Boolean flags using spaCy-enhanced detection
    found_placebo = any(p in text_lower for p in placebo_positive_patterns) and not has_negated_placebo
    found_negative = any(n in text_lower for n in placebo_negative_patterns) or has_negated_placebo
    has_blinding = any(b in text_lower for b in blinding_patterns)
    has_control = any(c in text_lower for c in control_patterns)
    head_to_head_trial = any(phrase in text_lower for phrase in comparison_phrases) and len(drug_names) >= 2
    
    # Use spaCy for more sophisticated comparison detection
    has_comparison = False
    for token in doc:
        if token.lemma_ in ["compare", "versus", "vs"] or token.text.lower() in ["vs", "versus"]:
            has_comparison = True
            break
    
    # Enhanced head-to-head detection using spaCy
    if has_comparison and len(drug_names) >= 2:
        head_to_head_trial = True
    
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
    
    # 7. Standard of care comparisons using spaCy
    soc_detected = False
    for token in doc:
        if token.lemma_ in ["standard", "usual", "current"] and any(next_token.lemma_ in ["care", "practice", "treatment"] for next_token in doc[token.i:token.i+3]):
            soc_detected = True
            break
    
    if soc_detected:
        if not found_placebo and not has_blinding:
            return "Compared to standard of care without placebo or blinding: possible confounding bias"
    
    # 5. Single-arm or unclear design
    if not found_placebo and not has_control and not head_to_head_trial:
        return "No placebo or control group mentioned: possible single-arm study or selection bias"
    
    # 6. Blinding without placebo
    if has_blinding and not found_placebo and not head_to_head_trial:
        return "Claims blinding without placebo mention: verify blinding method"
    
    # 8. No placebo in non-head-to-head
    if not found_placebo and not head_to_head_trial:
        return "No mention of placebo in apparent controlled trial: possible confounding or adjudication bias"
    
    # No bias detected
    return None

def check_blinding(text: str) -> str | None:
    """
    Checks for missing or unreported blinding using:
    - Keyword variation matching
    - Explicit negative phrasing
    - Sentence-level analysis via spaCy
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

    # Assess blinding completeness (for detailed feedback)
    blinding_details = {
        "double_blind": "double-blind" in text or "double blind" in text,
        "single_blind": "single-blind" in text or "single blind" in text,
        "triple_blind": "triple-blind" in text or "triple blind" in text,
        "participants_blinded": any(p in text for p in ["participants blinded", "patients blinded", "subjects blinded"]),
        "investigators_blinded": any(i in text for i in ["investigators blinded", "clinicians blinded", "physicians blinded"]),
        "assessors_blinded": any(a in text for a in ["assessors blinded", "evaluators blinded", "outcome assessors blinded"]),
        "has_blinding_method": has_placebo or "double-dummy" in text or "matching" in text
    }

    # Decision logic with comprehensive rules
    
    # Rule 1: Explicitly unblinded OR negation detected
    if found_unblinded or blinding_negated:
        return (
            "Blinding explicitly not performed (e.g. open-label or unblinded): "
            "high risk of performance, detection, and adjudication bias"
        )

    # Rule 2: Has placebo but no blinding mentioned (concerning)
    if has_placebo and not found_blinding:
        return (
            "Placebo/sham mentioned but no blinding reported: "
            "verify if blinding was properly implemented and reported"
        )

    # Rule 3: RCT design but no blinding mentioned
    if found_study_design and not found_blinding and not has_placebo:
        return (
            "Randomized controlled trial without blinding mention: "
            "possible performance and detection bias"
        )

    # Rule 4: No mention of blinding at all
    if not found_blinding:
        return (
            "No mention of blinding: possible performance, detection, "
            "and adjudication bias due to inadequate blinding"
        )

    # Rule 5: Basic blinding mentioned but lacks detail
    if found_blinding:
        if blinding_details["double_blind"] or blinding_details["triple_blind"]:
            # High-quality blinding detected
            return None
        elif blinding_details["single_blind"]:
            return (
                "Single-blind design: possible performance bias from unblinded participants/investigators"
            )
        elif not any([blinding_details["participants_blinded"], 
                     blinding_details["investigators_blinded"], 
                     blinding_details["assessors_blinded"]]):
            return (
                "Blinding mentioned but lacks detail on who was blinded: "
                "verify completeness of blinding implementation"
            )
        elif not blinding_details["has_blinding_method"]:
            return (
                "Blinding reported but no blinding method described: "
                "verify adequacy of blinding procedure"
            )

    # Rule 6: Well-described blinding
    return None

# Example usage and testing
if __name__ == "__main__":
    # Test with COPD abstract

    # Example 1: Single abstract analysis with CSV export
    print("🔬 SINGLE ABSTRACT ANALYSIS")
    print("="*60)
    mainFunction(abstract, save_to_csv=True, csv_filename="single_analysis.csv")
    

    # Example 2: Multiple abstracts analysis

    abstractDataFram = pd.read_csv("pubmed_abstracts.csv")
    test_abstracts = abstractDataFram['Abstract'].to_list()
    print("\n\n🔬 BATCH ANALYSIS OF MULTIPLE ABSTRACTS")
    print("="*60)
    analyze_multiple_abstracts(test_abstracts, "batch_analysis.csv")
    
    print("\n\n📁 FILES CREATED:")
    print("- single_analysis.csv (single abstract)")
    print("- batch_analysis.csv (multiple abstracts)")
    print("\n✅ Analysis complete! Check the CSV files for detailed results.")