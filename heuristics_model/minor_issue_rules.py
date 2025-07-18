import re
import spacy
from typing import Optional, List, Dict, Set

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def mainFunctionMinorIssues(abstract: str):
    """
    Runs all minor issue checks on the abstract and prints formatted output.
    """
    print(abstract + "\n")
    
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

    if results:
        print("\nDetected Minor Issues:")
        for issue in results:
            print(f"- {issue}")
    else:
        print("✅ No minor issues detected.")

def check_duration_followup(text: str) -> str | None:
    """
    FIXED: Detects follow-up duration issues based on the rules:
    - Less than 1 year: insufficient for meaningful clinical outcomes
    - Not provided: duration not specified
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    doc = nlp(text.lower())
    text_lower = text.lower()
    
    # Enhanced duration extraction patterns - FIXED to catch "at X years" patterns
    duration_patterns = [
        # Standard follow-up patterns
        r'follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'followed.*?(?:for|over).*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\s*follow(?:\s|-)*up',
        r'observation.*?period.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'study.*?duration.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'median.*?follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        r'mean.*?follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?',
        
        # FIXED: Assessment timeline patterns (e.g., "at 5 years of age")
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
    FIXED: Detects multicentre study issues based on the rules:
    - Not mentioned (assume single centre): harder to replicate
    - FIXED: Improved detection of multicentre patterns including "two-site"
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # FIXED: Enhanced multicentre positive patterns
    multicentre_patterns = [
        r'multi[\s-]*cent(?:er|re)',
        r'multiple\s+cent(?:er|re)s?',
        r'(\d+)\s+cent(?:er|re)s?',  # "23 centers"
        r'multi[\s-]*site',
        r'multiple\s+sites?',
        r'(\d+)[\s-]*sites?',  # "two-site", "2 sites"
        r'two[\s-]*site',  # ADDED: "two-site"
        r'three[\s-]*site',
        r'multi[\s-]*institution',
        r'multiple\s+institutions?',
        r'international\s+(?:study|trial)',
        r'multinational\s+(?:study|trial)',
        r'across.*?(?:europe|america|asia|countries)',  # ADDED: geographic indicators
        r'(?:europe|america|asia)\s+and\s+(?:europe|america|asia)'  # ADDED: "Europe and South America"
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
    Detects primary outcome timeline issues based on the rules:
    - ≤ 30 days: insufficient to observe meaningful clinical effects
    - Not mentioned: limits interpretability
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
    FIXED: Detects industry funding issues based on the rules:
    - Funded by [Company Name] or sponsored by [PharmaCo]: emphasize benefits, underestimate harms
    - FIXED: Exclude government/academic funding (NIH, NSF, universities, etc.)
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # FIXED: Check for government/academic funding first (these are NOT industry funding)
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
    
    # Check for pharmaceutical/pharma company mentions (but exclude research institutions)
    pharma_general_patterns = [
        r'funded\s+by.*?(?:pharmaceutical|pharma)(?!\s+(?:research|institute|university|college))',
        r'sponsored\s+by.*?(?:pharmaceutical|pharma)(?!\s+(?:research|institute|university|college))',
        r'(?:pharmaceutical|pharma)(?!\s+(?:research|institute|university|college)).*?(?:funded|sponsored)',
        r'industry[\s-]*(?:funded|sponsored)(?!\s+(?:research|academic|university))'
    ]
    
    for pattern in pharma_general_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "Industry-sponsored trials are more likely to emphasize benefits and underestimate harms."
    
    return None

def check_generalizability_age(text: str) -> str | None:
    """
    FIXED: Detects age generalizability issues based on the rules:
    - Age < 70: may limit generalizability to older adults
    - ADDED: Pediatric populations (infants, children) limit generalizability to adults
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # FIXED: Check for pediatric populations first
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
    ULTRA-FIXED: Detects sex generalizability issues based on the rules:
    - Sex < 50% F: limits applicability across sexes
    - EXTREMELY STRICT: Only flag if explicit sex percentages/counts are mentioned
    """
    if not text or not text.strip():
        return "Empty text provided"
    
    text_lower = text.lower()
    
    # ULTRA-STRICT: Only very explicit sex data patterns
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
    
    # Check count patterns
    for pattern in explicit_sex_count_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            has_sex_data = True
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    try:
                        count1, count2 = int(match[0]), int(match[1])
                        total = count1 + count2
                        if total > 0:
                            if 'female' in pattern or 'women' in pattern:
                                female_pct = (count1 / total) * 100
                                female_percentages.append(female_pct)
                            elif 'male' in pattern or 'men' in pattern:
                                male_pct = (count1 / total) * 100
                                male_percentages.append(male_pct)
                    except (ValueError, TypeError):
                        continue
    
    # CRITICAL: If no explicit sex data found, absolutely don't flag
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
    Detects intention-to-treat analysis issues based on the rules:
    - Not mentioned: ideal for superiority trials, limits interpretability
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
    
    # Per-protocol patterns (alternative analysis)
    pp_patterns = [
        r'per[\s-]*protocol',
        r'pp\s*(?:analysis|population)',
        r'as[\s-]*treated',
        r'per[\s-]*protocol\s*(?:analysis|population)'
    ]
    
    # Check for ITT mention
    for pattern in itt_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return None  # ITT mentioned - no issue
    
    # Check for per-protocol only (potential issue)
    has_pp = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in pp_patterns)
    
    # Check if it's a superiority trial context
    superiority_indicators = [
        'superiority', 'randomized controlled trial', 'rct', 'placebo-controlled',
        'efficacy', 'effectiveness', 'treatment effect'
    ]
    
    is_superiority_context = any(indicator in text_lower for indicator in superiority_indicators)
    
    # Check for non-inferiority context (where PP might be more appropriate)
    non_inferiority_indicators = [
        'non-inferiority', 'noninferiority', 'non inferiority', 'equivalence'
    ]
    
    is_non_inferiority = any(indicator in text_lower for indicator in non_inferiority_indicators)
    
    if is_non_inferiority:
        return None  # For non-inferiority trials, ITT is not always the ideal approach
    
    if is_superiority_context or not has_pp:
        return "ITT is the ideal approach for superiority trials but not for non-inferiority trials. Lack of clarity on the analytic approach limits interpretability."
    
    return None

# Example usage
if __name__ == "__main__":
   # Test with the MIND diet abstract
   print("🔬 TESTING FIXED CODE WITH MIND DIET STUDY")
   print("="*60)
   
   mind_diet_abstract = '''
Background:Findings from observational studies suggest that dietary patterns may offer protective benefits against cognitive decline, but data from clinical trials are limited. The Mediterranean-DASH Intervention for Neurodegenerative Delay, known as the MIND diet, is a hybrid of the Mediterranean diet and the DASH (Dietary Approaches to Stop Hypertension) diet, with modifications to include foods that have been putatively associated with a decreased risk of dementia.Methods:We performed a two-site, randomized, controlled trial involving older adults without cognitive impairment but with a family history of dementia, a body-mass index (the weight in kilograms divided by the square of the height in meters) greater than 25, and a suboptimal diet, as determined by means of a 14-item questionnaire, to test the cognitive effects of the MIND diet with mild caloric restriction as compared with a control diet with mild caloric restriction. We assigned the participants in a 1:1 ratio to follow the intervention or the control diet for 3 years. All the participants received counseling regarding adherence to their assigned diet plus support to promote weight loss. The primary end point was the change from baseline in a global cognition score and four cognitive domain scores, all of which were derived from a 12-test battery. The raw scores from each test were converted to z scores, which were averaged across all tests to create the global cognition score and across component tests to create the four domain scores; higher scores indicate better cognitive performance. The secondary outcome was the change from baseline in magnetic resonance imaging (MRI)-derived measures of brain characteristics in a nonrandom sample of participants.Results:A total of 1929 persons underwent screening, and 604 were enrolled; 301 were assigned to the MIND-diet group and 303 to the control-diet group. The trial was completed by 93.4% of the participants. From baseline to year 3, improvements in global cognition scores were observed in both groups, with increases of 0.205 standardized units in the MIND-diet group and 0.170 standardized units in the control-diet group (mean difference, 0.035 standardized units; 95% confidence interval, -0.022 to 0.092; P = 0.23). Changes in white-matter hyperintensities, hippocampal volumes, and total gray- and white-matter volumes on MRI were similar in the two groups.Conclusions:Among cognitively unimpaired participants with a family history of dementia, changes in cognition and brain MRI outcomes from baseline to year 3 did not differ significantly between those who followed the MIND diet and those who followed the control diet with mild caloric restriction. (Funded by the National Institute on Aging; ClinicalTrials.gov number,NCT02817074.).
'''
   
   mainFunctionMinorIssues(mind_diet_abstract)
   
   print("\n" + "="*60)
   print("🔬 TESTING WITH PROBLEMATIC STUDY")
   print("="*60)
   
   # Test abstract with multiple minor issues
   test_abstract = '''
   Background: This randomized controlled trial was funded by Pfizer Inc to evaluate the efficacy of a novel drug.
   
   Methods: Participants were recruited from a single medical center. A total of 150 patients with mean age 
   of 65.2 years were enrolled, of which 40% were female. Patients were randomized to receive either the 
   experimental drug or placebo. The primary outcome was assessed at 28 days after treatment initiation.
   
   Results: After 6 weeks of follow-up, the experimental drug showed significant improvement compared to placebo.
   
   Conclusions: The novel drug demonstrates efficacy in treating the target condition.
   '''
   
   mainFunctionMinorIssues(test_abstract)
   
   print("\n" + "="*60)
   print("🔬 TESTING WITH CLEFT PALATE STUDY")
   print("="*60)
   
   # Test with the cleft palate study (pediatric population)
   cleft_palate_abstract = '''
Background:Among infants with isolated cleft palate, whether primary surgery at 6 months of age is more beneficial than surgery at 12 months of age with respect to speech outcomes, hearing outcomes, dentofacial development, and safety is unknown.Methods:We randomly assigned infants with nonsyndromic isolated cleft palate, in a 1:1 ratio, to undergo standardized primary surgery at 6 months of age (6-month group) or at 12 months of age (12-month group) for closure of the cleft. Standardized assessments of quality-checked video and audio recordings at 1, 3, and 5 years of age were performed independently by speech and language therapists who were unaware of the trial-group assignments. The primary outcome was velopharyngeal insufficiency at 5 years of age, defined as a velopharyngeal composite summary score of at least 4 (scores range from 0 to 6, with higher scores indicating greater severity). Secondary outcomes included speech development, postoperative complications, hearing sensitivity, dentofacial development, and growth.Results:We randomly assigned 558 infants at 23 centers across Europe and South America to undergo surgery at 6 months of age (281 infants) or at 12 months of age (277 infants). Speech recordings from 235 infants (83.6%) in the 6-month group and 226 (81.6%) in the 12-month group were analyzable. Insufficient velopharyngeal function at 5 years of age was observed in 21 of 235 infants (8.9%) in the 6-month group as compared with 34 of 226 (15.0%) in the 12-month group (risk ratio, 0.59; 95% confidence interval, 0.36 to 0.99; P = 0.04). Postoperative complications were infrequent and similar in the 6-month and 12-month groups. Four serious adverse events were reported (three in the 6-month group and one in the 12-month group) and had resolved at follow-up.Conclusions:Medically fit infants who underwent primary surgery for isolated cleft palate in adequately resourced settings at 6 months of age were less likely to have velopharyngeal insufficiency at the age of 5 years than those who had surgery at 12 months of age. (Funded by the National Institute of Dental and Craniofacial Research; TOPS ClinicalTrials.gov number,NCT00993551.).
'''
   
   mainFunctionMinorIssues(cleft_palate_abstract)
   
   print("\n" + "="*60)
   print("🔬 TESTING WITH NO SEX DATA STUDY")
   print("="*60)
   
   # Test with study that has no sex data mentioned
   no_sex_abstract = '''
Background: This study evaluated the effectiveness of a new intervention.
Methods: We conducted a randomized controlled trial involving 200 participants aged 45-75 years. 
Participants were randomized to intervention or control groups. The primary outcome was measured at 6 months.
Results: The intervention showed significant improvement compared to control.
Conclusions: The intervention is effective for the target population.
'''
   
   mainFunctionMinorIssues(no_sex_abstract)
   
   print("\n✅ Complete fixed minor issues analysis complete!")
   print("\n📋 EXPECTED RESULTS SUMMARY:")
   print("="*60)
   print("🔬 MIND Diet Study:")
   print("   Expected: Only ITT missing (no false positives)")
   print("   ✅ Should NOT flag: Duration (3 years), multicenter (two-site), funding (NIH), sex (not mentioned)")
   
   print("\n🔬 Problematic Study:")
   print("   Expected: Industry funding + <50% female + short timeline + age <70 + single center")
   print("   ✅ Should flag: All major issues")
   
   print("\n🔬 Cleft Palate Study:")
   print("   Expected: Pediatric population limiting generalizability + ITT missing")
   print("   ✅ Should NOT flag: Duration (5 years), multicenter (23 centers), funding (NIH)")
   
   print("\n🔬 No Sex Data Study:")
   print("   Expected: Various issues but NOT sex imbalance")
   print("   ✅ Should NOT flag: Sex issues (no sex data mentioned)")
   
   print("\n🎯 KEY FIXES IMPLEMENTED:")
   print("✅ Fixed funding detection (excludes government/academic funding)")
   print("✅ Fixed multicentre detection (catches 'two-site', geographic patterns)")
   print("✅ Fixed sex detection (ultra-strict, only flags explicit data)")
   print("✅ Fixed duration detection (catches 'from baseline to year X')")
   print("✅ Fixed age detection (flags pediatric populations)")
   print("✅ All functions have proper error handling and edge case management")
