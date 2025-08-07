
# 📘 Regex Heuristic Scoring Explanation

This document explains the reasoning behind assigning different scores to regex patterns used to identify whether a medical abstract describes a **Randomized Controlled Trial (RCT)**.

---

## 🧠 Why Use Scores in Regex Heuristic Patterns?

The goal is to classify abstracts as RCTs or not. Regex patterns are used to search for specific phrases. Each phrase has a different **strength of evidence**, so we group them and assign **scores** accordingly.

---

## 🔴 1. Strong Indicators (Score: 3)

### ✅ Examples:
- `"randomized controlled trial"`
- `"double-blind"`
- `"placebo-controlled"`
- `"RCT"`

### 💡 Why score = 3?
- Highly specific to RCTs.
- Very strong evidence.
- Rarely appear outside actual randomized trials.

---

## 🟠 2. Moderate Indicators (Score: 2)

### ✅ Examples:
- `"single-blind"`
- `"multicenter"`
- `"crossover"`
- `"prospective study"`

### 💡 Why score = 2?
- Common in RCTs but not exclusive.
- Supportive evidence, but not definitive alone.

---

## 🟡 3. Weak Indicators (Score: 1)

### ✅ Examples:
- `"cohort study"`
- `"pilot study"`
- `"feasibility study"`
- `"case-control study"`

### 💡 Why score = 1?
- General research terms.
- Usually not randomized.
- Weak signal for RCT classification.

---

## ⚖️ Why Use Scores?

Scoring helps differentiate the strength of evidence. For example:

### 🎯 Example Threshold Logic:
- **5 or more** → Likely RCT
- **3–4** → Possible RCT
- **<3** → Unlikely RCT

This balances strong and weak evidence intelligently.

---

## 🧪 Can Scores Be Modified?

Yes. These are heuristic (expert-defined) rules and can be:
- Adjusted based on testing
- Learned via machine learning in the future

---

## ✅ Summary Table

| Category  | Score | Example                 | Why This Score? |
|-----------|--------|--------------------------|-----------------|
| Strong    | 3      | `"randomized trial"`     | Strong evidence, specific to RCTs |
| Moderate  | 2      | `"multicenter study"`    | Common in RCTs but not exclusive |
| Weak      | 1      | `"pilot study"`          | General research term, not enough by itself |



---
---
# 📘 Regex Heuristic Scoring: Methodology Patterns

This document explains the reasoning behind assigning different scores to **methodology-related regex patterns** used to assess whether a medical abstract follows a structured clinical trial methodology (especially RCTs).

---

## 🔴 Strong Methodology Indicators (Score: 3)

These indicate strong evidence of randomized trial methodology.

### ✅ Examples:
- `\brandom(?:ly\s+)?(?:assigned|allocated)\b`  
  Matches: "randomly assigned", "randomly allocated"  
  **Why**: Indicates clear randomization, a hallmark of RCTs.

- `\bintention-to-treat\b`  
  Matches: "intention-to-treat"  
  **Why**: Refers to a specific analysis method used in RCTs.

- `\bper-protocol\s+analysis\b`  
  Matches: "per-protocol analysis"  
  **Why**: Indicates strict adherence to study protocol in analysis.

---

## 🟠 Moderate Methodology Indicators (Score: 2)

These are relevant to RCTs but not exclusive to them.

### ✅ Examples:
- `\bstratified\s+randomization\b`  
  Matches: "stratified randomization"  
  **Why**: Shows controlled assignment across strata, common in RCTs.

- `\bblinded?\s+(?:assessment|evaluation)\b`  
  Matches: "blinded assessment", "blind evaluation"  
  **Why**: Reduces bias in outcome evaluation, supports RCT design.

- `\bunblinded?\b`  
  Matches: "unblinded", "unblind"  
  **Why**: Indicates transparency of treatment groups; still design-related.

- `\bopen-label\b`  
  Matches: "open-label"  
  **Why**: A trial format with no blinding, but still structured.

---

## 🟡 Basic Methodology Indicators (Score: 1)

These suggest a research structure but are very general.

### ✅ Examples:
- `\bbaseline\s+characteristics\b`  
  Matches: "baseline characteristics"  
  **Why**: Common descriptive element in all types of studies.

- `\bfollow-up\s+(?:period|duration)\b`  
  Matches: "follow-up period", "follow-up duration"  
  **Why**: Refers to observation length, not unique to RCTs.

- `\benrollment\b`  
  Matches: "enrollment"  
  **Why**: Just states participants were recruited — very generic.

- `\bscreening\b`  
  Matches: "screening"  
  **Why**: Common term in both clinical and observational studies.

- `\beligibility\s+criteria\b`  
  Matches: "eligibility criteria"  
  **Why**: Describes inclusion/exclusion, not specific to trials.

---

## ✅ Summary Table

| Category      | Score | Example                   | Why This Score?                                |
|---------------|-------|---------------------------|------------------------------------------------|
| **Strong**    | 3     | "randomly assigned"       | Specific to structured RCT methodology         |
| **Moderate**  | 2     | "blinded assessment"      | Relevant to trial design but not exclusive     |
| **Basic**     | 1     | "enrollment", "screening" | Common across all research types               |

---

## ⚙️ Purpose of These Scores

These methodology scores help determine how well-structured the study appears to be, and contribute to the overall heuristic assessment of whether the abstract describes an RCT.


---
---

# 📘 Regex Heuristic Scoring: Statistical Patterns

This document explains the reasoning behind assigning scores to **statistical-related regex patterns**, used to assess whether a medical abstract demonstrates rigorous statistical reporting — often a sign of a well-conducted RCT.

---

## 🔬 Statistical Analysis Terms (Score: 2–3)

These indicate formal statistical analysis and effect size reporting.

### ✅ Patterns and Explanations:

- `p\s*[<>=]\s*0\.\d+`  
  Matches: `p < 0.05`, `p=0.001`  
  **Score**: 3  
  **Why**: Clear evidence of statistical testing, common in RCTs.

- `\b95%\s*(?:confidence\s+interval|ci)\b`  
  Matches: `95% confidence interval`, `95% CI`  
  **Score**: 3  
  **Why**: Standard for expressing uncertainty; highly used in trials.

- `\bhazard\s+ratio\b`  
  **Score**: 3  
  **Why**: Advanced effect size measure; strong statistical evidence.

- `\bodds\s+ratio\b`  
  **Score**: 3  
  **Why**: Core measure in many RCT outcomes.

- `\brisk\s+ratio\b`  
  **Score**: 3  
  **Why**: Reflects relative risk between groups — very RCT-relevant.

- `\bmean\s+difference\b`  
  **Score**: 2  
  **Why**: Indicates between-group comparison, but simpler.

- `\bstatistical\s+(?:analysis|significance)\b`  
  Matches: `statistical analysis`, `statistical significance`  
  **Score**: 2  
  **Why**: General, useful for identifying statistical intent.

- `\bpower\s+(?:analysis|calculation)\b`  
  Matches: `power analysis`, `power calculation`  
  **Score**: 2  
  **Why**: Indicates planning rigor — good trial design practice.

- `\bsample\s+size\s+calculation\b`  
  **Score**: 2  
  **Why**: Shows effort to estimate necessary population — well-designed methodology.

---

## 👥 Sample Size Reporting (Score: 1–2)

Patterns that suggest the number of participants but vary in rigor.

- `\bn\s*=\s*\d+`  
  Matches: `n=100`, `n = 50`  
  **Score**: 2  
  **Why**: Directly states the sample size — clean, informative.

- `\b\d+\s+(?:participants?|patients?|subjects?)\b`  
  Matches: `100 participants`, `30 patients`  
  **Score**: 1  
  **Why**: Informal quantity mention — weak indicator on its own.

---

## ✅ Summary Table

| Pattern Example                  | Score | Why?                                           |
|----------------------------------|-------|------------------------------------------------|
| `p < 0.05`                      | 3     | Core statistical test result                   |
| `95% confidence interval`       | 3     | Measures uncertainty — RCT staple              |
| `hazard ratio`, `odds ratio`    | 3     | Strong effect size measures                    |
| `mean difference`               | 2     | Useful but simpler stat                        |
| `statistical analysis`          | 2     | General indicator                              |
| `sample size calculation`       | 2     | Sign of planning rigor                         |
| `n = 100`                       | 2     | Transparent reporting                          |
| `100 patients`                  | 1     | Weak, informal indicator                       |

---

## ⚙️ Purpose of These Scores

These patterns help capture the **statistical strength and rigor** of a study. The higher the score, the more the abstract reflects robust and thoughtful statistical design, analysis, and reporting — all critical in confirming that it’s a high-quality RCT.


---
---
# 📘 Regex Heuristic Scoring: Structure Patterns

This document explains the rationale behind scoring **structure-related regex patterns**, used to assess whether a medical abstract follows a standard scientific format — a key sign of formal research quality.

---

## 🟥 Structured Abstract Sections (Score: 2)

These patterns match **explicit section headers** used in formal scientific abstracts. Their presence indicates that the abstract is likely structured according to journal standards.

### ✅ Patterns and Explanations:

- `\b(?:background|introduction)\s*:`  
  Matches: `Background:`, `Introduction:`  
  **Score**: 2  
  **Why**: Common starting point of structured abstracts.

- `\b(?:objective|aim)s?\s*:`  
  Matches: `Objective:`, `Aims:`  
  **Score**: 2  
  **Why**: Indicates clearly defined research purpose.

- `\bmethods?\s*:`  
  Matches: `Method:`, `Methods:`  
  **Score**: 2  
  **Why**: Presence of methodology section suggests scientific rigor.

- `\bresults?\s*:`  
  Matches: `Result:`, `Results:`  
  **Score**: 2  
  **Why**: Denotes data reporting section — key to research validity.

- `\bconclusions?\s*:`  
  Matches: `Conclusion:`, `Conclusions:`  
  **Score**: 2  
  **Why**: Indicates summary and interpretation of findings.

---

## 🟨 Content Flow Indicators (Score: 1)

These are terms commonly found in abstracts that reflect logical flow, depth of reporting, and trial design clarity — but they’re not section headers.

### ✅ Patterns and Explanations:

- `\bprimary\s+(?:endpoint|outcome)\b`  
  Matches: `primary outcome`, `primary endpoint`  
  **Score**: 1  
  **Why**: Identifies main objective — useful in trial context.

- `\bsecondary\s+(?:endpoint|outcome)\b`  
  Matches: `secondary outcome`, `secondary endpoint`  
  **Score**: 1  
  **Why**: Indicates additional study targets.

- `\bmain\s+(?:finding|result)\b`  
  Matches: `main finding`, `main result`  
  **Score**: 1  
  **Why**: Signals central result of the study.

- `\bclinical\s+(?:significance|implication)\b`  
  Matches: `clinical significance`, `clinical implication`  
  **Score**: 1  
  **Why**: Reflects real-world medical relevance of the findings.

---

## ✅ Summary Table

| Pattern Example             | Score | Why This Score?                                 |
|-----------------------------|-------|--------------------------------------------------|
| `Methods:`                 | 2     | Standard header in structured scientific writing |
| `Results:`                 | 2     | Indicates organized data reporting               |
| `primary outcome`          | 1     | Describes main target of the study               |
| `clinical significance`    | 1     | Adds real-world interpretability                 |

---

## ⚙️ Purpose of These Scores

These patterns help confirm that the abstract:
- Follows a structured format used in peer-reviewed journals
- Contains clearly defined sections and research flow
- Likely represents a well-formed scientific study (possibly an RCT)

While not direct evidence of randomization or methodology, structure patterns provide **contextual support** in the overall classification.



---
---
# 📘 Regex Heuristic Scoring: News Patterns

This document explains the scoring of regex patterns used to detect **news-style language** that suggests the text is a **secondary source** — such as a press release, article, or media report — rather than a primary scientific abstract.

---

## 🟥 Strong News Indicators (Score: 3)

These patterns are characteristic of **journalistic writing**, **media summaries**, or **press announcements**. Their presence strongly suggests the content is **not a research abstract**.

### ✅ Patterns and Explanations:

- `\b(?:reports?|announces?)\s+(?:show|indicate|find)\b`  
  Matches: `report shows`, `announces find`, `reports indicate`  
  **Score**: 3  
  **Why**: Typical news-style phrasing used to summarize external research findings.

- `\baccording\s+to\s+(?:a\s+)?(?:new\s+)?study\b`  
  Matches: `according to a study`, `according to new study`  
  **Score**: 3  
  **Why**: Common news lead-in to introduce a summarized study.

- `\bresearchers?\s+(?:say|report|told\s+reporters)\b`  
  Matches: `researchers say`, `researchers report`, `researcher told reporters`  
  **Score**: 3  
  **Why**: Third-party narration of research — not used in scientific abstracts.

- `\bpublished\s+(?:today|yesterday|this\s+week)\b`  
  Matches: `published today`, `published this week`  
  **Score**: 3  
  **Why**: Indicates recency for public interest — a journalistic element.

- `\bnews\s+release\b`  
  Matches: `news release`  
  **Score**: 3  
  **Why**: Directly signals a media communication, not a peer-reviewed study.

- `\bpress\s+release\b`  
  Matches: `press release`  
  **Score**: 3  
  **Why**: Same as above — identifies promotional or journalistic content.

---

## ✅ Summary Table

| Pattern Example                  | Score | Why This Score?                                 |
|----------------------------------|-------|--------------------------------------------------|
| `report shows`, `announces find` | 3     | News summary of scientific results               |
| `according to a study`          | 3     | Common in journalistic retellings                |
| `researchers say`               | 3     | External paraphrasing, not original research     |
| `published today`               | 3     | Time-sensitive news wording                      |
| `news release`                  | 3     | Indicates secondary source                       |
| `press release`                 | 3     | Confirms media format, not scientific document   |

---

## ⚙️ Purpose of These Scores

These patterns help identify and filter **non-research content**, ensuring that the heuristic pipeline analyzes only **primary scientific abstracts**, not:

- Media reports
- Blog posts
- Newsroom summaries
- Promotional material

This improves the accuracy of downstream analysis by **excluding irrelevant or secondhand sources** from RCT classification and issue detection pipelines.


---
---
# 📘 Regex Heuristic Scoring: Opinion Patterns

This document explains the scoring system for regex patterns used to identify **opinion-based or editorial content**. These patterns help detect when a passage reflects personal beliefs, recommendations, or subjective commentary — rather than objective research reporting.

---

## 🟥 Strong Opinion Indicators (Score: 3)

These phrases strongly suggest that the content is **opinionated**, **editorial**, or **not scientific reporting**. They often appear in:
- Editorials
- Opinion pieces
- Blog posts
- Letters to the editor

### ✅ Patterns and Explanations:

- `\bi\s+(?:believe|think|argue)\b`  
  Matches: `I believe`, `I think`, `I argue`  
  **Score**: 3  
  **Why**: First-person opinion — rarely used in formal scientific writing.

- `\bin\s+my\s+opinion\b`  
  Matches: `in my opinion`  
  **Score**: 3  
  **Why**: Classic phrase for expressing personal views.

- `\bwe\s+(?:should|must|need\s+to)\b`  
  Matches: `we should`, `we must`, `we need to`  
  **Score**: 3  
  **Why**: Prescriptive tone — calls for action, typical in advocacy or commentary.

- `\beditorial\b`  
  Matches: `editorial`  
  **Score**: 3  
  **Why**: Denotes an explicitly non-research format.

- `\bcommentary\b`  
  Matches: `commentary`  
  **Score**: 3  
  **Why**: Usually refers to subjective discussion of research or policy.

---

## 🟧 Moderate Opinion Indicators (Score: 2)

These indicate **suggestive or subjective framing**, but may also appear in reflective or theoretical research sections.

### ✅ Patterns and Explanations:

- `\bperspective\b`  
  Matches: `perspective`  
  **Score**: 2  
  **Why**: Can indicate viewpoint-based narrative.

- `\bviewpoint\b`  
  Matches: `viewpoint`  
  **Score**: 2  
  **Why**: Indicates interpretation or opinion-based framing.

- `\bit\s+is\s+(?:important|crucial|essential)\s+that\b`  
  Matches: `it is important that`, `it is essential that`  
  **Score**: 2  
  **Why**: Suggests evaluative, subjective language.

- `\bwe\s+advocate\b`  
  Matches: `we advocate`  
  **Score**: 2  
  **Why**: Suggests recommendation or policy proposal, not objective analysis.

---

## ✅ Summary Table

| Pattern Example              | Score | Why This Score?                                    |
|------------------------------|-------|-----------------------------------------------------|
| `I believe`, `we must`       | 3     | Strongly subjective and action-oriented             |
| `in my opinion`, `editorial` | 3     | Explicit personal or editorial expression           |
| `perspective`, `viewpoint`   | 2     | Indicates interpretive or reflective discussion     |
| `it is important that`       | 2     | Evaluative or suggestive tone                       |
| `we advocate`                | 2     | Opinion/recommendation phrasing                    |

---

## ⚙️ Purpose of These Scores

These patterns help distinguish **subjective or opinion-driven text** from **objective, data-based research**. This is essential for filtering out:
- Editorials
- Policy commentaries
- Theoretical perspectives

…from the pipeline, especially when the system is expected to process **scientific abstracts** and **trial reports** only.



---
---
# 📘 Regex Heuristic Scoring: Review Patterns

This document explains the scoring of regex patterns used to identify **review articles**, such as systematic reviews, meta-analyses, and literature summaries. These patterns help distinguish **secondary research** from original, primary studies like RCTs.

---

## 🟥 Strong Review Indicators (Score: 3)

These phrases are highly specific to **review-based articles**. If detected, they strongly suggest the text is **not a primary study**, but rather a secondary synthesis of multiple studies.

### ✅ Patterns and Explanations:

- `\bsystematic\s+review\b`  
  Matches: `systematic review`  
  **Score**: 3  
  **Why**: Clearly identifies the abstract as a systematic review.

- `\bmeta-analysis\b`  
  Matches: `meta-analysis`  
  **Score**: 3  
  **Why**: Indicates statistical pooling of multiple studies — not original research.

- `\bthis\s+review\b`  
  Matches: `this review`  
  **Score**: 3  
  **Why**: Shows that the article is not presenting new trial data.

- `\bliterature\s+review\b`  
  Matches: `literature review`  
  **Score**: 3  
  **Why**: Another direct signal that the abstract is summarizing others' work.

---

## 🟧 Moderate Review Indicators (Score: 2)

These are more subtle references to literature reviews or background discussions. They may appear in introductions or discussion sections of both primary and secondary research, so they carry **moderate weight**.

### ✅ Patterns and Explanations:

- `\brecent\s+studies?\s+have\s+shown\b`  
  Matches: `recent studies have shown`  
  **Score**: 2  
  **Why**: Introduces a synthesis of prior findings.

- `\bseveral\s+studies?\s+(?:have\s+)?(?:shown|demonstrated)\b`  
  Matches: `several studies have shown`, `several studies demonstrated`  
  **Score**: 2  
  **Why**: Typical summary language in reviews or intro sections.

- `\bmultiple\s+studies?\b`  
  Matches: `multiple studies`  
  **Score**: 2  
  **Why**: Indicates evidence is being pooled or summarized.

- `\bsummary\s+of\s+(?:evidence|literature)\b`  
  Matches: `summary of evidence`, `summary of literature`  
  **Score**: 2  
  **Why**: Suggests synthesis — classic of review articles.

- `\boverview\s+of\b`  
  Matches: `overview of`  
  **Score**: 2  
  **Why**: Indicates a broad discussion, often in reviews.

---

## ✅ Summary Table

| Pattern Example                  | Score | Why This Score?                                     |
|----------------------------------|-------|-----------------------------------------------------|
| `systematic review`             | 3     | Directly denotes review article                     |
| `meta-analysis`                 | 3     | Aggregates data from multiple studies               |
| `literature review`             | 3     | Signals secondary research                          |
| `several studies have shown`    | 2     | Summary of existing evidence                        |
| `summary of evidence`           | 2     | Indicates synthesis without new experimentation     |
| `overview of`                   | 2     | Suggests a narrative or thematic review             |

---

## ⚙️ Purpose of These Scores

These patterns help identify **secondary literature**, allowing your pipeline to:

- **Exclude non-primary studies** from certain analyses (e.g., RCT-specific scoring)
- **Differentiate** between original trials and literature reviews
- **Improve precision** when classifying medical abstracts

This is especially useful in filtering out articles that don't contribute original trial data.

---
---
# 📘 Regex Heuristic Scoring: Case Report Patterns

This document explains the scoring of regex patterns used to detect **case reports** and **case series**, which are descriptive studies focusing on one or a few patients. These patterns help filter out anecdotal or observational narratives that are **not randomized clinical trials (RCTs)**.

---

## 🟥 Strong Case Report Indicators (Score: 3)

These patterns clearly indicate that the abstract is a **case report or case series** — both are forms of observational research focused on individual patient experiences.

### ✅ Patterns and Explanations:

- `\bcase\s+report\b`  
  Matches: `case report`  
  **Score**: 3  
  **Why**: Directly identifies the article as a case report.

- `\bcase\s+series\b`  
  Matches: `case series`  
  **Score**: 3  
  **Why**: Indicates a descriptive collection of cases — not a controlled study.

- `\bwe\s+(?:present|report)\s+(?:a\s+)?case\b`  
  Matches: `we present a case`, `we report a case`  
  **Score**: 3  
  **Why**: First-person language used in anecdotal or clinical case write-ups.

---

## 🟧 Moderate Case Indicators (Score: 2)

These patterns suggest the presence of **a clinical narrative** or patient description but are not as conclusive as the above. They can also appear in introduction or background sections of other study types.

### ✅ Patterns and Explanations:

- `\ba\s+\d+-year-old\s+(?:man|woman|male|female|patient)\b`  
  Matches: `a 45-year-old male`, `a 23-year-old patient`  
  **Score**: 2  
  **Why**: Common way to open a case report with a patient’s age and gender.

- `\bpatient\s+presented\s+with\b`  
  Matches: `patient presented with chest pain`  
  **Score**: 2  
  **Why**: Describes patient symptom onset — typical in clinical reporting.

- `\bclinical\s+case\b`  
  Matches: `clinical case`  
  **Score**: 2  
  **Why**: Describes a detailed patient scenario — often not experimental.

- `\bcase\s+(?:study|description)\b`  
  Matches: `case study`, `case description`  
  **Score**: 2  
  **Why**: Refers to descriptive narratives, not structured trials.

---

## ✅ Summary Table

| Pattern Example                  | Score | Why This Score?                                  |
|----------------------------------|-------|---------------------------------------------------|
| `case report`                   | 3     | Explicit signal of case-based, non-RCT research   |
| `we present a case`             | 3     | Common phrasing in clinical anecdotal literature  |
| `a 45-year-old patient`         | 2     | Standard intro to a patient-based narrative       |
| `clinical case`, `case study`   | 2     | General terms for descriptive patient accounts    |

---

## ⚙️ Purpose of These Scores

These patterns help the pipeline:

- Detect **case reports and case series**, which are **not primary RCTs**
- Avoid misclassifying descriptive observational studies as trials
- Maintain relevance filtering, especially when building a dataset of RCT abstracts

Case reports provide valuable clinical insights but do not offer the methodological rigor or generalizability of randomized studies.

