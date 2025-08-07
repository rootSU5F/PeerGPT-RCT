Here’s a clear, comprehensive README for the `hybrid_pipeline` (Hybrid Models) in your PeerGPT-RCT project:

---

# Hybrid Pipeline (Hybrid Models) — PeerGPT-RCT

The **Hybrid Pipeline** in PeerGPT-RCT fuses the strengths of rule-based heuristics and AI-powered reasoning to deliver advanced, context-aware, and clinically relevant quality assessments for randomized controlled trial (RCT) abstracts.

## 🚀 What Are Hybrid Models?

Hybrid models combine:
- **Heuristic (Rule-Based) Analysis:** Transparent, systematic checks based on clinical research standards.
- **AI (LLM) Reasoning:** Flexible, context-aware evaluation using large language models.

This fusion enables:
- More accurate detection of subtle or complex methodological issues.
- Contextual understanding that adapts to diverse study designs.
- Reduced false positives/negatives compared to using heuristics or AI alone.

---

## 📁 Directory Structure

```
hybrid_pipeline/
├── MinorIssues.py        # Hybrid minor issues assessment (AI + heuristics)
└── RCTBiasDetector.py    # Hybrid bias detection and risk assessment
```

---

## 🧩 Main Components

### 1. `MinorIssues.py`
- **Purpose:** Provides a hybrid assessment of minor methodological issues in RCT abstracts.
- **How it works:** 
  - Runs both heuristic checks (from `heuristics_model/MinorIssues/`) and AI-based analysis.
  - Merges results to provide a more nuanced, context-aware evaluation.
- **Output:** 
  - Structured results showing both heuristic and AI findings.
  - Combined assessment with confidence scores and clinical implications.

### 2. `RCTBiasDetector.py`
- **Purpose:** Delivers a comprehensive hybrid evaluation of bias and risk in RCTs.
- **How it works:**
  - Integrates heuristic bias detection (e.g., blinding, allocation, reporting) with AI-driven bias assessment.
  - Produces a detailed risk profile, including explanations and confidence levels.
- **Output:**
  - Bias assessment for multiple domains (e.g., selection, performance, detection, reporting).
  - Overall risk rating (low, moderate, high) with supporting evidence.

---

## 🛠️ How Hybrid Models Work

1. **Input:** RCT abstract text.
2. **Heuristic Analysis:** Rule-based modules extract features and flag issues.
3. **AI Analysis:** LLM models interpret the abstract, simulating expert review.
4. **Fusion:** Results are merged, cross-validated, and synthesized.
5. **Output:** 
   - Detailed, multi-perspective report.
   - Confidence scores and clinical implications.
   - Recommendations for reviewers or researchers.

---

## 📝 Example Usage

```python
from hybrid_pipeline.MinorIssues import create_minor_issues_detector

detector = create_minor_issues_detector()
result = detector.detect_minor_issues(abstract_text)
print(result)
```

```python
from hybrid_pipeline.RCTBiasDetector import create_bias_detector

bias_detector = create_bias_detector()
bias_result = bias_detector.detect_bias(abstract_text)
print(bias_result)
```

---

## 🔍 Output Structure

- **heuristic_results:** Output from rule-based modules.
- **ai_results:** Output from AI/LLM models.
- **combined_assessment:** Synthesized summary, risk level, and confidence.
- **explanations:** Reasoning and clinical implications for each finding.

---

## ⚡ Why Use Hybrid Models?

- **Accuracy:** Reduces missed issues and false alarms.
- **Context Awareness:** Adapts to complex or unusual study designs.
- **Transparency:** Shows both systematic and AI-driven reasoning.
- **Clinical Relevance:** Provides actionable insights for peer reviewers and researchers.

---

## 🧑‍🔬 Clinical & Research Applications

- **Peer Review:** Enhanced, multi-perspective quality checks.
- **Evidence Synthesis:** More reliable study inclusion/exclusion.
- **Training:** Demonstrates both rule-based and expert-style reasoning.
- **Regulatory Compliance:** Supports robust, auditable quality assessment.

---

## 🛠️ Extending Hybrid Models

- Add new hybrid modules for other domains (e.g., major issues, reporting quality).
- Integrate additional AI models or update heuristics as standards evolve.
- Customize fusion logic for specific research needs.

---

**Hybrid Pipeline — PeerGPT-RCT:**  
Bringing together the best of rules and AI for next-generation clinical trial quality assessment.

---