# PeerGPT-RCT

PeerGPT-RCT is a hybrid peer-review system for evaluating clinical research abstracts, with an initial focus on Randomized Controlled Trials (RCTs).

It combines two models:
- **Heuristics model** (rule-based): Flags issues based on transparent, manually defined criteria.
- **LLM model** (language-based): Uses prompt engineering to simulate expert-style review.

## 🔍 System Pipeline
1. Relevance check: Is it a research abstract?
2. Study design classification: RCT or not?
3. Summary generation (LLM only)
4. Major and minor issue detection
5. Final peer-review assessment

## 📁 Project Structure
See `/docs` for the system design, heuristics table, and prompt templates.



## 💻 Usage
Coming soon...

## 🧠 Authors








