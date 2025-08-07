# PeerGPT-RCT

PeerGPT-RCT is a comprehensive hybrid peer-review system for evaluating clinical research abstracts, with a primary focus on Randomized Controlled Trials (RCTs). The system combines rule-based heuristics with AI-powered analysis to provide thorough quality assessment of clinical trial abstracts.

## 🎯 System Overview

The system employs a **three-tier analysis approach** that provides multiple layers of quality assessment:

1. **🔍 Heuristic Analysis** (Primary) - Rule-based systematic quality checks using transparent, manually defined criteria
2. **⚡ Hybrid Analysis** (Enhanced) - AI + heuristic fusion for improved accuracy and clinical reasoning
3. **🤖 AI Analysis** (Supplementary) - Pure AI clinical reasoning for expert-style peer review

## 🔍 Analysis Pipeline

### Gateway Validation
1. **Relevance Check** - Determines if the text is a research abstract
2. **Study Design Classification** - Confirms if it's a randomized controlled trial
3. **Comprehensive Quality Assessment** - Multi-modal analysis

### Quality Assessment Categories

#### 🚨 Major Issues (Critical Methodological Concerns)
These are fundamental flaws that can significantly impact study validity:

- **Blinding Assessment** - Evaluates double/triple-blind vs open-label studies
- **Placebo Control** - Assesses adequacy of control group design
- **Sample Size** - Checks statistical power adequacy (≥1000 participants recommended)
- **Composite Outcomes** - Analyzes single vs composite primary outcomes
- **Primary Outcome Events** - Validates adequate event counts (≥30 events per group)

#### 📋 Minor Issues (Additional Considerations)
These affect generalizability and interpretation:

- **Age Generalizability** - Evaluates appropriate age representation
- **Sex Generalizability** - Assesses balanced sex distribution
- **Follow-up Duration** - Checks adequacy of outcome assessment timeline
- **Funding Disclosure** - Evaluates transparency in funding sources
- **ITT Analysis** - Assesses intention-to-treat analysis approach
- **Multi-center Status** - Evaluates single vs multi-center design
- **Primary Timeline** - Checks appropriateness of outcome assessment timing

## 🏗️ Architecture

```
PeerGPT-RCT/
├── heuristics_model/          # Rule-based analysis modules
│   ├── PreChecks/            # Gateway validation (relevance, RCT classification)
│   ├── MajorIssues/          # Critical quality checks (blinding, placebo, sample size, etc.)
│   └── MinorIssues/          # Additional considerations (generalizability, funding, etc.)
├── llm_pipeline/             # AI-powered analysis
│   ├── MajorIssues/          # AI clinical evaluation
│   ├── PreChecks/            # AI content classification
│   └── Summarizer/           # AI trial summarization
├── hybrid_pipeline/          # AI + heuristic fusion
│   ├── MinorIssues.py        # Enhanced minor issues assessment
│   └── RCTBiasDetector.py    # Comprehensive bias detection
├── streamlit_app/            # Web interface
└── docs/                     # System documentation and design specs
```

## 🌐 Live Application

**The PeerGPT-RCT application is now live and available for use!**

🌐 **Access the application**: [http://138.197.30.1/](http://138.197.30.1/)

## 💻 Usage

### Web Interface (Recommended)
1. **Visit** [http://138.197.30.1/](http://138.197.30.1/)
2. **Paste your RCT abstract** in the text area
3. **Select analysis options**:
   - Heuristic Analysis (always included)
   - Hybrid Analysis (recommended for enhanced accuracy)
   - AI Analysis (optional for additional clinical reasoning)
4. **Click "Analyze Quality"** for comprehensive assessment

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd PeerGPT-RCT

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app/app.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t peergpt-rct .
docker run -p 8501:8501 peergpt-rct
```

### Programmatic Usage
```python
from runner import run_all_heuristics

# Analyze an abstract
abstract = "Your RCT abstract text here..."
results = run_all_heuristics(abstract)

# Access results by category
major_issues = results["MajorIssues"]
minor_issues = results["MinorIssues"]
```

## ⚙️ Technical Approach

### Heuristic Analysis (Rule-Based)
The system uses **transparent, manually defined criteria** based on clinical research standards:

- **Pattern Matching**: Advanced regex patterns with spaCy NLP enhancement
- **Clinical Validation**: Expert-reviewed detection rules
- **Systematic Coverage**: Comprehensive quality checklist
- **Transparent Logic**: All rules are explicit and auditable

**Example Heuristic Rule**:
```python
# Sample size adequacy check
if sample_size < 1000:
    return "Inadequate sample size for reliable results"
elif sample_size < 500:
    return "Very small sample size - high risk of chance findings"
```

### Hybrid Analysis (AI + Heuristics)
Combines the **precision of rule-based systems** with the **flexibility of AI**:

- **Enhanced Accuracy**: AI reasoning + rule-based consistency
- **Context Awareness**: Clinical reasoning integration
- **Reduced False Positives**: Cross-validation between approaches
- **Balanced Assessment**: Systematic + flexible evaluation

### AI Analysis (Pure AI)
Provides **expert-style clinical reasoning**:

- **Clinical Evaluation**: Simulates expert peer review
- **Natural Language Understanding**: Context-aware interpretation
- **Flexible Reasoning**: Adapts to study-specific nuances
- **Comprehensive Assessment**: Holistic quality evaluation

## 📊 Analysis Results

### Heuristic Analysis Output
Each check returns structured results:

```python
{
    "code": 0,  # 0=good, 1-3=increasing concerns
    "message": "Adequate sample size for reliable results",
    "clinical_implications": "Good statistical power for detecting treatment effects",
    "reasoning": "Sample size of 1200 participants provides adequate power...",
    "features": {
        "sample_size": 1200,
        "power_analysis": True,
        "statistical_adequacy": "Good"
    }
}
```

### Hybrid Analysis Features
- **Enhanced Accuracy**: AI + rule-based fusion
- **Context Awareness**: Clinical reasoning integration
- **Bias Detection**: Comprehensive risk assessment
- **Confidence Scoring**: Reliability indicators

### AI Analysis Capabilities
- **Clinical Evaluation**: Expert-style peer review
- **Content Classification**: Intelligent content validation
- **Study Summarization**: Key findings extraction
- **Risk Assessment**: Comprehensive bias detection

## ⚙️ Configuration

### Environment Variables
```bash
# LLM API Configuration
GROQ_API_KEY=your_groq_api_key

# Optional: Custom model settings
LLM_MODEL=llama3-70b-8192
```

### Analysis Options
- **Heuristic Analysis**: Always enabled (baseline)
- **Hybrid Analysis**: Recommended for enhanced accuracy
- **AI Analysis**: Optional for additional clinical reasoning

## 📈 Performance Metrics

### Analysis Speed
- **Heuristic Analysis**: ~1-2 seconds
- **Hybrid Analysis**: ~3-5 seconds  
- **Full AI Analysis**: ~5-10 seconds
- **Concurrent Execution**: All models run in parallel

### Accuracy Features
- **Systematic Coverage**: Comprehensive quality checklist
- **Clinical Validation**: Expert-reviewed detection rules
- **Multi-Modal Verification**: Heuristic + AI cross-validation
- **Context Awareness**: Study-specific assessment criteria

## 🎯 Clinical Applications

### Quality Assessment
- **Peer Review Support**: Systematic quality evaluation
- **Evidence Synthesis**: Meta-analysis quality weighting
- **Clinical Decision Support**: Treatment recommendation validation
- **Research Training**: Educational quality assessment tool

### Regulatory Compliance
- **CONSORT Guidelines**: Reporting standard compliance
- **FDA Requirements**: Regulatory submission support
- **Cochrane Standards**: Systematic review integration
- **Journal Requirements**: Publication quality assessment

## 🔬 Technical Details

### Core Technologies
- **Python 3.10**: Primary development language
- **spaCy 3.7.2**: Advanced NLP processing
- **Streamlit 1.35.0**: Web interface framework
- **LangChain**: LLM integration framework
- **Groq API**: High-performance LLM access

### Advanced Features
- **Concurrent Processing**: Multi-threaded analysis
- **Error Handling**: Robust failure recovery
- **Memory Optimization**: Efficient large document processing
- **Scalable Architecture**: Container-ready deployment

## 📚 Documentation

### System Design
- **Major Issues Design**: `docs/MajorIssueDesign.md`
- **Minor Issues Design**: `docs/MinorIssuesDesign.md`
- **Relevance Scoring**: `docs/relevanceScoring.md`

### Sample Data
- **RCT Abstracts**: `docs/pubmed_abstracts.csv`
- **Labeled Data**: `docs/abstracts_labeled_input.json`

## 🤝 Contributing

### Development Setup
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-analysis`
3. **Add tests** for new functionality
4. **Submit pull request** with detailed description

### Code Standards
- **Type Hints**: Full type annotation
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for all modules
- **Style**: PEP 8 compliance

## 📄 License

[License information to be added]

## 👥 Authors

[Author information to be added]

## 🙏 Acknowledgments

- **Clinical Experts**: Medical professionals for validation
- **Research Community**: Open-source contributions
- **Academic Partners**: Research institution collaboration

---

**PeerGPT-RCT**: Advancing clinical research quality assessment through hybrid AI and systematic analysis. Access the live application at [http://138.197.30.1/](http://138.197.30.1/)