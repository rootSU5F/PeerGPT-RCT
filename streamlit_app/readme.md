# Streamlit App - PeerGPT-RCT

The **Streamlit App** is the user-friendly web interface for PeerGPT-RCT, providing an intuitive way to access the comprehensive hybrid peer-review system for evaluating clinical research abstracts.

## �� Live Application

**The PeerGPT-RCT application is now live and available for use!**

🌐 **Access the application**: [http://138.197.30.1/](http://138.197.30.1/)

You can now:
- **Paste RCT abstracts** directly into the web interface
- **Run comprehensive quality assessments** using the hybrid analysis system
- **Get instant feedback** on major and minor methodological issues
- **Compare results** across heuristic, hybrid, and AI analysis approaches

## 🎯 Overview

The Streamlit app provides a modern, responsive web interface that orchestrates all three analysis approaches:

1. **�� Heuristic Analysis** (Primary) - Rule-based systematic quality checks
2. **⚡ Hybrid Analysis** (Enhanced) - AI + heuristic fusion for improved accuracy
3. **🤖 AI Analysis** (Supplementary) - Pure AI clinical reasoning

## 🏗️ Architecture

```
streamlit_app/
└── app.py                      # Main Streamlit application
```

The app integrates with:
- **heuristics_model/**: Rule-based analysis modules
- **llm_pipeline/**: AI-powered analysis
- **hybrid_pipeline/**: AI + heuristic fusion
- **runner.py**: Orchestration and execution engine

## 💻 User Interface Features

### �� Modern Design
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Professional Styling**: Medical-grade interface design
- **Intuitive Navigation**: Clear, logical user flow
- **Visual Feedback**: Progress indicators and status updates

### �� Input Options
- **Text Area**: Direct abstract pasting
- **Sample Abstracts**: Pre-loaded examples for testing
- **Real-time Validation**: Input quality checking
- **Character Count**: Text length monitoring

### 🔧 Analysis Configuration
- **Heuristic Analysis**: Always enabled (baseline)
- **Hybrid Analysis**: Recommended for enhanced accuracy
- **AI Analysis**: Optional for additional clinical reasoning
- **Pipeline Selection**: Choose specific models to run

### 📊 Results Display
- **Gateway Checks**: Content validation and RCT classification
- **Quality Metrics**: Major and minor issues summary
- **Detailed Cards**: Individual assessment results
- **Clinical Implications**: Impact on study validity
- **Multi-Modal Comparison**: Cross-approach analysis

## 🚀 Key Features

### Gateway Validation
1. **Content Relevance Check**: Ensures submitted text is a research abstract
2. **Study Design Classification**: Confirms if it's a randomized controlled trial
3. **Quality Assessment**: Comprehensive multi-modal analysis

### Analysis Categories

#### 🚨 Major Issues (Critical Methodological Concerns)
- **Blinding Assessment**: Double/triple-blind vs open-label studies
- **Placebo Control**: Adequate control group design
- **Sample Size**: Statistical power adequacy (≥1000 participants recommended)
- **Composite Outcomes**: Single vs composite primary outcomes
- **Primary Outcome Events**: Adequate event counts (≥30 events per group)

#### 📋 Minor Issues (Additional Considerations)
- **Age Generalizability**: Appropriate age representation
- **Sex Generalizability**: Balanced sex distribution
- **Follow-up Duration**: Adequate outcome assessment timeline
- **Funding Disclosure**: Transparency in funding sources
- **ITT Analysis**: Intention-to-treat analysis approach
- **Multi-center Status**: Single vs multi-center design
- **Primary Timeline**: Appropriate outcome assessment timing

### Enhanced Analysis Options

#### ⚡ Hybrid Analysis (Recommended)
- **Enhanced Accuracy**: AI + rule-based fusion
- **Context Awareness**: Clinical reasoning integration
- **Bias Detection**: Comprehensive risk assessment
- **Confidence Scoring**: Reliability indicators

#### 🤖 AI Analysis (Optional)
- **Clinical Evaluation**: Expert-style peer review
- **Content Classification**: Intelligent content validation
- **Study Summarization**: Key findings extraction
- **Risk Assessment**: Comprehensive bias detection

## 📱 User Experience

### Step-by-Step Process
1. **Input Abstract**: Paste or select sample abstract
2. **Configure Analysis**: Choose analysis options
3. **Run Assessment**: Click "Analyze Quality"
4. **Review Results**: Examine comprehensive findings
5. **Compare Approaches**: Multi-modal analysis comparison

### Sample Abstracts
The app includes pre-loaded sample abstracts for testing:
- **�� Sample RCT**: High-quality randomized controlled trial
- **❌ Non-RCT Study**: Observational study example
- **❌ News Article**: Non-research content example
- **✏️ Enter Your Own**: Custom text input

### Results Visualization
- **Summary Metrics**: Quick overview of issues found
- **Detailed Cards**: Individual assessment results with explanations
- **Clinical Implications**: Impact on study validity and clinical practice
- **Multi-Modal Comparison**: Side-by-side analysis approaches

## 🔧 Technical Implementation

### Core Technologies
- **Streamlit 1.35.0**: Web application framework
- **Python 3.10**: Backend processing
- **spaCy 3.7.2**: NLP processing
- **LangChain**: LLM integration
- **Groq API**: High-performance LLM access

### Advanced Features
- **Concurrent Processing**: Multi-threaded analysis execution
- **Error Handling**: Robust failure recovery and user feedback
- **Memory Optimization**: Efficient large document processing
- **Real-time Updates**: Live progress indicators and status updates

### Performance Optimization
- **Asynchronous Execution**: Non-blocking analysis operations
- **Caching**: Efficient result storage and retrieval
- **Resource Management**: Optimized memory and CPU usage
- **Scalable Architecture**: Container-ready deployment

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

## 🚀 Deployment

### Live Application
The application is currently deployed and accessible at:
**�� [http://138.197.30.1/](http://138.197.30.1/)**

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

## 📊 Performance Metrics

### Analysis Speed
- **Heuristic Analysis**: ~1-2 seconds
- **Hybrid Analysis**: ~3-5 seconds
- **Full AI Analysis**: ~5-10 seconds
- **Concurrent Execution**: All models run in parallel

### User Experience
- **Responsive Design**: Works on all device sizes
- **Real-time Feedback**: Live progress updates
- **Error Recovery**: Graceful handling of issues
- **Intuitive Interface**: Easy-to-use design

## 🔍 Quality Assurance

### User Interface Testing
- **Cross-browser Compatibility**: Works on Chrome, Firefox, Safari, Edge
- **Mobile Responsiveness**: Optimized for tablet and mobile use
- **Accessibility**: Screen reader compatible
- **Performance Testing**: Optimized for speed and reliability

### Error Handling
- **Input Validation**: Robust text processing
- **API Failures**: Graceful degradation
- **Network Issues**: Retry mechanisms
- **User Feedback**: Clear error messages

## 🎯 Best Practices

### User Experience
- **Clear Instructions**: Step-by-step guidance
- **Visual Feedback**: Progress indicators and status updates
- **Error Prevention**: Input validation and sanitization
- **Help Documentation**: Inline help and tooltips

### Performance
- **Optimized Loading**: Fast initial page load
- **Efficient Processing**: Streamlined analysis pipeline
- **Resource Management**: Memory and CPU optimization
- **Scalable Architecture**: Ready for high-traffic deployment

## �� Configuration

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

## 🎯 Clinical Applications

### Research Quality Assessment
- **Peer Review Support**: Systematic quality evaluation
- **Evidence Synthesis**: Meta-analysis quality weighting
- **Clinical Decision Support**: Treatment recommendation validation
- **Research Training**: Educational quality assessment tool

### Regulatory Compliance
- **CONSORT Guidelines**: Reporting standard compliance
- **FDA Requirements**: Regulatory submission support
- **Cochrane Standards**: Systematic review integration
- **Journal Requirements**: Publication quality assessment

---

**Streamlit App - PeerGPT-RCT**: Providing an intuitive, powerful web interface for comprehensive RCT quality assessment. Access the live application at [http://138.197.30.1/](http://138.197.30.1/).