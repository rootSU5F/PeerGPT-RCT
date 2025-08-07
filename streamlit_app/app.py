import streamlit as st
import importlib
from typing import Dict, List, Any
import traceback
import json
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add your path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="RCT Quality Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LLM Pipeline Configuration
LLM_PIPELINES = {
    "MajorIssues": {
        "module_path": "llm_pipeline.MajorIssues.MajorIssues",
        "class_name": "MajorIssues",
        "factory_function": "create_major_issues_checker",
        "description": "AI evaluation of critical methodological issues",
        "icon": "🚨"
    },
    "LLMRelevance": {
        "module_path": "llm_pipeline.PreChecks.LLMRelevanceCheck",
        "class_name": "LLMRelevance", 
        "factory_function": "create_classifier",
        "description": "AI-powered content relevance assessment",
        "icon": "🔍"
    },
    "RCT_classification": {
        "module_path": "llm_pipeline.PreChecks.RCT_classification",
        "class_name": "RCTClassifier",
        "factory_function": "create_rct_classifier", 
        "description": "AI study design classification",
        "icon": "🔬"
    },
    "RCTSummarizer": {
        "module_path": "llm_pipeline.Summarizer.RCTSummarizer",
        "class_name": "RCTSummarizer",
        "factory_function": "create_summarizer",
        "description": "AI-powered trial summary generation",
        "icon": "📋"
    }
}

# HYBRID Pipeline Configuration - NEW
HYBRID_PIPELINES = {
    "MinorIssues": {
        "module_path": "hybrid_pipeline.MinorIssues",
        "class_name": "RCTMinorIssuesDetector",
        "factory_function": "create_minor_issues_detector",
        "description": "Hybrid AI + Heuristic minor issues assessment",
        "icon": "⚡"
    },
    "RCTBiasDetection": {
        "module_path": "hybrid_pipeline.RCTBiasDetector",
        "class_name": "RCTBiasDetector", 
        "factory_function": "create_bias_detector",
        "description": "Hybrid bias detection and risk assessment",
        "icon": "🎯"
    }
}

CHECKS = {
    "PreChecks": {
        "RCT_classification": "heuristics_model.PreChecks.RCT_classification",
        "relevance_check": "heuristics_model.PreChecks.relevance_check"
    },
    "MajorIssues": {
        "blindingCheck": "heuristics_model.MajorIssues.blindingCheck",
        "compositeCheck": "heuristics_model.MajorIssues.compositeChecker",
        "placeboCheck": "heuristics_model.MajorIssues.placeboCheck",
        "PrimaryOutCome": "heuristics_model.MajorIssues.PrimaryOutCome",
        "SampleSize": "heuristics_model.MajorIssues.SampleSize"
    },
    "MinorIssues": {
        "AgeGeneralization": "heuristics_model.MinorIssues.AgeGeneralizability",
        "FollowDuration": "heuristics_model.MinorIssues.FollowDuration",
        "Funding": "heuristics_model.MinorIssues.Funding",
        "ITT": "heuristics_model.MinorIssues.ITT",
        "MultiCenter": "heuristics_model.MinorIssues.MultiCenter",
        "PrimaryTimeLine": "heuristics_model.MinorIssues.PrimaryTimeLine",
        "SexGeneralization": "heuristics_model.MinorIssues.SexGeneralizability"
    }
}

# Comprehensive code mapping for all heuristic modules
CODE_MAPPINGS = {
    "blindingCheck": {
        0: {"icon": "✅", "type": "success", "text": "Properly Blinded", "description": "Double/triple-blind design - Good methodological design"},
        1: {"icon": "⚠️", "type": "warning", "text": "Not Blinded", "description": "Open-label or not blinded - Methodological issue"},
        2: {"icon": "⚠️", "type": "warning", "text": "Blinding Not Mentioned", "description": "No mention of blinding - Potential methodological issue"},
        3: {"icon": "⚠️", "type": "warning", "text": "Blinding Unclear", "description": "Blinding status unclear or inadequate"}
    },
    "compositeCheck": {
        0: {"icon": "✅", "type": "success", "text": "Single Clear Outcome", "description": "Single, clear primary outcome - Good design"},
        1: {"icon": "⚠️", "type": "warning", "text": "Composite Outcome Present", "description": "Composite outcome used - Interpretation difficulties"},
        2: {"icon": "⚠️", "type": "warning", "text": "Outcome Unclear", "description": "Primary outcome not clearly defined"}
    },
    "placeboCheck": {
        0: {"icon": "✅", "type": "success", "text": "Placebo Controlled", "description": "Placebo control mentioned - Good methodological design"},
        1: {"icon": "⚠️", "type": "warning", "text": "Drug vs Standard Care", "description": "Drug vs standard care - Methodological concern"},
        2: {"icon": "⚠️", "type": "warning", "text": "No Placebo Mentioned", "description": "No mention of placebo - Potential methodological issue"},
        3: {"icon": "⚠️", "type": "warning", "text": "Control Unclear", "description": "Control group not clearly defined"}
    },
    "PrimaryOutCome": {
        0: {"icon": "✅", "type": "success", "text": "Adequate Events", "description": "≥30 events in both groups - Good statistical power"},
        1: {"icon": "⚠️", "type": "warning", "text": "Insufficient Events", "description": "Insufficient events - May be underpowered"},
        2: {"icon": "⚠️", "type": "warning", "text": "Event Count Unclear", "description": "Event counts not clearly reported"}
    },
    "SampleSize": {
        0: {"icon": "✅", "type": "success", "text": "Adequate Sample Size", "description": "≥1000 participants - Good statistical power"},
        1: {"icon": "⚠️", "type": "warning", "text": "Moderate Sample Size", "description": "500-1000 participants - May limit effect detection"},
        2: {"icon": "⚠️", "type": "warning", "text": "Small Sample Size", "description": "Small sample size - Risk of chance findings"},
        3: {"icon": "⚠️", "type": "warning", "text": "Sample Size Not Reported", "description": "No sample size information provided"}
    },
    "AgeGeneralization": {
        0: {"icon": "✅", "type": "success", "text": "Good Age Generalizability", "description": "Appropriate age representation for target population"},
        1: {"icon": "⚠️", "type": "warning", "text": "Limited Age Generalizability", "description": "Age distribution may limit generalizability"},
        2: {"icon": "⚠️", "type": "warning", "text": "Age Not Reported", "description": "No age information provided"},
        3: {"icon": "⚠️", "type": "warning", "text": "Age Information Unclear", "description": "Age information present but unclear"}
    },
    "FollowDuration": {
        0: {"icon": "✅", "type": "success", "text": "Adequate Follow-up", "description": "Adequate follow-up duration for outcomes"},
        1: {"icon": "⚠️", "type": "warning", "text": "Short Follow-up", "description": "Follow-up duration may be insufficient"},
        2: {"icon": "⚠️", "type": "warning", "text": "Follow-up Not Provided", "description": "No follow-up duration mentioned"}
    },
    "Funding": {
        0: {"icon": "✅", "type": "success", "text": "Funding Disclosed", "description": "Funding source disclosed - Good transparency"},
        1: {"icon": "⚠️", "type": "warning", "text": "No Funding Information", "description": "No funding information provided"}
    },
    "ITT": {
        0: {"icon": "✅", "type": "success", "text": "ITT Analysis Mentioned", "description": "Intention-to-treat analysis specified - Good methodological approach"},
        1: {"icon": "⚠️", "type": "warning", "text": "Non-ITT Analysis", "description": "Non-ITT analysis approach used"},
        2: {"icon": "⚠️", "type": "warning", "text": "Analysis Not Mentioned", "description": "No analysis approach specified"},
        3: {"icon": "⚠️", "type": "warning", "text": "Analysis Approach Unclear", "description": "Analysis approach unclear"}
    },
    "MultiCenter": {
        0: {"icon": "✅", "type": "success", "text": "Multi-center Study", "description": "Multi-center design - Better generalizability"},
        1: {"icon": "⚠️", "type": "warning", "text": "Single Center", "description": "Single center study - Limited generalizability"},
        2: {"icon": "⚠️", "type": "warning", "text": "Center Status Not Mentioned", "description": "Study setting not clearly specified"}
    },
    "PrimaryTimeLine": {
        0: {"icon": "✅", "type": "success", "text": "Appropriate Timeline", "description": "Appropriate outcome assessment timeline"},
        1: {"icon": "⚠️", "type": "warning", "text": "Short Timeline", "description": "Timeline may be too short for outcomes"},
        2: {"icon": "⚠️", "type": "warning", "text": "Timeline Not Mentioned", "description": "Outcome assessment timeline not specified"}
    },
    "SexGeneralization": {
        0: {"icon": "✅", "type": "success", "text": "Adequate Sex Representation", "description": "Balanced sex representation"},
        1: {"icon": "⚠️", "type": "warning", "text": "Limited Sex Representation", "description": "Imbalanced sex distribution may limit generalizability"},
        2: {"icon": "⚠️", "type": "warning", "text": "Sex Not Reported", "description": "No sex/gender information provided"}
    },
    "RCT_classification": {
        0: {"icon": "✅", "type": "success", "text": "Randomized Controlled Trial", "description": "Confirmed RCT design - Good evidence level"},
        1: {"icon": "⚠️", "type": "warning", "text": "Non-RCT Study", "description": "Non-randomized study design"},
        2: {"icon": "⚠️", "type": "warning", "text": "Study Design Unclear", "description": "Study design cannot be clearly determined"}
    },
    "relevance_check": {
        0: {"icon": "✅", "type": "success", "text": "Research Abstract", "description": "Appropriate research abstract for analysis"},
        1: {"icon": "⚠️", "type": "warning", "text": "Non-Research Content", "description": "Content may not be suitable for clinical analysis"},
        2: {"icon": "⚠️", "type": "warning", "text": "Insufficient Content", "description": "Insufficient content for comprehensive analysis"},
        3: {"icon": "⚠️", "type": "warning", "text": "News Article", "description": "News article - not primary research"},
        4: {"icon": "⚠️", "type": "warning", "text": "Opinion/Editorial", "description": "Opinion or editorial content"},
        5: {"icon": "⚠️", "type": "warning", "text": "Review Summary", "description": "Review or summary - not primary research"},
        6: {"icon": "⚠️", "type": "warning", "text": "Case Report", "description": "Case report - limited evidence level"},
        7: {"icon": "⚠️", "type": "warning", "text": "Conference Abstract", "description": "Conference abstract - may have limited detail"},
        8: {"icon": "⚠️", "type": "warning", "text": "Book Chapter", "description": "Book chapter content"},
        9: {"icon": "⚠️", "type": "warning", "text": "Content Type Unclear", "description": "Content type could not be determined"}
    }
}

class LLMPipelineManager:
    """Manages multiple LLM pipelines and concurrent execution"""
    
    def __init__(self):
        self.llm_instances = {}
        self.initialization_errors = {}
        self._initialize_llm_pipelines()
    
    def _initialize_llm_pipelines(self):
        """Initialize all available LLM pipelines"""
        for pipeline_name, config in LLM_PIPELINES.items():
            try:
                module = importlib.import_module(config["module_path"])
                factory_func = getattr(module, config["factory_function"])
                instance = factory_func()
                
                if instance:
                    self.llm_instances[pipeline_name] = {
                        "instance": instance,
                        "config": config
                    }
                else:
                    self.initialization_errors[pipeline_name] = "Factory function returned None"
                    
            except Exception as e:
                self.initialization_errors[pipeline_name] = str(e)
    
    def run_single_llm_pipeline(self, pipeline_name: str, abstract: str) -> Dict[str, Any]:
        """Run a single LLM pipeline"""
        if pipeline_name not in self.llm_instances:
            return {
                "success": False,
                "error": f"Pipeline {pipeline_name} not available",
                "execution_time": 0,
                "result": None
            }
        
        try:
            start_time = time.time()
            instance = self.llm_instances[pipeline_name]["instance"]
            
            # Try pipeline-specific methods
            pipeline_methods = {
                "MajorIssues": ["identify_major_issues"],
                "LLMRelevance": ["is_research_abstract"],
                "RCT_classification": ["is_rct"],
                "RCTSummarizer": ["summarize_abstract"]
            }
            
            result = None
            if pipeline_name in pipeline_methods:
                for method_name in pipeline_methods[pipeline_name]:
                    if hasattr(instance, method_name):
                        method = getattr(instance, method_name)
                        result = method(abstract)
                        break
                else:
                    generic_methods = ['run_check', 'evaluate', 'analyze', 'process']
                    for method_name in generic_methods:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            result = method(abstract)
                            break
                    else:
                        return {
                            "success": False,
                            "error": f"No suitable method found for {pipeline_name}. Available methods: {[m for m in dir(instance) if not m.startswith('_')]}",
                            "execution_time": 0,
                            "result": None
                        }
            
            end_time = time.time()
            return {
                "success": True,
                "error": None,
                "execution_time": round(end_time - start_time, 2),
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "result": None
            }
    
    def run_all_llm_pipelines(self, abstract: str, selected_pipelines: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run multiple LLM pipelines concurrently"""
        pipelines_to_run = selected_pipelines if selected_pipelines else list(self.llm_instances.keys())
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(pipelines_to_run)) as executor:
            future_to_pipeline = {
                executor.submit(self.run_single_llm_pipeline, pipeline_name, abstract): pipeline_name
                for pipeline_name in pipelines_to_run
            }
            
            for future in as_completed(future_to_pipeline):
                pipeline_name = future_to_pipeline[future]
                try:
                    result = future.result()
                    results[pipeline_name] = result
                except Exception as e:
                    results[pipeline_name] = {
                        "success": False,
                        "error": f"Thread execution error: {str(e)}",
                        "execution_time": 0,
                        "result": None
                    }
        
        return results

class HybridPipelineManager:
    """Manages hybrid AI + heuristic pipelines"""
    
    def __init__(self):
        self.hybrid_instances = {}
        self.initialization_errors = {}
        self._initialize_hybrid_pipelines()
    
    def _initialize_hybrid_pipelines(self):
        """Initialize all available hybrid pipelines"""
        for pipeline_name, config in HYBRID_PIPELINES.items():
            try:
                module = importlib.import_module(config["module_path"])
                factory_func = getattr(module, config["factory_function"])
                instance = factory_func()
                
                if instance:
                    self.hybrid_instances[pipeline_name] = {
                        "instance": instance,
                        "config": config
                    }
                else:
                    self.initialization_errors[pipeline_name] = "Factory function returned None"
                    
            except Exception as e:
                self.initialization_errors[pipeline_name] = str(e)
    
    def run_single_hybrid_pipeline(self, pipeline_name: str, abstract: str) -> Dict[str, Any]:
        """Run a single hybrid pipeline"""
        if pipeline_name not in self.hybrid_instances:
            return {
                "success": False,
                "error": f"Hybrid pipeline {pipeline_name} not available",
                "execution_time": 0,
                "result": None
            }
        
        try:
            start_time = time.time()
            instance = self.hybrid_instances[pipeline_name]["instance"]
            
            # Try pipeline-specific methods first
            pipeline_methods = {
                "MinorIssues": ["detect_minor_issues"],
                "RCTBiasDetection": ["detect_bias"]
            }
            
            result = None
            if pipeline_name in pipeline_methods:
                for method_name in pipeline_methods[pipeline_name]:
                    if hasattr(instance, method_name):
                        method = getattr(instance, method_name)
                        result = method(abstract)
                        break
                else:
                    # Try generic methods
                    generic_methods = ['run_check', 'evaluate', 'analyze', 'process', 'assess']
                    for method_name in generic_methods:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            result = method(abstract)
                            break
                    else:
                        return {
                            "success": False,
                            "error": f"No suitable method found for {pipeline_name}. Available methods: {[m for m in dir(instance) if not m.startswith('_')]}",
                            "execution_time": 0,
                            "result": None
                        }
            
            end_time = time.time()
            return {
                "success": True,
                "error": None,
                "execution_time": round(end_time - start_time, 2),
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "result": None
            }
    
    def run_all_hybrid_pipelines(self, abstract: str, selected_pipelines: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run multiple hybrid pipelines concurrently"""
        pipelines_to_run = selected_pipelines if selected_pipelines else list(self.hybrid_instances.keys())
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(pipelines_to_run)) as executor:
            future_to_pipeline = {
                executor.submit(self.run_single_hybrid_pipeline, pipeline_name, abstract): pipeline_name
                for pipeline_name in pipelines_to_run
            }
            
            for future in as_completed(future_to_pipeline):
                pipeline_name = future_to_pipeline[future]
                try:
                    result = future.result()
                    results[pipeline_name] = result
                except Exception as e:
                    results[pipeline_name] = {
                        "success": False,
                        "error": f"Thread execution error: {str(e)}",
                        "execution_time": 0,
                        "result": None
                    }
        
        return results

def run_single_check(abstract: str, check_name: str, module_path: str) -> dict:
    """Run a single check module on the given abstract."""
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, "run_check"):
            result = module.run_check(abstract)
            return result
        else:
            return {"error": "Missing run_check()"}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

def run_all_heuristics(abstract: str) -> Dict[str, Dict[str, dict]]:
    """Run all registered heuristic modules on the given abstract."""
    results = {}

    for category, modules in CHECKS.items():
        category_results = {}
        for module_name, module_path in modules.items():
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, "run_check"):
                    result = module.run_check(abstract)
                    category_results[module_name] = result
                else:
                    category_results[module_name] = {"error": "Missing run_check()"}
            except Exception as e:
                category_results[module_name] = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }

        results[category] = category_results
    return results

def check_is_research_abstract(relevance_result):
    """Check if the text is a research abstract."""
    if has_error(relevance_result):
        return False, "❌ **Analysis Error:** Unable to determine if this is a research abstract due to technical issues."
    
    relevance_code = safe_get_attr(relevance_result, 'code')
    
    if hasattr(relevance_code, 'value'):
        relevance_code = relevance_code.value
    
    is_research_abstract = (relevance_code == 0)
    
    if not is_research_abstract:
        return False, "🚫 **Submitted text is not a research abstract.**\n\nThe analysis has been stopped. Please submit a research abstract for quality assessment."
    
    return True, None

def check_is_rct(rct_result):
    """Check if the text is an RCT."""
    if has_error(rct_result):
        return False, "❌ **Analysis Error:** Unable to classify study type due to technical issues."
    
    rct_code = safe_get_attr(rct_result, 'code')
    
    if hasattr(rct_code, 'value'):
        rct_code = rct_code.value
    
    is_rct = (rct_code == 0)
    
    if not is_rct:
        return False, "🚫 **Submitted text is not a randomized controlled trial abstract.**\n\nThe analysis has been stopped. Please submit an RCT abstract for quality assessment."
    
    return True, None

def get_severity_info(code_value, check_name):
    """Return icon, color, and severity text based on code value and check name"""
    if hasattr(code_value, 'value'):
        severity = code_value.value
    else:
        severity = code_value
    
    if check_name in CODE_MAPPINGS and severity in CODE_MAPPINGS[check_name]:
        mapping = CODE_MAPPINGS[check_name][severity]
        return mapping["icon"], mapping["type"], mapping["text"], mapping["description"]
    else:
        return "❓", "info", f"Code {severity}", "Unknown code value"

def safe_get_attr(obj, attr_name, default=None):
    """Safely get attribute from either dict or object"""
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    else:
        return getattr(obj, attr_name, default)

def has_error(result):
    """Check if result has error in both dict and object format"""
    if isinstance(result, dict):
        return "error" in result
    else:
        return hasattr(result, 'error') and getattr(result, 'error') is not None

def display_result_card(title, result, category_type, original_check_name):
    """Display individual result in a modern card format"""
    
    if has_error(result):
        error_msg = safe_get_attr(result, 'error', 'Unknown error')
        with st.container():
            st.error(f"❌ **{title}** - {error_msg}")
            with st.expander("🔍 View Error Details"):
                traceback_info = safe_get_attr(result, 'traceback', 'No traceback available')
                st.code(traceback_info)
        return
    
    try:
        code = safe_get_attr(result, 'code', 'Unknown')
        message = safe_get_attr(result, 'message', 'No message available')
    except Exception as e:
        st.error(f"Error processing {title}: {str(e)}")
        return
    
    icon, severity_type, severity_text, description = get_severity_info(code, original_check_name)
    
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {icon} {title}")
        
        with col2:
            if severity_type == "success":
                st.success(severity_text)
            elif severity_type == "warning":
                st.warning(severity_text)
            elif severity_type == "error":
                st.error(severity_text)
            else:
                st.info(severity_text)
        
        st.markdown(f"**{description}**")
        
        if message and message != description and message != 'No message available':
            st.markdown(f"*Additional details: {message}*")
        
        with st.expander("📋 View Details"):
            clinical_implications = safe_get_attr(result, 'clinical_implications')
            if clinical_implications:
                st.markdown("#### 🏥 Clinical Impact")
                st.markdown(clinical_implications)
                st.divider()
            
            reasoning = safe_get_attr(result, 'reasoning')
            if reasoning:
                st.markdown("#### 🧠 Analysis")
                if isinstance(reasoning, list):
                    for i, reason in enumerate(reasoning, 1):
                        st.markdown(f"{i}. {reason}")
                else:
                    st.markdown(reasoning)
                st.divider()
            
            features = safe_get_attr(result, 'features')
            if features:
                st.markdown("#### ⚙️ Technical Details")
                try:
                    if hasattr(features, '__dict__'):
                        feature_dict = features.__dict__
                        relevant_features = {k: v for k, v in feature_dict.items() 
                                           if v and k not in ['detected_patterns', 'extracted_info']}
                        if relevant_features:
                            for key, value in relevant_features.items():
                                if isinstance(value, bool):
                                    status = "✅ Yes" if value else "❌ No"
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {status}")
                                elif isinstance(value, (int, float)) and key.endswith('_score'):
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                                elif isinstance(value, list) and len(value) <= 5:
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                    elif isinstance(features, dict):
                        relevant_features = {k: v for k, v in features.items() 
                                           if v and k not in ['detected_patterns', 'extracted_info']}
                        if relevant_features:
                            st.json(relevant_features, expanded=False)
                except Exception as e:
                    st.caption(f"Could not display technical details: {str(e)}")
        
        st.markdown("<br>", unsafe_allow_html=True)

def display_gateway_failure(message):
    """Display gateway failure message with consistent styling"""
    st.markdown("---")
    
    st.markdown("""
    <div style="
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    ">
    """, unsafe_allow_html=True)
    
    st.warning(message)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 💡 What to do next:")
    
    st.markdown("""
    **For Research Abstracts:**
    - Make sure your text is from a peer-reviewed research study
    - Include methodology, results, and conclusions
    - Avoid news articles, opinion pieces, or review summaries
    
    **For RCT Abstracts:**
    - Ensure your abstract describes a randomized controlled trial
    - Look for keywords like "randomized", "controlled", "trial"
    - Include details about study design and participant allocation
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Try one of the sample abstracts from the sidebar to see how the tool works!")

def display_llm_pipeline_results(pipeline_name: str, result: Dict[str, Any], config: Dict[str, str]):
    """Display results from a single LLM pipeline"""
    
    icon = config.get("icon", "🤖")
    description = config.get("description", "AI Analysis")
    
    with st.expander(f"{icon} {pipeline_name} - {description}", expanded=True):
        
        if not result.get("success"):
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        execution_time = result.get("execution_time", 0)
        st.caption(f"⏱️ Completed in {execution_time}s")
        
        llm_result = result.get("result")
        
        if llm_result is None:
            st.warning("No result returned from pipeline")
            return
        
        if isinstance(llm_result, str):
            st.markdown("### Analysis Results")
            
            if any(keyword in llm_result.lower() for keyword in ['internal validity', 'external', 'generalizability']):
                sections = llm_result.split('\n\n')
                for section in sections:
                    if section.strip():
                        lines = section.strip().split('\n')
                        if len(lines) > 1 and len(lines[0]) < 100 and any(char in lines[0] for char in [':', '-', '.']):
                            header = lines[0].replace(':', '').replace('-', '').replace('.', '').strip()
                            content = '\n'.join(lines[1:])
                            
                            st.markdown(f"**{header}**")
                            st.markdown(content)
                            st.markdown("---")
                        else:
                            st.markdown(section)
            else:
                st.markdown(llm_result)
                
        elif isinstance(llm_result, bool):
            if pipeline_name in ["LLMRelevance", "RCT_classification"]:
                if llm_result:
                    st.success(f"✅ Positive classification")
                else:
                    st.warning(f"⚠️ Negative classification")
            else:
                st.info(f"Result: {llm_result}")
                
        elif isinstance(llm_result, dict):
            st.markdown("### Analysis Results")
            
            for key, value in llm_result.items():
                if key in ['success', 'error', 'execution_time']:
                    continue
                    
                if isinstance(value, bool):
                    status = "✅ Yes" if value else "❌ No"
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {status}")
                elif isinstance(value, (int, float)):
                    if key.endswith('_score') or key.endswith('_confidence'):
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                elif isinstance(value, str) and len(value) < 200:
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                elif isinstance(value, str):
                    st.markdown(f"**{key.replace('_', ' ').title()}:**")
                    st.markdown(value)
                elif isinstance(value, list) and len(value) <= 10:
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                else:
                    with st.expander(f"📄 {key.replace('_', ' ').title()}"):
                        if isinstance(value, (list, dict)):
                            st.json(value)
                        else:
                            st.text(str(value))
        
        elif isinstance(llm_result, list):
            st.markdown("### Analysis Results")
            for i, item in enumerate(llm_result, 1):
                st.markdown(f"{i}. {item}")
        
        else:
            st.markdown("### Analysis Results")
            st.text(str(llm_result))

def display_hybrid_pipeline_results(pipeline_name: str, result: Dict[str, Any], config: Dict[str, str]):
    """Display results from a single hybrid pipeline with enhanced formatting"""
    
    icon = config.get("icon", "⚡")
    description = config.get("description", "Hybrid Analysis")
    
    with st.expander(f"{icon} {pipeline_name} - {description}", expanded=True):
        
        if not result.get("success"):
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        execution_time = result.get("execution_time", 0)
        st.caption(f"⏱️ Completed in {execution_time}s")
        
        hybrid_result = result.get("result")
        
        if hybrid_result is None:
            st.warning("No result returned from hybrid pipeline")
            return
        
        # Handle different result formats
        if isinstance(hybrid_result, dict):
            # Check for structured hybrid results
            if 'heuristic_results' in hybrid_result and 'ai_results' in hybrid_result:
                st.markdown("### 🔄 Hybrid Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Heuristic Component")
                    heuristic_res = hybrid_result['heuristic_results']
                    if isinstance(heuristic_res, dict):
                        for key, value in heuristic_res.items():
                            if isinstance(value, bool):
                                status = "✅ Pass" if value else "❌ Fail"
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {status}")
                            elif isinstance(value, (int, float)):
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                            elif isinstance(value, str) and len(value) < 100:
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.text(str(heuristic_res))
                
                with col2:
                    st.markdown("#### 🤖 AI Component")
                    ai_res = hybrid_result['ai_results']
                    if isinstance(ai_res, dict):
                        for key, value in ai_res.items():
                            if isinstance(value, bool):
                                status = "✅ Yes" if value else "❌ No"
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {status}")
                            elif isinstance(value, (int, float)):
                                if key.endswith('_confidence') or key.endswith('_score'):
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                                else:
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                            elif isinstance(value, str) and len(value) < 100:
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.text(str(ai_res))
                
                # Combined assessment if available
                if 'combined_assessment' in hybrid_result:
                    st.markdown("#### 🎯 Combined Assessment")
                    combined = hybrid_result['combined_assessment']
                    if isinstance(combined, dict):
                        for key, value in combined.items():
                            if key == 'overall_risk':
                                if value.lower() in ['low', 'minimal']:
                                    st.success(f"**Overall Risk:** {value}")
                                elif value.lower() in ['medium', 'moderate']:
                                    st.warning(f"**Overall Risk:** {value}")
                                else:
                                    st.error(f"**Overall Risk:** {value}")
                            elif key == 'confidence':
                                st.metric("Confidence", f"{value:.1%}" if isinstance(value, float) else str(value))
                            elif isinstance(value, str):
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.markdown(combined)
                        
            elif 'bias_assessment' in hybrid_result or 'risk_scores' in hybrid_result:
                # Handle bias detection results
                st.markdown("### 🎯 Bias Detection Results")
                
                if 'bias_assessment' in hybrid_result:
                    bias_assess = hybrid_result['bias_assessment']
                    if isinstance(bias_assess, dict):
                        for bias_type, assessment in bias_assess.items():
                            if isinstance(assessment, dict) and 'risk' in assessment:
                                risk_level = assessment['risk'].lower()
                                if risk_level in ['low', 'minimal']:
                                    st.success(f"**{bias_type.replace('_', ' ').title()}:** {assessment['risk']}")
                                elif risk_level in ['moderate', 'medium']:
                                    st.warning(f"**{bias_type.replace('_', ' ').title()}:** {assessment['risk']}")
                                else:
                                    st.error(f"**{bias_type.replace('_', ' ').title()}:** {assessment['risk']}")
                                
                                if 'explanation' in assessment:
                                    with st.expander(f"Details - {bias_type.replace('_', ' ').title()}"):
                                        st.markdown(assessment['explanation'])
                            else:
                                st.markdown(f"**{bias_type.replace('_', ' ').title()}:** {assessment}")
                
                if 'risk_scores' in hybrid_result:
                    st.markdown("#### 📊 Risk Scores")
                    risk_scores = hybrid_result['risk_scores']
                    if isinstance(risk_scores, dict):
                        cols = st.columns(len(risk_scores))
                        for i, (risk_type, score) in enumerate(risk_scores.items()):
                            with cols[i]:
                                st.metric(risk_type.replace('_', ' ').title(), f"{score:.2f}")
                                
            else:
                # Generic dict handling
                st.markdown("### Analysis Results")
                for key, value in hybrid_result.items():
                    if key in ['success', 'error', 'execution_time']:
                        continue
                        
                    if isinstance(value, bool):
                        status = "✅ Yes" if value else "❌ No"
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {status}")
                    elif isinstance(value, (int, float)):
                        if key.endswith('_score') or key.endswith('_confidence'):
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    elif isinstance(value, str) and len(value) < 200:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    elif isinstance(value, str):
                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        st.markdown(value)
                    elif isinstance(value, list) and len(value) <= 10:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                    else:
                        with st.expander(f"📄 {key.replace('_', ' ').title()}"):
                            if isinstance(value, (list, dict)):
                                st.json(value)
                            else:
                                st.text(str(value))
                                
        elif isinstance(hybrid_result, str):
            st.markdown("### Analysis Results")
            st.markdown(hybrid_result)
        
        elif isinstance(hybrid_result, list):
            st.markdown("### Analysis Results")
            for i, item in enumerate(hybrid_result, 1):
                st.markdown(f"{i}. {item}")
        
        else:
            st.markdown("### Analysis Results")
            st.text(str(hybrid_result))

def display_all_llm_results(llm_results: Dict[str, Dict[str, Any]]):
    """Display results from all LLM pipelines"""
    
    if not llm_results or "error" in llm_results:
        st.error(f"❌ LLM Analysis Failed: {llm_results.get('error', 'Unknown error')}")
        return
    
    st.markdown('<div class="category-header">🤖 AI Clinical Analysis</div>', unsafe_allow_html=True)
    
    total_pipelines = len(llm_results)
    successful_pipelines = sum(1 for result in llm_results.values() if result.get("success"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Pipelines", total_pipelines)
    with col2:
        st.metric("Successful", successful_pipelines)
    with col3:
        avg_time = sum(result.get("execution_time", 0) for result in llm_results.values()) / max(total_pipelines, 1)
        st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Display each pipeline's results
    for pipeline_name, result in llm_results.items():
        config = LLM_PIPELINES.get(pipeline_name, {})
        display_llm_pipeline_results(pipeline_name, result, config)

def display_all_hybrid_results(hybrid_results: Dict[str, Dict[str, Any]]):
    """Display results from all hybrid pipelines"""
    
    if not hybrid_results or "error" in hybrid_results:
        st.error(f"❌ Hybrid Analysis Failed: {hybrid_results.get('error', 'Unknown error')}")
        return
    
    st.markdown('<div class="category-header">⚡ Hybrid AI + Heuristic Analysis</div>', unsafe_allow_html=True)
    
    total_pipelines = len(hybrid_results)
    successful_pipelines = sum(1 for result in hybrid_results.values() if result.get("success"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hybrid Pipelines", total_pipelines)
    with col2:
        st.metric("Successful", successful_pipelines)
    with col3:
        avg_time = sum(result.get("execution_time", 0) for result in hybrid_results.values()) / max(total_pipelines, 1)
        st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Display each pipeline's results
    for pipeline_name, result in hybrid_results.items():
        config = HYBRID_PIPELINES.get(pipeline_name, {})
        display_hybrid_pipeline_results(pipeline_name, result, config)

def run_comprehensive_analysis(abstract: str, selected_llm_pipelines: List[str], selected_hybrid_pipelines: List[str]) -> Dict[str, Any]:
    """Run heuristics first, then LLM and hybrid analysis concurrently"""
    
    results = {
        "heuristics": None,
        "llm_pipelines": {},
        "hybrid_pipelines": {},
        "execution_times": {}
    }
    
    # Run heuristics first (sequential)
    heuristic_start = time.time()
    results["heuristics"] = run_all_heuristics(abstract)
    results["execution_times"]["heuristics"] = round(time.time() - heuristic_start, 2)
    
    # Then run LLM and Hybrid pipelines concurrently
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit LLM pipelines
        if selected_llm_pipelines:
            llm_manager = LLMPipelineManager()
            llm_future = executor.submit(llm_manager.run_all_llm_pipelines, abstract, selected_llm_pipelines)
            futures.append(("llm", llm_future))
        
        # Submit Hybrid pipelines
        if selected_hybrid_pipelines:
            hybrid_manager = HybridPipelineManager()
            hybrid_future = executor.submit(hybrid_manager.run_all_hybrid_pipelines, abstract, selected_hybrid_pipelines)
            futures.append(("hybrid", hybrid_future))
        
        # Collect results
        for pipeline_type, future in futures:
            try:
                if pipeline_type == "llm":
                    llm_start = time.time()
                    results["llm_pipelines"] = future.result()
                    results["execution_times"]["llm_total"] = round(time.time() - llm_start, 2)
                elif pipeline_type == "hybrid":
                    hybrid_start = time.time()
                    results["hybrid_pipelines"] = future.result()
                    results["execution_times"]["hybrid_total"] = round(time.time() - hybrid_start, 2)
            except Exception as e:
                if pipeline_type == "llm":
                    results["llm_pipelines"] = {"error": str(e)}
                elif pipeline_type == "hybrid":
                    results["hybrid_pipelines"] = {"error": str(e)}
    
    return results

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stContainer > div {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .hybrid-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🩺 RCT Quality Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive quality assessment for clinical trial abstracts</p>', unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("## 📝 Input Your Abstract")
        
        # Analysis options
        st.markdown("### 🔧 Analysis Options")
        
        # Always run heuristics (primary analysis)
        st.markdown("📊 **Heuristic Analysis** - Always included")
        st.caption("Rule-based systematic quality checks")
        
        # Optional AI analysis
        run_llm_analysis = st.checkbox(
            "🤖 Add AI Analysis", 
            value=False,
            help="Run additional AI-powered clinical evaluation"
        )
        
        # NEW: Optional Hybrid analysis
        run_hybrid_analysis = st.checkbox(
            "⚡ Add Hybrid Analysis", 
            value=True,
            help="Run hybrid AI + heuristic models for enhanced assessment"
        )
        
        # LLM pipeline selection (only show if LLM analysis enabled)
        selected_llm_pipelines = []
        if run_llm_analysis:
            try:
                temp_manager = LLMPipelineManager()
                available_pipelines = list(temp_manager.llm_instances.keys())
                
                if available_pipelines:
                    st.markdown("#### Select AI Models:")
                    selected_llm_pipelines = st.multiselect(
                        "Choose AI pipelines:",
                        available_pipelines,
                        default=available_pipelines,
                        help="AI models to run concurrently",
                        format_func=lambda x: f"{LLM_PIPELINES.get(x, {}).get('icon', '🤖')} {x}"
                    )
                    
                    if temp_manager.initialization_errors:
                        with st.expander("⚠️ AI Pipeline Issues"):
                            for pipeline, error in temp_manager.initialization_errors.items():
                                st.error(f"{pipeline}: {error}")
                else:
                    st.warning("⚠️ No AI pipelines available")
                    run_llm_analysis = False
                    
            except Exception as e:
                st.error(f"Could not initialize AI pipelines: {str(e)}")
                run_llm_analysis = False
        
        # NEW: Hybrid pipeline selection
        selected_hybrid_pipelines = []
        if run_hybrid_analysis:
            try:
                temp_hybrid_manager = HybridPipelineManager()
                available_hybrid_pipelines = list(temp_hybrid_manager.hybrid_instances.keys())
                
                if available_hybrid_pipelines:
                    st.markdown("#### Select Hybrid Models:")
                    selected_hybrid_pipelines = st.multiselect(
                        "Choose hybrid pipelines:",
                        available_hybrid_pipelines,
                        default=available_hybrid_pipelines,
                        help="Hybrid AI + heuristic models for comprehensive assessment",
                        format_func=lambda x: f"{HYBRID_PIPELINES.get(x, {}).get('icon', '⚡')} {x}"
                    )
                    
                    if temp_hybrid_manager.initialization_errors:
                        with st.expander("⚠️ Hybrid Pipeline Issues"):
                            for pipeline, error in temp_hybrid_manager.initialization_errors.items():
                                st.error(f"{pipeline}: {error}")
                else:
                    st.warning("⚠️ No hybrid pipelines available")
                    run_hybrid_analysis = False
                    
            except Exception as e:
                st.error(f"Could not initialize hybrid pipelines: {str(e)}")
                run_hybrid_analysis = False
        
        st.markdown("---")
        
        # Sample abstracts for testing
        sample_abstracts = {
            "🔬 Sample RCT": """Abstract
Background: Among infants with isolated cleft palate, whether primary surgery at 6 months of age is more beneficial than surgery at 12 months of age with respect to speech outcomes, hearing outcomes, dentofacial development, and safety is unknown.

Methods: We randomly assigned infants with nonsyndromic isolated cleft palate, in a 1:1 ratio, to undergo standardized primary surgery at 6 months of age (6-month group) or at 12 months of age (12-month group) for closure of the cleft. Standardized assessments of quality-checked video and audio recordings at 1, 3, and 5 years of age were performed independently by speech and language therapists who were unaware of the trial-group assignments. The primary outcome was velopharyngeal insufficiency at 5 years of age, defined as a velopharyngeal composite summary score of at least 4 (scores range from 0 to 6, with higher scores indicating greater severity). Secondary outcomes included speech development, postoperative complications, hearing sensitivity, dentofacial development, and growth.

Results: We randomly assigned 558 infants at 23 centers across Europe and South America to undergo surgery at 6 months of age (281 infants) or at 12 months of age (277 infants). Speech recordings from 235 infants (83.6%) in the 6-month group and 226 (81.6%) in the 12-month group were analyzable. Insufficient velopharyngeal function at 5 years of age was observed in 21 of 235 infants (8.9%) in the 6-month group as compared with 34 of 226 (15.0%) in the 12-month group (risk ratio, 0.59; 95% confidence interval, 0.36 to 0.99; P = 0.04). Postoperative complications were infrequent and similar in the 6-month and 12-month groups. Four serious adverse events were reported (three in the 6-month group and one in the 12-month group) and had resolved at follow-up.

Conclusions: Medically fit infants who underwent primary surgery for isolated cleft palate in adequately resourced settings at 6 months of age were less likely to have velopharyngeal insufficiency at the age of 5 years than those who had surgery at 12 months of age. (Funded by the National Institute of Dental and Craniofacial Research; TOPS ClinicalTrials.gov number, NCT00993551.).

Copyright © 2023 Massachusetts Medical Society.

PubMed Disclaimer

""",
            "❌ Non-RCT Study": "This observational cohort study followed 500 patients with diabetes for 5 years to assess cardiovascular outcomes. Patients were not randomized but selected based on their current treatment regimen.",
            "❌ News Article": "A new drug for treating cancer has shown promising results in early trials, according to researchers at University Hospital. The breakthrough could potentially help thousands of patients worldwide.",
            "✏️ Enter Your Own": ""
        }
        
        selected_sample = st.selectbox("Choose a sample or enter your own:", list(sample_abstracts.keys()))
        
        if selected_sample == "✏️ Enter Your Own":
            abstract_text = st.text_area(
                "📋 Paste your RCT abstract here:",
                height=250,
                placeholder="Copy and paste your randomized controlled trial abstract here...",
                help="Enter the full abstract text for comprehensive analysis"
            )
        else:
            abstract_text = st.text_area(
                "📋 Abstract text:",
                value=sample_abstracts[selected_sample],
                height=250,
                help="You can edit this sample text or choose 'Enter Your Own' above"
            )
        
        st.markdown("---")
        
        # Analyze button with better styling
        analyze_button = st.button(
            "🔍 Analyze Quality", 
            type="primary",
            help="Run comprehensive quality assessment",
            use_container_width=True
        )
    
    # Main content area
    if analyze_button and abstract_text.strip():
        st.markdown("---")
        
        # Gateway Step 1: Check if it's a research abstract
        with st.spinner("🔍 Step 1: Checking if this is a research abstract..."):
            try:
                relevance_result = run_single_check(
                    abstract_text, 
                    "relevance_check", 
                    CHECKS["PreChecks"]["relevance_check"]
                )
                
                is_research_abstract, message = check_is_research_abstract(relevance_result)
                
                if not is_research_abstract:
                    display_gateway_failure(message)
                    return
                
                st.success("✅ Step 1: Confirmed as research abstract")
                
            except Exception as e:
                st.error(f"❌ Step 1 failed: {str(e)}")
                with st.expander("🔧 Technical Details"):
                    st.code(traceback.format_exc())
                return
        
        # Gateway Step 2: Check if it's an RCT
        with st.spinner("🔍 Step 2: Checking if this is a randomized controlled trial..."):
            try:
                rct_result = run_single_check(
                    abstract_text, 
                    "RCT_classification", 
                    CHECKS["PreChecks"]["RCT_classification"]
                )
                
                is_rct, message = check_is_rct(rct_result)
                
                if not is_rct:
                    display_gateway_failure(message)
                    return
                
                st.success("✅ Step 2: Confirmed as randomized controlled trial")
                
            except Exception as e:
                st.error(f"❌ Step 2 failed: {str(e)}")
                with st.expander("🔧 Technical Details"):
                    st.code(traceback.format_exc())
                return
        
        # Step 3: Run comprehensive analysis
        analysis_components = []
        analysis_components.append("heuristic assessment")
        
        if run_llm_analysis and selected_llm_pipelines:
            analysis_components.append(f"{len(selected_llm_pipelines)} AI models")
        
        if run_hybrid_analysis and selected_hybrid_pipelines:
            analysis_components.append(f"{len(selected_hybrid_pipelines)} hybrid models")
        
        analysis_description = " + ".join(analysis_components)
        
        with st.spinner(f"🔍 Step 3: Running comprehensive {analysis_description}..."):
            try:
                # Run comprehensive analysis
                results = run_comprehensive_analysis(
                    abstract_text, 
                    selected_llm_pipelines if run_llm_analysis else [],
                    selected_hybrid_pipelines if run_hybrid_analysis else []
                )
                
                st.success("✅ Step 3: Comprehensive quality assessment completed successfully!")
                
                # HEURISTICS RESULTS FIRST (Primary Analysis)
                heuristic_results = results["heuristics"]
                
                st.markdown("## 📊 Quality Assessment Results")
                
                # Count issues by category for summary
                major_issues = 0
                minor_issues = 0
                
                # Count major issues
                if "MajorIssues" in heuristic_results:
                    for check_name, result in heuristic_results["MajorIssues"].items():
                        try:
                            if has_error(result):
                                continue
                            code = safe_get_attr(result, 'code')
                            if code is None:
                                continue
                            severity = code.value if hasattr(code, 'value') else code
                            
                            if check_name in CODE_MAPPINGS and severity in CODE_MAPPINGS[check_name]:
                                mapping = CODE_MAPPINGS[check_name][severity]
                                if mapping["type"] != "success":
                                    major_issues += 1
                            else:
                                if severity != 0:
                                    major_issues += 1
                        except:
                            continue
                
                # Count minor issues
                if "MinorIssues" in heuristic_results:
                    for check_name, result in heuristic_results["MinorIssues"].items():
                        try:
                            if has_error(result):
                                continue
                            code = safe_get_attr(result, 'code')
                            if code is None:
                                continue
                            severity = code.value if hasattr(code, 'value') else code
                            
                            if check_name in CODE_MAPPINGS and severity in CODE_MAPPINGS[check_name]:
                                mapping = CODE_MAPPINGS[check_name][severity]
                                if mapping["type"] != "success":
                                    minor_issues += 1
                            else:
                                if severity != 0:
                                    minor_issues += 1
                        except:
                            continue
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🚨 Major Issues Found", 
                        value=major_issues,
                        help="Critical methodological concerns that may significantly impact study validity"
                    )
                
                with col2:
                    st.metric(
                        label="📋 Minor Issues Found",
                        value=minor_issues,
                        help="Additional considerations that may affect generalizability or interpretation"
                    )
                
                with col3:
                    heuristic_time = results["execution_times"].get("heuristics", 0)
                    st.metric(
                        label="⏱️ Heuristic Time",
                        value=f"{heuristic_time}s",
                        help="Time taken for heuristic analysis"
                    )
                
                with col4:
                    total_time = sum(results["execution_times"].values())
                    st.metric(
                        label="🕐 Total Time",
                        value=f"{total_time:.1f}s",
                        help="Total analysis time"
                    )
                
                st.markdown("---")
                
                # Display heuristic results by category
                category_order = ["PreChecks", "MajorIssues", "MinorIssues"]
                category_titles = {
                    "PreChecks": "🔍 Initial Assessment",
                    "MajorIssues": "🚨 Critical Quality Issues", 
                    "MinorIssues": "📋 Additional Considerations"
                }
                
                for category in category_order:
                    if category in heuristic_results:
                        category_results = heuristic_results[category]
                        
                        st.markdown(f'<div class="category-header">{category_titles[category]}</div>', 
                                   unsafe_allow_html=True)
                        
                        if len(category_results) > 1:
                            cols = st.columns(2)
                            for idx, (check_name, result) in enumerate(category_results.items()):
                                with cols[idx % 2]:
                                    display_name = check_name.replace('Check', '').replace('_', ' ')
                                    if display_name == "composite":
                                        display_name = "Composite Outcomes"
                                    elif display_name == "PrimaryOutCome":
                                        display_name = "Primary Outcome Events"
                                    elif display_name == "AgeGeneralization":
                                        display_name = "Age Generalizability"
                                    elif display_name == "SexGeneralization":
                                        display_name = "Sex Generalizability"
                                    elif display_name == "RCT classification":
                                        display_name = "Study Design"
                                    elif display_name == "relevance ":
                                        display_name = "Content Relevance"
                                    elif display_name == "FollowDuration":
                                        display_name = "Follow-up Duration"
                                    elif display_name == "MultiCenter":
                                        display_name = "Multi-center Status"
                                    elif display_name == "PrimaryTimeLine":
                                        display_name = "Primary Outcome Timeline"
                                    elif display_name == "blinding":
                                        display_name = "Blinding Assessment"
                                    elif display_name == "placebo":
                                        display_name = "Placebo Control"
                                    
                                    display_result_card(display_name, result, category, check_name)
                        else:
                            for check_name, result in category_results.items():
                                display_name = check_name.replace('Check', '').replace('_', ' ')
                                if display_name == "composite":
                                    display_name = "Composite Outcomes"
                                elif display_name == "PrimaryOutCome":
                                    display_name = "Primary Outcome Events"
                                elif display_name == "AgeGeneralization":
                                    display_name = "Age Generalizability"
                                elif display_name == "SexGeneralization":
                                    display_name = "Sex Generalizability"
                                elif display_name == "RCT classification":
                                    display_name = "Study Design"
                                elif display_name == "relevance ":
                                    display_name = "Content Relevance"
                                elif display_name == "FollowDuration":
                                    display_name = "Follow-up Duration"
                                elif display_name == "MultiCenter":
                                    display_name = "Multi-center Status"
                                elif display_name == "PrimaryTimeLine":
                                    display_name = "Primary Outcome Timeline"
                                elif display_name == "blinding":
                                    display_name = "Blinding Assessment"
                                elif display_name == "placebo":
                                    display_name = "Placebo Control"
                                
                                display_result_card(display_name, result, category, check_name)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                
                # HYBRID RESULTS SECOND (Enhanced Analysis)
                if run_hybrid_analysis and results.get("hybrid_pipelines"):
                    st.markdown("---")
                    display_all_hybrid_results(results["hybrid_pipelines"])
                
                # LLM RESULTS THIRD (Additional Analysis)
                if run_llm_analysis and results.get("llm_pipelines"):
                    st.markdown("---")
                    display_all_llm_results(results["llm_pipelines"])
                
                # COMPREHENSIVE ANALYSIS SUMMARY
                if (run_hybrid_analysis and results.get("hybrid_pipelines")) or (run_llm_analysis and results.get("llm_pipelines")):
                    st.markdown("---")
                    st.markdown("## 🔄 Multi-Modal Analysis Comparison")
                    
                    # Create comparison metrics
                    comparison_cols = []
                    if run_hybrid_analysis:
                        comparison_cols.append("Hybrid")
                    if run_llm_analysis:
                        comparison_cols.append("AI Only")
                    comparison_cols.append("Heuristic")
                    
                    cols = st.columns(len(comparison_cols))
                    
                    # Heuristic column
                    heuristic_col_idx = len(comparison_cols) - 1
                    with cols[heuristic_col_idx]:
                        st.markdown("### 📊 Heuristic Analysis")
                        st.metric("Major Issues", major_issues)
                        st.metric("Minor Issues", minor_issues)
                        st.metric("Execution Time", f"{results['execution_times'].get('heuristics', 0)}s")
                        st.caption("✅ Systematic rule-based assessment")
                    
                    # Hybrid column (if enabled)
                    if run_hybrid_analysis:
                        hybrid_col_idx = 0
                        with cols[hybrid_col_idx]:
                            st.markdown("### ⚡ Hybrid Analysis")
                            successful_hybrid = sum(1 for r in results["hybrid_pipelines"].values() if r.get("success"))
                            total_hybrid = len(results["hybrid_pipelines"])
                            avg_hybrid_time = sum(r.get("execution_time", 0) for r in results["hybrid_pipelines"].values()) / max(total_hybrid, 1)
                            
                            st.metric("Successful Pipelines", f"{successful_hybrid}/{total_hybrid}")
                            st.metric("Avg Hybrid Time", f"{avg_hybrid_time:.1f}s")
                            st.metric("Total Hybrid Time", f"{results['execution_times'].get('hybrid_total', 0)}s")
                            st.caption("⚡ AI + Heuristic fusion")
                    
                    # AI-only column (if enabled)
                    if run_llm_analysis:
                        ai_col_idx = 1 if run_hybrid_analysis else 0
                        with cols[ai_col_idx]:
                            st.markdown("### 🤖 AI Analysis")
                            successful_ai = sum(1 for r in results["llm_pipelines"].values() if r.get("success"))
                            total_ai = len(results["llm_pipelines"])
                            avg_ai_time = sum(r.get("execution_time", 0) for r in results["llm_pipelines"].values()) / max(total_ai, 1)
                            
                            st.metric("Successful Pipelines", f"{successful_ai}/{total_ai}")
                            st.metric("Avg AI Time", f"{avg_ai_time:.1f}s")
                            st.metric("Total AI Time", f"{results['execution_times'].get('llm_total', 0)}s")
                            st.caption("🤖 Pure AI reasoning")
                    
                    # Analysis insights
                    st.markdown("### 💡 Analysis Insights")
                    
                    insight_messages = []
                    
                    if run_hybrid_analysis and run_llm_analysis:
                        insight_messages.append("🔄 **Multi-Modal Assessment**: Compare systematic heuristics, hybrid AI+rules, and pure AI reasoning for comprehensive quality evaluation")
                    elif run_hybrid_analysis:
                        insight_messages.append("⚡ **Hybrid Enhancement**: Hybrid models combine rule-based consistency with AI clinical reasoning for balanced assessment")
                    elif run_llm_analysis:
                        insight_messages.append("🤖 **AI Augmentation**: AI models provide clinical reasoning perspective to supplement systematic heuristic findings")
                    
                    if run_hybrid_analysis and results.get("hybrid_pipelines"):
                        successful_hybrid = sum(1 for r in results["hybrid_pipelines"].values() if r.get("success"))
                        if successful_hybrid > 0:
                            insight_messages.append("✅ **Hybrid Success**: Hybrid pipelines successfully combined heuristic and AI components for enhanced accuracy")
                    
                    for message in insight_messages:
                        st.info(message)
                
            except Exception as e:
                st.error(f"❌ Step 3: Quality assessment failed: {str(e)}")
                with st.expander("🔧 Technical Details"):
                    st.code(traceback.format_exc())
                    st.markdown("**Troubleshooting:**")
                    st.markdown("- Check that your abstract is properly formatted")
                    st.markdown("- Ensure all required modules are installed")
                    st.markdown("- Try with a shorter abstract if the issue persists")
                    st.markdown("- Verify that hybrid pipeline modules are available")
    
    elif analyze_button:
        st.warning("⚠️ Please enter an abstract to analyze.")
    
    # Footer information
    st.markdown("---")
    with st.expander("ℹ️ About This Enhanced Analysis Tool"):
        st.markdown("""
        ### 📊 Primary Analysis: Heuristic Assessment
        **Always runs first** - Rule-based systematic quality checks covering:
        - **PreChecks**: Content validation and study design confirmation
        - **Major Issues**: Critical methodological concerns (blinding, placebo, sample size, etc.)
        - **Minor Issues**: Additional considerations (generalizability, funding, follow-up, etc.)
        
        ### ⚡ NEW: Hybrid AI + Heuristic Models
        **Advanced analysis** - Combines rule-based precision with AI clinical reasoning:
        - **MinorIssues Hybrid**: Enhanced minor issues assessment using AI + heuristics
        - **RCTBiasDetection Hybrid**: Comprehensive bias detection and risk assessment
        - **Benefits**: More accurate, context-aware, and clinically relevant evaluations
        
        ### 🤖 Optional Analysis: Pure AI Clinical Evaluation
        **Supplementary AI reasoning** - AI-powered clinical reasoning including:
        - **MajorIssues AI**: Clinical epidemiologist evaluation
        - **LLM Relevance**: AI content classification verification
        - **RCT Classification**: AI study design validation
        - **RCT Summarizer**: Intelligent trial summary and key findings
        
        ### 🔄 Enhanced Analysis Flow
        1. **Gateway checks** ensure appropriate content (heuristic)
        2. **Heuristic analysis** provides systematic quality baseline
        3. **Hybrid analysis** (recommended) combines AI + rules for enhanced accuracy
        4. **Pure AI analysis** (optional) adds clinical reasoning perspective
        5. **Multi-modal comparison** highlights agreements and differences across approaches
        
        ### ⚡ Performance & Accuracy
        - **Heuristics**: Fast, consistent, systematic (baseline)
        - **Hybrid Models**: Best of both worlds - rule precision + AI reasoning
        - **Pure AI**: Clinical context understanding, flexible interpretation
        - **Concurrent execution**: All models run in parallel for efficiency
        - **Comprehensive evaluation**: Multi-modal approach provides most thorough assessment
        
        ### 🎯 Recommended Configuration
        For optimal results, enable **Hybrid Analysis** which provides:
        - Enhanced accuracy through AI + heuristic fusion
        - Context-aware clinical reasoning
        - Balanced precision and flexibility
        - Comprehensive bias and quality assessment
        """)
        
        st.markdown("### 🔧 Model Architecture Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Hybrid Models:**
            - Combine rule-based precision with AI flexibility
            - Context-aware decision making
            - Enhanced clinical relevance
            - Reduced false positives/negatives
            """)
        
        with col2:
            st.markdown("""
            **Pure AI Models:**
            - Clinical reasoning capabilities
            - Natural language understanding
            - Flexible interpretation
            - Contextual insights
            """)

if __name__ == "__main__":
    main()