# heuristics_runner.py

import importlib
from typing import Dict
import traceback

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
        "AgeGeneralization": "heuristics_model.MinorIssues.AgeGeneralization",
        "FollowDuration": "heuristics_model.MinorIssues.FollowDuration",
        "Funding": "heuristics_model.MinorIssues.Funding",
        "ITT": "heuristics_model.MinorIssues.ITT",
        "MultiCenter": "heuristics_model.MinorIssues.MultiCenter",
        "PrimaryTimeLine": "heuristics_model.MinorIssues.PrimaryTimeLine",
        "SexGeneralization": "heuristics_model.MinorIssues.SexGeneralization"
    }
}


def run_all_heuristics(abstract: str) -> Dict[str, Dict[str, dict]]:
    """
    Run all registered heuristic modules on the given abstract.

    Returns:
        results: dict structured as {category: {module_name: output}}
    """
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
                    category_results[module_name] = {
                        "error": "Missing run_check()"
                    }
            except Exception as e:
                category_results[module_name] = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }

        results[category] = category_results

    return results

abstract = """
This randomized trial evaluated Drug A versus standard care in 250 patients...
"""

results = run_all_heuristics(abstract)
print(results)
