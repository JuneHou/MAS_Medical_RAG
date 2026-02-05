#!/usr/bin/env python3
"""
Analyze patients with missing retrieve files to determine if they have
retrieval behavior in their debate logs, or if the retrieval system failed.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path("/data/wang/junh/githubs/Debate/KARE")
RESULTS_DIR = BASE_DIR / "results/rag/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__projects_slmreasoning_junh_Debate_KARE_verl_models_prediction_brier_7b_step150_MedCPT_8_8"
DEBATE_LOGS_DIR = RESULTS_DIR / "debate_logs"

ROLES = [
    "target_patient_analyst",
    "mortality_risk_assessor", 
    "protective_factor_analyst",
    "mortality_assessment",
    "survival_assessment"
]

def get_all_patient_ids():
    """Get all patient IDs from debate log filenames."""
    patient_ids = set()
    for log_file in DEBATE_LOGS_DIR.glob("debate_responses_*.log"):
        patient_id = log_file.stem.replace("debate_responses_", "")
        patient_ids.add(patient_id)
    return sorted(patient_ids)

def get_patients_with_retrieve(role):
    """Get patient IDs that have retrieve files for a specific role."""
    patient_ids = set()
    for retrieve_file in DEBATE_LOGS_DIR.glob(f"retrieve_{role}_*.json"):
        patient_id = retrieve_file.stem.replace(f"retrieve_{role}_", "")
        patient_ids.add(patient_id)
    return patient_ids

def check_retrieval_in_log(patient_id, role):
    """
    Check if retrieval behavior exists in the debate log for this patient/role.
    Returns dict with:
        - has_retrieve_call: bool (whether retrieve() is called)
        - retrieve_queries: list of queries
        - has_error: bool (whether there are errors)
        - error_messages: list of error messages
    """
    log_file = DEBATE_LOGS_DIR / f"debate_responses_{patient_id}.log"
    
    if not log_file.exists():
        return {
            "has_log": False,
            "has_retrieve_call": False,
            "retrieve_queries": [],
            "has_error": False,
            "error_messages": []
        }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Search for role-specific section
    role_pattern = re.escape(role.replace("_", " ").title())
    
    # Look for retrieve calls - both in code and in PARSED TOOL CALL logs
    retrieve_call_pattern = r'retrieve\s*\(["\']([^"\']+)["\']\)'
    tool_call_pattern = r'PARSED TOOL CALL:\s*tool=["\']retrieve["\'],\s*query=["\']([^"\']+)["\']'
    
    retrieve_queries = []
    retrieve_queries.extend(re.findall(retrieve_call_pattern, content))
    retrieve_queries.extend(re.findall(tool_call_pattern, content))
    
    # Look for errors
    error_pattern = r'(error|exception|failed|Error|Exception|Failed)'
    error_matches = re.findall(f'{error_pattern}.*', content, re.IGNORECASE)
    
    # Filter out common non-error matches
    real_errors = [e for e in error_matches if not any(skip in e.lower() for skip in ['format_enforcer', 'no error'])]
    
    return {
        "has_log": True,
        "has_retrieve_call": len(retrieve_queries) > 0,
        "retrieve_queries": retrieve_queries,
        "has_error": len(real_errors) > 0,
        "error_messages": real_errors[:5]  # First 5 errors
    }

def analyze_missing_retrieve_files():
    """Main analysis function."""
    
    print("=" * 80)
    print("ANALYZING PATIENTS WITH MISSING RETRIEVE FILES")
    print("=" * 80)
    print()
    
    all_patients = get_all_patient_ids()
    print(f"Total patients with debate logs: {len(all_patients)}")
    print()
    
    # For each role, find missing retrieve files
    results = defaultdict(list)
    
    for role in ROLES:
        print(f"\n{'='*80}")
        print(f"ROLE: {role}")
        print(f"{'='*80}")
        
        patients_with_retrieve = get_patients_with_retrieve(role)
        missing_patients = sorted(set(all_patients) - patients_with_retrieve)
        
        print(f"Patients with retrieve files: {len(patients_with_retrieve)}")
        print(f"Patients missing retrieve files: {len(missing_patients)}")
        
        if len(missing_patients) == 0:
            print("✓ All patients have retrieve files for this role!")
            continue
        
        print(f"\nAnalyzing {len(missing_patients)} patients missing retrieve files...")
        print()
        
        # Categorize missing patients
        has_retrieval_attempt = []
        has_errors = []
        no_retrieval_evidence = []
        
        for patient_id in missing_patients:
            log_info = check_retrieval_in_log(patient_id, role)
            
            if not log_info["has_log"]:
                continue
                
            if log_info["has_retrieve_call"]:
                has_retrieval_attempt.append({
                    "patient_id": patient_id,
                    "queries": log_info["retrieve_queries"],
                    "has_error": log_info["has_error"],
                    "errors": log_info["error_messages"]
                })
            else:
                no_retrieval_evidence.append(patient_id)
            
            if log_info["has_error"]:
                has_errors.append({
                    "patient_id": patient_id,
                    "errors": log_info["error_messages"]
                })
        
        # Print summary
        print(f"SUMMARY FOR {role}:")
        print(f"  Patients with retrieval attempts but no file: {len(has_retrieval_attempt)}")
        print(f"  Patients with no retrieval evidence: {len(no_retrieval_evidence)}")
        print(f"  Patients with errors in log: {len(has_errors)}")
        print()
        
        # Show details for patients with retrieval attempts but missing files
        if has_retrieval_attempt:
            print(f"\n  RETRIEVAL ATTEMPTS WITHOUT FILES (first 10):")
            for item in has_retrieval_attempt[:10]:
                print(f"    Patient {item['patient_id']}:")
                print(f"      Queries: {len(item['queries'])} retrieve calls")
                if item['queries'][:2]:
                    for q in item['queries'][:2]:
                        print(f"        - '{q[:80]}...'")
                if item['has_error']:
                    print(f"      ⚠ Has errors in log")
                    for err in item['errors'][:2]:
                        print(f"        - {err[:100]}...")
        
        # Show patients with no retrieval evidence
        if no_retrieval_evidence:
            print(f"\n  NO RETRIEVAL EVIDENCE (first 20):")
            print(f"    {', '.join(no_retrieval_evidence[:20])}")
        
        results[role] = {
            "total_missing": len(missing_patients),
            "has_retrieval_attempt": len(has_retrieval_attempt),
            "no_retrieval_evidence": len(no_retrieval_evidence),
            "has_errors": len(has_errors),
            "missing_patients": missing_patients,
            "retrieval_attempts": has_retrieval_attempt
        }
    
    # Overall summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print()
    print(f"{'Role':<30} {'Missing':<10} {'W/Retrieval':<15} {'No Evidence':<15}")
    print("-" * 80)
    for role in ROLES:
        r = results[role]
        print(f"{role:<30} {r['total_missing']:<10} {r['has_retrieval_attempt']:<15} {r['no_retrieval_evidence']:<15}")
    
    # Key finding
    print(f"\n\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    for role in ROLES:
        r = results[role]
        if r['has_retrieval_attempt'] > 0:
            print(f"\n{role}:")
            print(f"  ⚠ {r['has_retrieval_attempt']} patients show retrieval attempts in logs but have NO retrieve files")
            print(f"  → This suggests file writing failure, not retrieval system failure")
        
        if r['no_retrieval_evidence'] > 0:
            print(f"\n{role}:")
            print(f"  ⚠ {r['no_retrieval_evidence']} patients show NO retrieval evidence in logs")
            print(f"  → This suggests retrieval was never attempted for these patients")

if __name__ == "__main__":
    analyze_missing_retrieve_files()
