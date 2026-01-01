#!/usr/bin/env python3
"""
Analyze retrieve file coverage for mortality prediction analysis.
Check which patients have retrieve files and compare with always_wrong and retry patients.
"""

import os
import re
import json
import pandas as pd
from collections import defaultdict

# Configuration
patient_path = "/data/wang/junh/githubs/Debate/KARE/data/ehr_data/mimic3_mortality_samples_test.json"
base_path_rag = "/data/wang/junh/githubs/Debate/KARE/results/rag"
target_model = "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__projects_slmreasoning_junh_Debate_KARE_verl_models_prediction_brier_7b_step150_MedCPT_8_8"

# Roles to check
roles = [
    "target_patient_analyst",
    "mortality_risk_assessor",
    "protective_factor_analyst",
    "mortality_assessment",
    "survival_assessment",
]

role_names_display = {
    "target_patient_analyst": "Target Patient Analyst",
    "mortality_risk_assessor": "Mortality Risk Assessor",
    "protective_factor_analyst": "Protective Factor Analyst",
    "mortality_assessment": "Integrator Mortality",
    "survival_assessment": "Integrator Survival",
}

def extract_retrieved_doc_length(retrieve_file_path):
    """Extract total retrieved document content length from a retrieve JSON file (approx tokens)."""
    try:
        with open(retrieve_file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        total_length = 0
        for doc in data.get("retrieved_documents", []):
            content = doc.get("content", "")
            if isinstance(content, str):
                total_length += len(content)

        return total_length / 4.0  # ~4 chars/token
    except Exception:
        return None


def load_patient_data():
    """Load patient data from JSON file."""
    with open(patient_path, 'r') as f:
        patient_data = json.load(f)
    return patient_data


def get_always_wrong_patients():
    """Get list of patients that are always predicted wrong across all models - using target model only for speed."""
    # For speed, just check the target model
    results_file = os.path.join(base_path_rag, target_model, 'kare_debate_mortality_results.json')
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found at {results_file}")
        return []
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # For this quick analysis, just return patients with wrong predictions in this model
    always_wrong = []
    for result in results['results']:
        patient_id = result['patient_id']
        prediction = result['prediction']
        ground_truth = result['ground_truth']
        
        if prediction != ground_truth:
            always_wrong.append(patient_id)
    
    return always_wrong


def get_retry_patients():
    """
    Get list of patients that had retries in the integrator ACROSS ALL MODELS.
    
    Retry is triggered when mortality OR survival probability extraction fails (returns None).
    Detection methods:
    1. Look for "INTEGRATOR RETRY" log message
    2. Count "EXTRACTED MORTALITY/SURVIVAL PROBABILITY" entries (>1 means retry)
    
    Returns unique patient IDs that had retries in any model.
    """
    def check_for_integrator_retry(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Method 1: Check for explicit retry message
            has_retry_message = 'INTEGRATOR RETRY' in content
            
            # Method 2: Count extraction attempts
            # Pattern matches: "EXTRACTED MORTALITY PROBABILITY: 0.5" or "EXTRACTED MORTALITY PROBABILITY: None"
            mortality_pattern = r'EXTRACTED MORTALITY PROBABILITY:'
            survival_pattern = r'EXTRACTED SURVIVAL PROBABILITY:'
            
            mortality_count = len(re.findall(mortality_pattern, content))
            survival_count = len(re.findall(survival_pattern, content))
            
            # Retry occurred if:
            # - Explicit retry message exists, OR
            # - More than 1 extraction attempt (initial + retry)
            has_retry = has_retry_message or (mortality_count > 1 or survival_count > 1)
            
            return {
                'has_retry': has_retry,
                'retry_message_found': has_retry_message,
                'mortality_extractions': mortality_count,
                'survival_extractions': survival_count
            }
        except Exception:
            return {
                'has_retry': False,
                'retry_message_found': False,
                'mortality_extractions': 0,
                'survival_extractions': 0
            }
    
    # Check ALL models in the rag folder
    all_retry_patients = set()  # Use set to avoid duplicates
    
    model_dirs = [d for d in os.listdir(base_path_rag) if os.path.isdir(os.path.join(base_path_rag, d))]
    
    print(f"\nChecking retry patients across {len(model_dirs)} models...")
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path_rag, model_dir)
        debate_logs_path = os.path.join(model_path, 'debate_logs')
        
        if not os.path.exists(debate_logs_path):
            continue
        
        log_files = [f for f in os.listdir(debate_logs_path) if f.startswith('debate_responses_') and f.endswith('.log')]
        
        model_retry_count = 0
        for log_file in log_files:
            log_path = os.path.join(debate_logs_path, log_file)
            patient_id = log_file.replace('debate_responses_', '').replace('.log', '')
            
            retry_info = check_for_integrator_retry(log_path)
            
            if retry_info['has_retry']:
                all_retry_patients.add(patient_id)
                model_retry_count += 1
        
        if model_retry_count > 0:
            print(f"  {model_dir}: {model_retry_count} patients with retries")
    
    print(f"\nTotal unique patients with retries across all models: {len(all_retry_patients)}")
    
    return list(all_retry_patients)


def analyze_retrieve_files():
    """Analyze retrieve files for the target model."""
    print("=" * 80)
    print("RETRIEVE FILE COVERAGE ANALYSIS")
    print("=" * 80)
    print(f"\nTarget Model: {target_model}")
    print()
    
    model_path = os.path.join(base_path_rag, target_model)
    debate_logs_path = os.path.join(model_path, "debate_logs")
    
    if not os.path.isdir(debate_logs_path):
        print(f"ERROR: Model directory not found at {model_path}")
        return
    
    print(f"Debate logs path: {debate_logs_path}\n")
    
    # Initialize storage
    role_lengths = {role: [] for role in roles}
    role_file_counts = {role: 0 for role in roles}
    
    # Get all retrieve JSON files
    retrieve_files = [f for f in os.listdir(debate_logs_path) if f.startswith("retrieve_") and f.endswith(".json")]
    print(f"Total retrieve files found: {len(retrieve_files)}\n")
    
    # Parse retrieve files
    filename_re = re.compile(r"retrieve_(.+)_(\d+)_(\d+)\.json$")
    
    for retrieve_file in retrieve_files:
        m = filename_re.match(retrieve_file)
        if not m:
            continue
        
        role = m.group(1)
        patient_id = f"{m.group(2)}_{m.group(3)}"
        
        if role not in roles:
            continue
        
        role_file_counts[role] += 1
        retrieve_path = os.path.join(debate_logs_path, retrieve_file)
        doc_length = extract_retrieved_doc_length(retrieve_path)
        
        if doc_length is not None:
            role_lengths[role].append({"patient_id": patient_id, "doc_length": doc_length})
    
    # Print file counts per role
    print("RETRIEVE FILES PER ROLE:")
    print("-" * 80)
    for role in roles:
        parsed = len(role_lengths[role])
        print(f"  {role_names_display[role]:30s}: {role_file_counts[role]:4d} files, {parsed:4d} successfully parsed")
    
    # Get patient groups
    print("\n" + "=" * 80)
    print("LOADING PATIENT GROUPS")
    print("=" * 80)
    
    patient_data = load_patient_data()
    always_wrong_ids = get_always_wrong_patients()
    retry_patient_ids = get_retry_patients()
    
    print(f"\nTotal patients in dataset: {len(patient_data)}")
    print(f"Wrong predictions in target model: {len(always_wrong_ids)}")
    print(f"Retry patients across ALL models: {len(retry_patient_ids)}")
    
    # Convert to DataFrames
    role_dfs = {role: pd.DataFrame(v) for role, v in role_lengths.items() if len(v) > 0}
    
    # Get unique patient IDs from any role
    all_patients_with_retrieve = set()
    for role, df in role_dfs.items():
        all_patients_with_retrieve.update(df["patient_id"].unique().tolist())
    
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    print(f"\nUnique patients with retrieve files (any role): {len(all_patients_with_retrieve)}")
    print(f"Expected total patients: {len(patient_data)}")
    print(f"Missing: {len(patient_data) - len(all_patients_with_retrieve)}")
    
    # Check coverage for always_wrong
    if len(always_wrong_ids) > 0:
        always_wrong_with_retrieve = set(always_wrong_ids) & all_patients_with_retrieve
        always_wrong_missing = set(always_wrong_ids) - all_patients_with_retrieve
        
        print(f"\nALWAYS-WRONG PATIENTS:")
        print(f"  Total always-wrong: {len(always_wrong_ids)}")
        print(f"  With retrieve files: {len(always_wrong_with_retrieve)}")
        print(f"  Missing retrieve files: {len(always_wrong_missing)}")
        
        if len(always_wrong_missing) > 0:
            print(f"\n  Missing patient IDs (first 10):")
            for pid in sorted(list(always_wrong_missing))[:10]:
                print(f"    {pid}")
            if len(always_wrong_missing) > 10:
                print(f"    ... and {len(always_wrong_missing) - 10} more")
    
    # Check coverage for retry
    if len(retry_patient_ids) > 0:
        retry_with_retrieve = set(retry_patient_ids) & all_patients_with_retrieve
        retry_missing = set(retry_patient_ids) - all_patients_with_retrieve
        
        print(f"\nRETRY PATIENTS:")
        print(f"  Total retry patients: {len(retry_patient_ids)}")
        print(f"  With retrieve files: {len(retry_with_retrieve)}")
        print(f"  Missing retrieve files: {len(retry_missing)}")
        
        if len(retry_missing) > 0:
            print(f"\n  Missing patient IDs (first 10):")
            for pid in sorted(list(retry_missing))[:10]:
                print(f"    {pid}")
            if len(retry_missing) > 10:
                print(f"    ... and {len(retry_missing) - 10} more")
    
    # Per-role statistics
    print("\n" + "=" * 80)
    print("PER-ROLE STATISTICS")
    print("=" * 80)
    
    for role in roles:
        if role not in role_dfs:
            print(f"\n{role_names_display[role]}: No data")
            continue
        
        role_df = role_dfs[role]
        all_lengths = role_df["doc_length"]
        always_wrong_lengths = role_df[role_df["patient_id"].isin(always_wrong_ids)]["doc_length"]
        retry_lengths = role_df[role_df["patient_id"].isin(retry_patient_ids)]["doc_length"]
        
        print(f"\n{role_names_display[role]}:")
        print(f"  Total patients with data: {len(all_lengths)}")
        print(f"  Always-wrong patients: {len(always_wrong_lengths)} / {len(always_wrong_ids)}")
        print(f"  Retry patients: {len(retry_lengths)} / {len(retry_patient_ids)}")
        print(f"\n  Document Length Statistics (tokens):")
        print(f"    All patients - Mean: {all_lengths.mean():.2f}, Median: {all_lengths.median():.2f}, Std: {all_lengths.std():.2f}")
        
        if len(always_wrong_lengths) > 0:
            print(f"    Always-wrong - Mean: {always_wrong_lengths.mean():.2f}, Median: {always_wrong_lengths.median():.2f}, Std: {always_wrong_lengths.std():.2f}")
        
        if len(retry_lengths) > 0:
            print(f"    Retry        - Mean: {retry_lengths.mean():.2f}, Median: {retry_lengths.median():.2f}, Std: {retry_lengths.std():.2f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary_data = []
    for role in roles:
        if role not in role_dfs:
            summary_data.append({
                "Role": role_names_display[role],
                "Total": 0,
                "Always-Wrong": 0,
                "Retry": 0,
                "Mean_All": 0,
                "Mean_AlwaysWrong": 0,
                "Mean_Retry": 0,
            })
            continue
        
        df = role_dfs[role]
        all_s = df["doc_length"]
        aw_s = df[df["patient_id"].isin(always_wrong_ids)]["doc_length"]
        rt_s = df[df["patient_id"].isin(retry_patient_ids)]["doc_length"]
        
        summary_data.append({
            "Role": role_names_display[role],
            "Total": len(all_s),
            "Always-Wrong": len(aw_s),
            "Retry": len(rt_s),
            "Mean_All": all_s.mean(),
            "Mean_AlwaysWrong": aw_s.mean() if len(aw_s) > 0 else 0,
            "Mean_Retry": rt_s.mean() if len(rt_s) > 0 else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nPatient counts per role:")
    print(summary_df[["Role", "Total", "Always-Wrong", "Retry"]].to_string(index=False))
    
    print("\n\nMean document lengths (tokens) per role:")
    print(summary_df[["Role", "Mean_All", "Mean_AlwaysWrong", "Mean_Retry"]].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_retrieve_files()
