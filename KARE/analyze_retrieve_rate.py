#!/usr/bin/env python3
"""
Analyze retrieve rate: Check percentage of patients for which integrator generated a query,
and extract patient_id, ground_truth, prediction, query, and retrieve_score (for 8 docs).
"""

import json
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def extract_query_from_log(log_file: str) -> Optional[str]:
    """
    Extract query from debate log file.
    Looks for pattern: PARSED TOOL CALL: tool='...', query='...'
    
    Returns:
        Query string if found, None otherwise
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match: PARSED TOOL CALL: tool='...', query='...'
        pattern = r"PARSED TOOL CALL:\s*tool=['\"]([^'\"]+)['\"],\s*query=['\"]([^'\"]+)['\"]"
        match = re.search(pattern, content)
        
        if match:
            tool_name = match.group(1)
            query = match.group(2)
            # Return query only if it's not 'None'
            if query and query != 'None' and tool_name and tool_name != 'None':
                return query
        
        return None
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None


def get_retrieve_scores(patient_id: str, log_dir: Path) -> List[float]:
    """
    Get retrieve scores for 8 documents from retrieve JSON files.
    
    Args:
        patient_id: Patient ID (e.g., "10117_0")
        log_dir: Directory containing retrieve JSON files
        
    Returns:
        List of 8 scores (or empty list if not found)
    """
    scores = []
    
    # Try combined retrieve file first
    combined_file = log_dir / f"retrieve_integrator_combined_{patient_id}.json"
    if combined_file.exists():
        try:
            with open(combined_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                retrieved_docs = data.get('retrieved_documents', [])
                # Extract scores from documents
                for doc in retrieved_docs:
                    if 'score' in doc:
                        scores.append(float(doc['score']))
                return scores[:8]  # Return up to 8 scores
        except Exception as e:
            print(f"Error reading combined retrieve file {combined_file}: {e}")
    
    # Try separate medcorp and umls files
    medcorp_file = log_dir / f"retrieve_integrator_medcorp_balanced_clinical_integrator_{patient_id}.json"
    umls_file = log_dir / f"retrieve_integrator_umls_balanced_clinical_integrator_{patient_id}.json"
    
    medcorp_scores = []
    umls_scores = []
    
    if medcorp_file.exists():
        try:
            with open(medcorp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                medcorp_scores = data.get('scores', [])
        except Exception as e:
            print(f"Error reading medcorp file {medcorp_file}: {e}")
    
    if umls_file.exists():
        try:
            with open(umls_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                umls_scores = data.get('scores', [])
        except Exception as e:
            print(f"Error reading umls file {umls_file}: {e}")
    
    # Combine scores (medcorp first, then umls)
    all_scores = medcorp_scores + umls_scores
    return [float(s) for s in all_scores[:8]]  # Return up to 8 scores


def analyze_retrieve_rate(results_dir: str, output_csv: str):
    """
    Analyze retrieve rate for a results directory.
    
    Args:
        results_dir: Path to results directory (e.g., "rag_mor_...")
        output_csv: Output CSV file path
    """
    results_dir = Path(results_dir)
    
    # Find results JSON file
    results_json = results_dir / "kare_debate_mortality_results.json"
    if not results_json.exists():
        print(f"Error: Results JSON not found at {results_json}")
        return
    
    # Find debate_logs directory
    log_dir = results_dir / "debate_logs"
    if not log_dir.exists():
        print(f"Error: Debate logs directory not found at {log_dir}")
        return
    
    # Load results JSON
    print(f"Loading results from {results_json}...")
    with open(results_json, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    results = results_data.get('results', [])
    print(f"Found {len(results)} patients in results")
    
    # Process each patient
    csv_rows = []
    queries_generated = 0
    total_patients = len(results)
    
    for result in results:
        patient_id = result.get('patient_id', '')
        ground_truth = result.get('ground_truth', '')
        prediction = result.get('prediction', '')
        
        # Extract query from log file
        log_file = log_dir / f"debate_responses_{patient_id}.log"
        query = extract_query_from_log(str(log_file)) if log_file.exists() else None
        
        if query:
            queries_generated += 1
        
        # Get retrieve scores
        retrieve_scores = get_retrieve_scores(patient_id, log_dir)
        
        # Pad scores to 8 if needed
        while len(retrieve_scores) < 8:
            retrieve_scores.append(None)
        retrieve_scores = retrieve_scores[:8]
        
        # Create CSV row
        row = {
            'patient_id': patient_id,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'query': query if query else '',
            'retrieve_score_1': retrieve_scores[0] if len(retrieve_scores) > 0 else '',
            'retrieve_score_2': retrieve_scores[1] if len(retrieve_scores) > 1 else '',
            'retrieve_score_3': retrieve_scores[2] if len(retrieve_scores) > 2 else '',
            'retrieve_score_4': retrieve_scores[3] if len(retrieve_scores) > 3 else '',
            'retrieve_score_5': retrieve_scores[4] if len(retrieve_scores) > 4 else '',
            'retrieve_score_6': retrieve_scores[5] if len(retrieve_scores) > 5 else '',
            'retrieve_score_7': retrieve_scores[6] if len(retrieve_scores) > 6 else '',
            'retrieve_score_8': retrieve_scores[7] if len(retrieve_scores) > 7 else '',
        }
        csv_rows.append(row)
    
    # Calculate percentage
    query_percentage = (queries_generated / total_patients * 100) if total_patients > 0 else 0
    
    print(f"\n=== Analysis Results ===")
    print(f"Total patients: {total_patients}")
    print(f"Patients with queries generated: {queries_generated}")
    print(f"Query generation rate: {query_percentage:.2f}%")
    
    # Write CSV
    if csv_rows:
        fieldnames = ['patient_id', 'ground_truth', 'prediction', 'query',
                      'retrieve_score_1', 'retrieve_score_2', 'retrieve_score_3', 'retrieve_score_4',
                      'retrieve_score_5', 'retrieve_score_6', 'retrieve_score_7', 'retrieve_score_8']
        
        print(f"\nWriting results to {output_csv}...")
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Successfully wrote {len(csv_rows)} rows to {output_csv}")
    else:
        print("No data to write")


def main():
    """Main function to analyze both result directories."""
    base_dir = Path("/data/wang/junh/githubs/Debate/KARE/results_unbiased")
    
    # Two directories to analyze
    dirs_to_analyze = [
        "rag_mor__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8",
        "rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8"
    ]
    
    for dir_name in dirs_to_analyze:
        results_dir = base_dir / dir_name
        if not results_dir.exists():
            print(f"Warning: Directory {results_dir} does not exist, skipping...")
            continue
        
        # Create output CSV filename based on directory name
        output_csv = base_dir / f"{dir_name}_retrieve_analysis.csv"
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {dir_name}")
        print(f"{'='*80}")
        
        analyze_retrieve_rate(str(results_dir), str(output_csv))


if __name__ == "__main__":
    main()
