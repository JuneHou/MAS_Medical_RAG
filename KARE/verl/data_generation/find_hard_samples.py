"""
Find hard samples where mortality probability extraction failed (None).

This script scans debate log directories to identify samples where the integrator
failed to output a parseable mortality probability format.
"""

import os
import re
from typing import List, Tuple


def extract_mortality_probability_from_log(log_path: str) -> Tuple[str, str, bool]:
    """
    Check if a debate log has failed mortality probability extraction.
    
    Args:
        log_path: Path to debate log file
        
    Returns:
        Tuple of (patient_id, visit_id, is_hard_case)
        is_hard_case is True if "EXTRACTED MORTALITY PROBABILITY: None"
    """
    # Extract patient_id and visit_id from filename
    # Format: debate_responses_<patient_id>_<visit_id>.log
    filename = os.path.basename(log_path)
    match = re.search(r'debate_responses_(\d+)_(\d+)\.log', filename)
    if not match:
        return None, None, False
    
    patient_id, visit_id = match.groups()
    
    # Check if log contains "EXTRACTED MORTALITY PROBABILITY: None"
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            if "EXTRACTED MORTALITY PROBABILITY: None" in content:
                return patient_id, visit_id, True
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return patient_id, visit_id, False
    
    return patient_id, visit_id, False


def find_hard_samples(debate_dir: str) -> List[Tuple[str, str]]:
    """
    Find all hard samples in a debate directory.
    
    Args:
        debate_dir: Path to debate logs directory
        
    Returns:
        List of (patient_id, visit_id) tuples for hard cases
    """
    hard_samples = []
    
    log_dir = os.path.join(debate_dir, "debate_logs")
    if not os.path.exists(log_dir):
        print(f"Warning: {log_dir} does not exist")
        return hard_samples
    
    # Scan all log files
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    print(f"Scanning {len(log_files)} log files in {log_dir}...")
    
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        patient_id, visit_id, is_hard = extract_mortality_probability_from_log(log_path)
        
        if is_hard and patient_id and visit_id:
            hard_samples.append((patient_id, visit_id))
    
    print(f"Found {len(hard_samples)} hard samples")
    return hard_samples


def main():
    """Main entry point - find hard samples in all debate directories."""
    
    # Directories to scan
    debate_dirs = [
        "/data/wang/junh/githubs/Debate/KARE/results/arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8",
        "/data/wang/junh/githubs/Debate/KARE/results/fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8",
        "/data/wang/junh/githubs/Debate/KARE/results/long_rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8"
    ]
    
    all_hard_samples = {}
    
    for debate_dir in debate_dirs:
        if not os.path.exists(debate_dir):
            print(f"Skipping {debate_dir} (does not exist)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Scanning: {os.path.basename(debate_dir)}")
        print(f"{'='*80}")
        
        hard_samples = find_hard_samples(debate_dir)
        all_hard_samples[debate_dir] = hard_samples
        
        # Show first 10 examples
        if hard_samples:
            print(f"\nFirst 10 hard samples:")
            for patient_id, visit_id in hard_samples[:10]:
                print(f"  - {patient_id}_{visit_id}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_hard = sum(len(samples) for samples in all_hard_samples.values())
    print(f"Total hard samples across all directories: {total_hard}")
    
    for debate_dir, samples in all_hard_samples.items():
        if samples:
            print(f"\n{os.path.basename(debate_dir)}: {len(samples)} hard samples")


if __name__ == "__main__":
    main()
