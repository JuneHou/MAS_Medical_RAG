#!/usr/bin/env python3
"""
Update existing single-agent results.json files to match KARE format:
1. Remove any probability fields from individual results
2. Add pred_distribution and gt_distribution to metrics
"""

import json
from pathlib import Path

def update_results_file(filepath):
    """Update a single results.json file"""
    print(f"\nProcessing: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract predictions and ground truths
    predictions = []
    ground_truths = []
    
    # Update individual results - remove probability fields if they exist
    updated_results = []
    for result in data['results']:
        # Create clean result dict
        clean_result = {
            'patient_id': result['patient_id'],
            'visit_id': result['visit_id'],
            'ground_truth': result['ground_truth'],
            'prediction': result['prediction'],
            'is_fallback': result.get('is_fallback', False),
            'total_generation_time': result['total_generation_time']
        }
        updated_results.append(clean_result)
        
        # Collect for distribution calculation
        if result.get('prediction') is not None and result.get('ground_truth') is not None:
            predictions.append(result['prediction'])
            ground_truths.append(result['ground_truth'])
    
    # Calculate distributions
    pred_dist = {
        0: sum(1 for p in predictions if p == 0),
        1: sum(1 for p in predictions if p == 1)
    }
    gt_dist = {
        0: sum(1 for g in ground_truths if g == 0),
        1: sum(1 for g in ground_truths if g == 1)
    }
    
    # Update metrics
    data['metrics']['pred_distribution'] = pred_dist
    data['metrics']['gt_distribution'] = gt_dist
    
    # Update results
    data['results'] = updated_results
    
    # Save updated file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated {filepath.name}")
    print(f"  - Predictions: {pred_dist}")
    print(f"  - Ground truth: {gt_dist}")
    print(f"  - Removed probability fields from {len(updated_results)} results")

def main():
    results_dir = Path("/data/wang/junh/githubs/Debate/KARE/results")
    
    # List of result files to update
    files_to_update = [
        results_dir / "single_cot_mor_Qwen_Qwen2.5_7B_Instruct_few_shot" / "results.json",
        results_dir / "single_cot_mor_Qwen_Qwen2.5_7B_Instruct_zero_shot" / "results.json",
        results_dir / "single_rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_few_shot" / "results.json",
        results_dir / "single_rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_zero_shot" / "results.json",
    ]
    
    print("Updating single-agent results files to KARE format...")
    print("=" * 60)
    
    for filepath in files_to_update:
        if filepath.exists():
            update_results_file(filepath)
        else:
            print(f"✗ File not found: {filepath}")
    
    print("\n" + "=" * 60)
    print("Update complete!")

if __name__ == "__main__":
    main()
