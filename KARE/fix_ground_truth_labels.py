#!/usr/bin/env python3
"""
Ground Truth Label Remedy Script
Fixes the ground truth labels in already saved result files by matching patient IDs
with the correct labels from the source MIMIC-III test data.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any

def load_correct_labels(test_data_path: str) -> Dict[str, int]:
    """
    Load the correct ground truth labels from MIMIC-III test data.
    
    Args:
        test_data_path: Path to the mimic3_mortality_samples_test.json file
        
    Returns:
        Dictionary mapping patient_id to correct ground truth label
    """
    print(f"Loading correct labels from: {test_data_path}")
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Create mapping from KARE patient ID format to labels
    patient_to_label = {}
    
    for patient_data in test_data:
        base_patient_id = str(patient_data['patient_id'])
        
        # Calculate visit index (number of visits - 1)
        conditions_visits = patient_data.get('conditions', [[]])
        procedures_visits = patient_data.get('procedures', [[]])
        drugs_visits = patient_data.get('drugs', [[]])
        num_visits = max(len(conditions_visits), len(procedures_visits), len(drugs_visits))
        visit_index = num_visits - 1  # 0-based index
        
        # Construct KARE's temporal patient ID format
        kare_patient_id = f"{base_patient_id}_{visit_index}"
        
        # Get the correct label
        correct_label = patient_data.get('label', 0)  # Use 'label' not 'labels'
        
        patient_to_label[kare_patient_id] = correct_label
    
    # Count labels for verification
    label_counts = {}
    for label in patient_to_label.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Loaded {len(patient_to_label)} patient mappings")
    print(f"Label distribution: {label_counts}")
    
    return patient_to_label

def fix_result_file(result_file_path: str, correct_labels: Dict[str, int], 
                   backup: bool = True) -> None:
    """
    Fix ground truth labels in a result file.
    
    Args:
        result_file_path: Path to the result JSON file
        correct_labels: Dictionary mapping patient_id to correct labels
        backup: Whether to create a backup of the original file
    """
    print(f"\nProcessing result file: {result_file_path}")
    
    # Create backup if requested
    if backup:
        backup_path = f"{result_file_path}.backup"
        if not os.path.exists(backup_path):
            print(f"Creating backup: {backup_path}")
            os.system(f"cp '{result_file_path}' '{backup_path}'")
        else:
            print(f"Backup already exists: {backup_path}")
    
    # Load the result file
    with open(result_file_path, 'r') as f:
        result_data = json.load(f)
    
    # Track changes
    changes_made = 0
    missing_patients = []
    original_gt_counts = {}
    new_gt_counts = {}
    
    # Fix ground truth labels in results
    for result in result_data.get('results', []):
        patient_id = result.get('patient_id')
        original_gt = result.get('ground_truth', 0)
        
        # Count original labels
        original_gt_counts[original_gt] = original_gt_counts.get(original_gt, 0) + 1
        
        if patient_id in correct_labels:
            correct_gt = correct_labels[patient_id]
            if original_gt != correct_gt:
                result['ground_truth'] = correct_gt
                changes_made += 1
            
            # Count new labels
            new_gt_counts[correct_gt] = new_gt_counts.get(correct_gt, 0) + 1
        else:
            missing_patients.append(patient_id)
            # Keep original label for missing patients
            new_gt_counts[original_gt] = new_gt_counts.get(original_gt, 0) + 1
    
    print(f"Changes made: {changes_made}")
    print(f"Missing patients: {len(missing_patients)}")
    if missing_patients:
        print(f"First few missing: {missing_patients[:5]}")
    
    print(f"Original GT distribution: {original_gt_counts}")
    print(f"New GT distribution: {new_gt_counts}")
    
    # Recalculate metrics with correct ground truth labels
    if 'metrics' in result_data:
        print("Recalculating metrics...")
        new_metrics = calculate_metrics(result_data['results'])
        result_data['metrics'] = new_metrics
        
        # Update metadata
        if 'metadata' in result_data:
            result_data['metadata']['ground_truth_fixed'] = True
            result_data['metadata']['changes_made'] = changes_made
            result_data['metadata']['missing_patients'] = len(missing_patients)
    
    # Save the corrected file
    with open(result_file_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed result file saved: {result_file_path}")

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the predictions.
    
    Args:
        results: List of prediction results
        
    Returns:
        Dictionary of calculated metrics
    """
    if not results:
        return {}
    
    # Extract predictions and ground truth, handling None values
    predictions = []
    ground_truths = []
    none_predictions = 0
    
    for result in results:
        pred = result.get('prediction')
        gt = result.get('ground_truth')
        
        if pred is None:
            none_predictions += 1
            continue
            
        if gt is not None:
            predictions.append(pred)
            ground_truths.append(gt)
    
    if not predictions:
        return {}
    
    # Basic metrics
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / total
    
    # Calculate confusion matrix for mortality class (1)
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 0)
    
    # Check prediction distribution
    pred_0_count = sum(1 for p in predictions if p == 0)
    pred_1_count = sum(1 for p in predictions if p == 1)
    gt_0_count = sum(1 for g in ground_truths if g == 0)
    gt_1_count = sum(1 for g in ground_truths if g == 1)
    
    # Calculate metrics with better handling of edge cases
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # Specificity (true negative rate)
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'total_samples': total,
        'valid_predictions': len(predictions),
        'none_predictions': none_predictions,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'pred_distribution': {'0': pred_0_count, '1': pred_1_count},
        'gt_distribution': {'0': gt_0_count, '1': gt_1_count}
    }

def find_result_files(base_dir: str = "results") -> List[str]:
    """
    Find all result JSON files in the results directory.
    
    Args:
        base_dir: Base directory to search for result files
        
    Returns:
        List of paths to result files
    """
    result_files = []
    
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('_results.json') or file == 'kare_debate_mortality_results.json':
                    result_files.append(os.path.join(root, file))
    
    return result_files

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Fix ground truth labels in KARE debate result files")
    parser.add_argument('--test_data', type=str, 
                       default='data/ehr_data/mimic3_mortality_samples_test.json',
                       help='Path to MIMIC-III test data file')
    parser.add_argument('--result_file', type=str, default=None,
                       help='Specific result file to fix (if not provided, will find all)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--no_backup', action='store_true',
                       help='Skip creating backup files')
    
    args = parser.parse_args()
    
    print("KARE Ground Truth Label Remedy Script")
    print("=" * 50)
    
    # Load correct labels
    try:
        correct_labels = load_correct_labels(args.test_data)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Find result files to fix
    if args.result_file:
        result_files = [args.result_file] if os.path.exists(args.result_file) else []
    else:
        result_files = find_result_files(args.results_dir)
    
    if not result_files:
        print("No result files found to fix.")
        return
    
    print(f"\nFound {len(result_files)} result files to fix:")
    for f in result_files:
        print(f"  - {f}")
    
    # Fix each result file
    for result_file in result_files:
        try:
            fix_result_file(result_file, correct_labels, backup=not args.no_backup)
        except Exception as e:
            print(f"Error fixing {result_file}: {e}")
    
    print("\nAll result files have been processed!")

if __name__ == "__main__":
    main()