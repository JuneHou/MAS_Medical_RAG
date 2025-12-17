#!/usr/bin/env python3
"""
Analyze debate log files to identify real predictions vs fallback predictions.
Compare with ground truth from the final results JSON file.
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def extract_patient_info_from_log(log_path: str) -> Dict:
    """Extract prediction information from a log file."""
    result = {
        'patient_id': None,
        'mortality_prob': None,
        'survival_prob': None,
        'manual_prediction': None,
        'has_valid_mortality': False,
        'has_valid_survival': False,
        'is_fallback': False,
        'extraction_success': False
    }
    
    # Get patient ID from filename
    filename = os.path.basename(log_path)
    match = re.search(r'debate_responses_(.+)\.log$', filename)
    if match:
        result['patient_id'] = match.group(1)
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract mortality probability - use LAST occurrence (after retry if applicable)
        mort_matches = re.findall(r'EXTRACTED MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)', content)
        if mort_matches:
            result['mortality_prob'] = float(mort_matches[-1])  # Use last match
            result['has_valid_mortality'] = True
        
        # Extract survival probability - use LAST occurrence (after retry if applicable)
        surv_matches = re.findall(r'EXTRACTED SURVIVAL PROBABILITY:\s*([0-9]*\.?[0-9]+)', content)
        if surv_matches:
            result['survival_prob'] = float(surv_matches[-1])  # Use last match
            result['has_valid_survival'] = True
        
        # Extract manual final prediction - use LAST occurrence (after retry if applicable)
        pred_matches = re.findall(r'MANUAL FINAL PREDICTION:\s*(\d+)', content)
        if pred_matches:
            result['manual_prediction'] = int(pred_matches[-1])  # Use last match
        
        # Check if extraction was successful
        if result['has_valid_mortality'] and result['has_valid_survival']:
            result['extraction_success'] = True
            # Check if it's a fallback (survival > mortality and predicted 0)
            if result['survival_prob'] > result['mortality_prob'] and result['manual_prediction'] == 0:
                # This is a REAL prediction based on probabilities
                result['is_fallback'] = False
            elif result['survival_prob'] < result['mortality_prob'] and result['manual_prediction'] == 1:
                # This is also a REAL prediction
                result['is_fallback'] = False
            else:
                # Check for edge cases or tie-breaking
                result['is_fallback'] = False
        else:
            # Missing probabilities - likely a fallback
            result['is_fallback'] = True
            
    except Exception as e:
        print(f"Error processing {log_path}: {e}")
    
    return result

def load_ground_truth(test_data_path: str) -> Dict:
    """Load ground truth from original test data file."""
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    ground_truth = {}
    for sample in test_data:
        patient_id = sample['patient_id']
        visit_id = sample['visit_id']
        key = f"{patient_id}_{visit_id}"
        ground_truth[key] = {
            'ground_truth': sample.get('label', sample.get('mortality')),
            'patient_id': patient_id,
            'visit_id': visit_id
        }
    
    return ground_truth

def analyze_predictions(log_dir: str, test_data_path: str, results_json_path: str = None):
    """Main analysis function."""
    
    print("="*80)
    print("ANALYZING DEBATE PREDICTIONS: REAL vs FALLBACK")
    print("="*80)
    print()
    
    # Load ground truth and predictions from results.json if available
    if results_json_path and os.path.exists(results_json_path):
        print(f"Loading predictions and ground truth from: {results_json_path}")
        with open(results_json_path, 'r') as f:
            results_data = json.load(f)
        
        # Create mapping of patient_id to prediction and ground truth
        final_predictions = {}
        for r in results_data['results']:
            final_predictions[r['patient_id']] = {
                'prediction': r['prediction'],
                'ground_truth': r['ground_truth']
            }
        print(f"Loaded {len(final_predictions)} predictions from results.json")
        print()
    else:
        # Fallback to test data for ground truth only
        print(f"Loading ground truth from: {test_data_path}")
        ground_truth = load_ground_truth(test_data_path)
        print(f"Loaded {len(ground_truth)} ground truth entries")
        final_predictions = None
        print()
    
    # Process all log files
    log_files = list(Path(log_dir).glob("debate_responses_*.log"))
    print(f"Found {len(log_files)} log files")
    print()
    
    results = []
    for log_path in sorted(log_files):
        result = extract_patient_info_from_log(str(log_path))
        if result['patient_id']:
            # Add ground truth and FINAL prediction from results.json
            if final_predictions and result['patient_id'] in final_predictions:
                result['ground_truth'] = final_predictions[result['patient_id']]['ground_truth']
                result['manual_prediction'] = final_predictions[result['patient_id']]['prediction']
            elif not final_predictions and result['patient_id'] in ground_truth:
                result['ground_truth'] = ground_truth[result['patient_id']]['ground_truth']
            results.append(result)
    
    print(f"Successfully processed {len(results)} log files")
    print()
    
    # Categorize predictions
    real_predictions = []
    fallback_predictions = []
    missing_mortality = []
    missing_survival = []
    
    for r in results:
        if r['extraction_success']:
            real_predictions.append(r)
        else:
            fallback_predictions.append(r)
            if not r['has_valid_mortality']:
                missing_mortality.append(r)
            if not r['has_valid_survival']:
                missing_survival.append(r)
    
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total samples: {len(results)}")
    print(f"Real predictions (both probabilities extracted): {len(real_predictions)}")
    print(f"Fallback predictions (missing probabilities): {len(fallback_predictions)}")
    print(f"  - Missing mortality probability: {len(missing_mortality)}")
    print(f"  - Missing survival probability: {len(missing_survival)}")
    print()
    
    # Analyze real predictions
    if real_predictions:
        print("="*80)
        print("REAL PREDICTIONS ANALYSIS")
        print("="*80)
        real_correct = sum(1 for r in real_predictions if r.get('ground_truth') == r.get('manual_prediction'))
        real_total = len([r for r in real_predictions if r.get('ground_truth') is not None])
        
        # Count by ground truth
        real_gt_0 = sum(1 for r in real_predictions if r.get('ground_truth') == 0)
        real_gt_1 = sum(1 for r in real_predictions if r.get('ground_truth') == 1)
        
        # Count predictions
        real_pred_0 = sum(1 for r in real_predictions if r.get('manual_prediction') == 0)
        real_pred_1 = sum(1 for r in real_predictions if r.get('manual_prediction') == 1)
        
        # Calculate confusion matrix for real predictions
        real_tp = sum(1 for r in real_predictions if r.get('ground_truth') == 1 and r.get('manual_prediction') == 1)
        real_fp = sum(1 for r in real_predictions if r.get('ground_truth') == 0 and r.get('manual_prediction') == 1)
        real_fn = sum(1 for r in real_predictions if r.get('ground_truth') == 1 and r.get('manual_prediction') == 0)
        real_tn = sum(1 for r in real_predictions if r.get('ground_truth') == 0 and r.get('manual_prediction') == 0)
        
        print(f"Total real predictions: {len(real_predictions)}")
        print(f"Correct predictions: {real_correct}/{real_total} ({100*real_correct/real_total:.2f}%)")
        print()
        print("Ground truth distribution:")
        print(f"  Survival (0): {real_gt_0}")
        print(f"  Mortality (1): {real_gt_1}")
        print()
        print("Prediction distribution:")
        print(f"  Survival (0): {real_pred_0}")
        print(f"  Mortality (1): {real_pred_1}")
        print()
        print("Confusion Matrix (Real Predictions):")
        print(f"  TP (pred=1, gt=1): {real_tp}")
        print(f"  FP (pred=1, gt=0): {real_fp}")
        print(f"  FN (pred=0, gt=1): {real_fn}")
        print(f"  TN (pred=0, gt=0): {real_tn}")
        print()
        
        # Calculate metrics for real predictions
        if real_tp + real_fp > 0:
            real_precision = real_tp / (real_tp + real_fp)
            print(f"Precision: {real_precision:.4f}")
        else:
            print("Precision: N/A (no positive predictions)")
        
        if real_tp + real_fn > 0:
            real_recall = real_tp / (real_tp + real_fn)
            real_sensitivity = real_recall  # Sensitivity = Recall = TP / (TP + FN)
            print(f"Recall/Sensitivity: {real_recall:.4f}")
        else:
            print("Recall/Sensitivity: N/A (no positive ground truth)")
        
        if real_tn + real_fp > 0:
            real_specificity = real_tn / (real_tn + real_fp)
            print(f"Specificity: {real_specificity:.4f}")
        else:
            print("Specificity: N/A (no negative ground truth)")
        print()
    
    # Analyze fallback predictions - show which probabilities are None
    if fallback_predictions:
        print("="*80)
        print("FALLBACK PREDICTIONS ANALYSIS - MISSING PROBABILITY BREAKDOWN")
        print("="*80)
        
        # Categorize by which probabilities are missing
        both_none = [r for r in fallback_predictions if not r['has_valid_mortality'] and not r['has_valid_survival']]
        only_mortality_none = [r for r in fallback_predictions if not r['has_valid_mortality'] and r['has_valid_survival']]
        only_survival_none = [r for r in fallback_predictions if r['has_valid_mortality'] and not r['has_valid_survival']]
        
        print(f"Total fallback predictions: {len(fallback_predictions)}")
        print()
        print("Missing Probability Breakdown:")
        print(f"  Both probabilities None: {len(both_none)} ({100*len(both_none)/len(fallback_predictions):.1f}%)")
        print(f"  Only Mortality None: {len(only_mortality_none)} ({100*len(only_mortality_none)/len(fallback_predictions):.1f}%)")
        print(f"  Only Survival None: {len(only_survival_none)} ({100*len(only_survival_none)/len(fallback_predictions):.1f}%)")
        print()

    
    # Show examples of fallback cases
    if fallback_predictions:
        print("="*80)
        print("SAMPLE FALLBACK CASES (First 10)")
        print("="*80)
        for i, r in enumerate(fallback_predictions[:10]):
            print(f"\n{i+1}. Patient ID: {r['patient_id']}")
            # print(f"   Ground Truth: {r.get('ground_truth', 'N/A')}")
            # print(f"   Prediction: {r.get('manual_prediction', 'N/A')}")
            # print(f"   Has Mortality Prob: {r['has_valid_mortality']} (value: {r['mortality_prob']})")
            # print(f"   Has Survival Prob: {r['has_valid_survival']} (value: {r['survival_prob']})")
            # print(f"   Correct: {'✓' if r.get('ground_truth') == r.get('manual_prediction') else '✗'}")
    
    # Save detailed results to CSV
    output_csv = log_dir + "/../prediction_analysis.csv"
    print()
    print("="*80)
    print(f"Saving detailed results to: {output_csv}")
    print("="*80)
    
    import csv
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['patient_id', 'ground_truth', 'manual_prediction', 'mortality_prob', 
                     'survival_prob', 'has_valid_mortality', 'has_valid_survival', 
                     'is_fallback', 'correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'patient_id': r['patient_id'],
                'ground_truth': r.get('ground_truth', ''),
                'manual_prediction': r.get('manual_prediction', ''),
                'mortality_prob': r['mortality_prob'] if r['mortality_prob'] is not None else '',
                'survival_prob': r['survival_prob'] if r['survival_prob'] is not None else '',
                'has_valid_mortality': r['has_valid_mortality'],
                'has_valid_survival': r['has_valid_survival'],
                'is_fallback': not r['extraction_success'],
                'correct': r.get('ground_truth') == r.get('manual_prediction') if r.get('ground_truth') is not None else ''
            })
    
    print(f"CSV file saved successfully!")
    print()
    
    # Final summary
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"✓ Real predictions with extracted probabilities: {len(real_predictions)} ({100*len(real_predictions)/len(results):.2f}%)")
    print(f"✗ Fallback predictions (missing probabilities): {len(fallback_predictions)} ({100*len(fallback_predictions)/len(results):.2f}%)")
    print()
    print("="*80)
    print()
    
    # Calculate adjusted metrics treating fallback as incorrect
    print("="*80)
    print("ADJUSTED METRICS (TREATING FALLBACK AS INCORRECT)")
    print("="*80)
    print()
    print("If we treat all fallback predictions as INCORRECT (since they didn't")
    print("successfully extract probabilities), the metrics would be:")
    print()
    
    total_samples = len(results)
    # Only count real predictions as potentially correct
    adjusted_correct = sum(1 for r in real_predictions if r.get('ground_truth') == r.get('manual_prediction'))
    # All samples contribute to total (including fallbacks)
    total_with_gt = len([r for r in results if r.get('ground_truth') is not None])
    
    adjusted_accuracy = adjusted_correct / total_with_gt if total_with_gt > 0 else 0
    
    print(f"Total samples: {total_with_gt}")
    print(f"Real predictions (with extracted probabilities): {len(real_predictions)}")
    print(f"  - Correct: {adjusted_correct}")
    print(f"  - Incorrect: {len(real_predictions) - adjusted_correct}")
    print(f"Fallback predictions (counted as incorrect): {len(fallback_predictions)}")
    print()
    print(f"ADJUSTED ACCURACY: {adjusted_correct}/{total_with_gt} = {100*adjusted_accuracy:.2f}%")
    print()
    
    # Calculate adjusted confusion matrix treating fallback predictions as incorrect
    # Only real predictions (with extracted probabilities) can contribute to TP/TN
    # Fallback predictions are always counted as errors (FP or FN depending on prediction)
    
    # Start with real predictions only
    adj_tp = sum(1 for r in real_predictions if r.get('ground_truth') == 1 and r.get('manual_prediction') == 1)
    adj_fp = sum(1 for r in real_predictions if r.get('ground_truth') == 0 and r.get('manual_prediction') == 1)
    adj_fn = sum(1 for r in real_predictions if r.get('ground_truth') == 1 and r.get('manual_prediction') == 0)
    adj_tn = sum(1 for r in real_predictions if r.get('ground_truth') == 0 and r.get('manual_prediction') == 0)
    
    # For fallback predictions, count them as errors:
    # - If fallback predicted 1 and gt=1, it's NOT a TP (no extracted probabilities) -> count as FN
    # - If fallback predicted 1 and gt=0, it's FP
    # - If fallback predicted 0 and gt=1, it's FN
    # - If fallback predicted 0 and gt=0, it's NOT a TN (no extracted probabilities) -> count as FP
    fallback_tp_count = sum(1 for r in fallback_predictions if r.get('ground_truth') == 1 and r.get('manual_prediction') == 1)
    fallback_fp_count = sum(1 for r in fallback_predictions if r.get('ground_truth') == 0 and r.get('manual_prediction') == 1)
    fallback_fn_count = sum(1 for r in fallback_predictions if r.get('ground_truth') == 1 and r.get('manual_prediction') == 0)
    fallback_tn_count = sum(1 for r in fallback_predictions if r.get('ground_truth') == 0 and r.get('manual_prediction') == 0)
    
    # Add fallback errors to confusion matrix:
    # Fallback "TP" (pred=1, gt=1) -> treat as FN (failed to extract probability)
    adj_fn += fallback_tp_count
    # Fallback "FP" (pred=1, gt=0) -> keep as FP
    adj_fp += fallback_fp_count
    # Fallback "FN" (pred=0, gt=1) -> keep as FN
    adj_fn += fallback_fn_count
    # Fallback "TN" (pred=0, gt=0) -> treat as FP (failed to extract probability)
    adj_fp += fallback_tn_count
    
    print("Adjusted Confusion Matrix (Fallback predictions treated as errors):")
    print(f"  TP (pred=1, gt=1, with valid probabilities): {adj_tp}")
    print(f"  FP (pred=1, gt=0 OR fallback with pred=0, gt=0): {adj_fp}")
    print(f"    └─ Real FP: {adj_fp - fallback_tn_count}, Fallback 'lucky guess' TN counted as FP: {fallback_tn_count}")
    print(f"  FN (pred=0, gt=1 OR fallback with pred=1, gt=1): {adj_fn}")
    print(f"    └─ Real FN: {adj_fn - fallback_tp_count}, Fallback 'lucky guess' TP counted as FN: {fallback_tp_count}")
    print(f"  TN (pred=0, gt=0, with valid probabilities): {adj_tn}")
    print()
    print(f"Note: Fallback predictions broken down:")
    print(f"  - Fallback pred=1, gt=1: {fallback_tp_count} (counted as FN)")
    print(f"  - Fallback pred=1, gt=0: {fallback_fp_count} (counted as FP)")
    print(f"  - Fallback pred=0, gt=1: {fallback_fn_count} (counted as FN)")
    print(f"  - Fallback pred=0, gt=0: {fallback_tn_count} (counted as FP)")
    print()
    
    # Calculate metrics based on adjusted confusion matrix (consistent with adjusted accuracy)
    if adj_tp + adj_fp > 0:
        adj_precision = adj_tp / (adj_tp + adj_fp)
        print(f"Precision (treating fallback as incorrect): {adj_precision:.4f}")
    else:
        print("Precision: N/A (no positive predictions)")
    
    if adj_tp + adj_fn > 0:
        adj_recall = adj_tp / (adj_tp + adj_fn)
        adj_sensitivity = adj_recall  # Sensitivity = Recall = TP / (TP + FN)
        print(f"Recall/Sensitivity (treating fallback as incorrect): {adj_recall:.4f}")
    else:
        print("Recall/Sensitivity: N/A (no positive ground truth)")
    
    if adj_tn + adj_fp > 0:
        adj_specificity = adj_tn / (adj_tn + adj_fp)
        print(f"Specificity (treating fallback as incorrect): {adj_specificity:.4f}")
    else:
        print("Specificity: N/A (no negative ground truth)")
    
    if (adj_tp + adj_fp > 0) and (adj_tp + adj_fn > 0):
        adj_f1 = 2 * (adj_precision * adj_recall) / (adj_precision + adj_recall)
        print(f"F1 Score: {adj_f1:.4f}")
    
    # Verify accuracy matches
    verify_accuracy = (adj_tp + adj_tn) / total_with_gt if total_with_gt > 0 else 0
    print()
    print(f"Verification: Accuracy from adjusted confusion matrix = {adj_tp + adj_tn}/{total_with_gt} = {100*verify_accuracy:.2f}%")
    print(f"              This should match ADJUSTED ACCURACY above: {100*adjusted_accuracy:.2f}%")
    print()

if __name__ == "__main__":
    # Paths
    log_dir = "/data/wang/junh/githubs/Debate/KARE/results/results_verl/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__projects_slmreasoning_junh_Debate_KARE_verl_models_format_enforcer_7b_step57_MedCPT_8_8/debate_logs"
    # Use original test data for ground truth
    test_data_path = "/data/wang/junh/githubs/Debate/KARE/data/ehr_data/mimic3_mortality_samples_test.json"
    # Use results.json for FINAL predictions (more reliable than log files which contain intermediate states)
    results_json_path = "/data/wang/junh/githubs/Debate/KARE/results/results_verl/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__projects_slmreasoning_junh_Debate_KARE_verl_models_format_enforcer_7b_step57_MedCPT_8_8/kare_debate_mortality_results.json"
    
    analyze_predictions(log_dir, test_data_path, results_json_path)
