#!/usr/bin/env python3
"""
Utility script to compare VLLM and effGen results.
Helps identify any systematic differences between the two implementations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_metrics(vllm_data: Dict, effgen_data: Dict):
    """Compare metrics between VLLM and effGen."""
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    
    vllm_metrics = vllm_data.get('metrics', {})
    effgen_metrics = effgen_data.get('metrics', {})
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'macro_f1', 'specificity']
    
    print(f"\n{'Metric':<20} {'VLLM':<15} {'effGen':<15} {'Difference':<15}")
    print("-" * 65)
    
    for metric in metrics_to_compare:
        vllm_val = vllm_metrics.get(metric, 0.0)
        effgen_val = effgen_metrics.get(metric, 0.0)
        diff = effgen_val - vllm_val
        
        print(f"{metric:<20} {vllm_val:<15.4f} {effgen_val:<15.4f} {diff:+.4f}")
    
    # Confusion matrix
    print(f"\n{'Confusion Matrix':<20} {'VLLM':<15} {'effGen':<15} {'Difference':<15}")
    print("-" * 65)
    
    for cm_metric in ['tp', 'fp', 'fn', 'tn']:
        vllm_val = vllm_metrics.get(cm_metric, 0)
        effgen_val = effgen_metrics.get(cm_metric, 0)
        diff = effgen_val - vllm_val
        
        print(f"{cm_metric.upper():<20} {vllm_val:<15} {effgen_val:<15} {diff:+d}")


def compare_predictions(vllm_data: Dict, effgen_data: Dict):
    """Compare individual predictions between VLLM and effGen."""
    print("\n" + "="*80)
    print("PREDICTION COMPARISON")
    print("="*80)
    
    vllm_results = {r['patient_id']: r for r in vllm_data.get('results', [])}
    effgen_results = {r['patient_id']: r for r in effgen_data.get('results', [])}
    
    # Find common patients
    common_ids = set(vllm_results.keys()) & set(effgen_results.keys())
    
    if not common_ids:
        print("\nNo common patients found between VLLM and effGen results!")
        return
    
    print(f"\nComparing {len(common_ids)} common patients...")
    
    # Count agreements and disagreements
    agreements = 0
    disagreements = []
    
    for patient_id in sorted(common_ids):
        vllm_pred = vllm_results[patient_id].get('prediction')
        effgen_pred = effgen_results[patient_id].get('prediction')
        ground_truth = vllm_results[patient_id].get('ground_truth')
        
        if vllm_pred == effgen_pred:
            agreements += 1
        else:
            disagreements.append({
                'patient_id': patient_id,
                'ground_truth': ground_truth,
                'vllm_pred': vllm_pred,
                'effgen_pred': effgen_pred
            })
    
    print(f"\nAgreement Rate: {agreements}/{len(common_ids)} ({100*agreements/len(common_ids):.1f}%)")
    print(f"Disagreement Count: {len(disagreements)}")
    
    if disagreements:
        print(f"\nShowing first 10 disagreements:")
        print(f"{'Patient ID':<15} {'Ground Truth':<15} {'VLLM Pred':<15} {'effGen Pred':<15}")
        print("-" * 60)
        
        for item in disagreements[:10]:
            print(f"{item['patient_id']:<15} {item['ground_truth']:<15} {item['vllm_pred']:<15} {item['effgen_pred']:<15}")
        
        if len(disagreements) > 10:
            print(f"... and {len(disagreements) - 10} more disagreements")
    
    # Analyze disagreement patterns
    analyze_disagreement_patterns(disagreements, vllm_results, effgen_results)


def analyze_disagreement_patterns(disagreements: List[Dict], vllm_results: Dict, effgen_results: Dict):
    """Analyze patterns in disagreements."""
    if not disagreements:
        return
    
    print("\n" + "="*80)
    print("DISAGREEMENT PATTERN ANALYSIS")
    print("="*80)
    
    # Pattern 1: VLLM correct, effGen wrong
    vllm_correct = sum(1 for d in disagreements if d['vllm_pred'] == d['ground_truth'])
    
    # Pattern 2: effGen correct, VLLM wrong
    effgen_correct = sum(1 for d in disagreements if d['effgen_pred'] == d['ground_truth'])
    
    # Pattern 3: Both wrong
    both_wrong = len(disagreements) - vllm_correct - effgen_correct
    
    print(f"\nDisagreement Breakdown:")
    print(f"  VLLM correct, effGen wrong: {vllm_correct}")
    print(f"  effGen correct, VLLM wrong: {effgen_correct}")
    print(f"  Both wrong (but different): {both_wrong}")
    
    # Check if one consistently predicts higher/lower
    vllm_higher = sum(1 for d in disagreements if d['vllm_pred'] > d['effgen_pred'])
    effgen_higher = sum(1 for d in disagreements if d['effgen_pred'] > d['vllm_pred'])
    
    print(f"\nPrediction Bias:")
    print(f"  VLLM predicts mortality more often: {vllm_higher}/{len(disagreements)}")
    print(f"  effGen predicts mortality more often: {effgen_higher}/{len(disagreements)}")


def compare_runtimes(vllm_data: Dict, effgen_data: Dict):
    """Compare generation times between VLLM and effGen."""
    print("\n" + "="*80)
    print("RUNTIME COMPARISON")
    print("="*80)
    
    vllm_results = vllm_data.get('results', [])
    effgen_results = effgen_data.get('results', [])
    
    vllm_times = [r.get('total_generation_time', 0) for r in vllm_results if r.get('total_generation_time')]
    effgen_times = [r.get('total_generation_time', 0) for r in effgen_results if r.get('total_generation_time')]
    
    if vllm_times and effgen_times:
        vllm_avg = sum(vllm_times) / len(vllm_times)
        effgen_avg = sum(effgen_times) / len(effgen_times)
        
        print(f"\nAverage Generation Time:")
        print(f"  VLLM:   {vllm_avg:.2f}s per sample")
        print(f"  effGen: {effgen_avg:.2f}s per sample")
        print(f"  Ratio:  {effgen_avg/vllm_avg:.2f}x (effGen vs VLLM)")
        
        vllm_total = sum(vllm_times)
        effgen_total = sum(effgen_times)
        
        print(f"\nTotal Generation Time:")
        print(f"  VLLM:   {vllm_total:.2f}s ({vllm_total/60:.1f} min)")
        print(f"  effGen: {effgen_total:.2f}s ({effgen_total/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Compare VLLM and effGen results")
    parser.add_argument('--vllm', type=str, required=True, help='Path to VLLM results.json')
    parser.add_argument('--effgen', type=str, required=True, help='Path to effGen results.json')
    args = parser.parse_args()
    
    # Load data
    print("Loading results...")
    vllm_data = load_results(args.vllm)
    effgen_data = load_results(args.effgen)
    
    print(f"\nVLLM results: {len(vllm_data.get('results', []))} samples")
    print(f"effGen results: {len(effgen_data.get('results', []))} samples")
    
    # Compare metrics
    compare_metrics(vllm_data, effgen_data)
    
    # Compare predictions
    compare_predictions(vllm_data, effgen_data)
    
    # Compare runtimes
    compare_runtimes(vllm_data, effgen_data)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
