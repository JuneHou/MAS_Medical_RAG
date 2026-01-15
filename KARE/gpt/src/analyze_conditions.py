#!/usr/bin/env python3
"""
Analyze and compare results from Conditions A, B, and C.

This script loads results from all three conditions and baseline Qwen results,
then computes comparative metrics to answer:

1. Does GPT-4 improve performance? (Condition A vs Qwen baseline)
2. Is retrieval quality limiting performance? (Condition A vs Condition B)
3. Is analyst reasoning limiting performance? (Condition A vs Condition C)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_condition_results(condition_dir: Path) -> List[Dict[str, Any]]:
    """Load all results from a condition directory."""
    results = []
    
    if not condition_dir.exists():
        print(f"Warning: Condition directory not found: {condition_dir}")
        return results
    
    # First try to load comprehensive results.json
    results_json = condition_dir / "results.json"
    if results_json.exists():
        with open(results_json, 'r') as f:
            data = json.load(f)
            if 'results' in data:
                # Extract just the relevant fields
                for r in data['results']:
                    results.append({
                        'sample_id': r['sample_id'],
                        'label': r['label'],
                        'prediction': r['prediction'],
                        'error': r.get('error')
                    })
                return results
    
    # Fallback: load individual JSON files from cache
    for result_file in sorted(condition_dir.glob("*.json")):
        if result_file.name == "summary.json":
            continue
        
        with open(result_file, 'r') as f:
            result = json.load(f)
            results.append({
                'sample_id': result.get('sample_id'),
                'label': result.get('label'),
                'prediction': result.get('prediction'),
                'error': result.get('error')
            })
    
    return results


def calculate_metrics(results: List[Dict[str, Any]], condition_name: str) -> Dict[str, Any]:
    """Calculate performance metrics for a condition."""
    total = len(results)
    valid = [r for r in results if r.get('prediction') is not None]
    errors = [r for r in results if r.get('error') is not None]
    
    if not valid:
        return {
            'condition': condition_name,
            'total_samples': total,
            'valid_predictions': 0,
            'errors': len(errors),
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None
        }
    
    # Calculate basic metrics
    correct = sum(1 for r in valid if r['prediction'] == r['label'])
    accuracy = correct / len(valid)
    
    # Calculate confusion matrix
    tp = sum(1 for r in valid if r['prediction'] == 1 and r['label'] == 1)
    fp = sum(1 for r in valid if r['prediction'] == 1 and r['label'] == 0)
    fn = sum(1 for r in valid if r['prediction'] == 0 and r['label'] == 1)
    tn = sum(1 for r in valid if r['prediction'] == 0 and r['label'] == 0)
    
    # Precision, recall, F1 for mortality class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'condition': condition_name,
        'total_samples': total,
        'valid_predictions': len(valid),
        'errors': len(errors),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def compare_conditions(
    baseline_results: List[Dict[str, Any]],
    condition_a_results: List[Dict[str, Any]],
    condition_b_results: List[Dict[str, Any]],
    condition_c_results: List[Dict[str, Any]],
    condition_d_results: List[Dict[str, Any]],
    split_tags: Dict[str, str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create comparison table of all conditions, overall and by sample type."""
    
    # Calculate overall metrics for each condition
    baseline_metrics = calculate_metrics(baseline_results, "Qwen Baseline")
    condition_a_metrics = calculate_metrics(condition_a_results, "Condition A (GPT+GPT+GPT)")
    condition_b_metrics = calculate_metrics(condition_b_results, "Condition B (GPT+Qwen+GPT)")
    condition_c_metrics = calculate_metrics(condition_c_results, "Condition C (Qwen+GPT+GPT)")
    condition_d_metrics = calculate_metrics(condition_d_results, "Condition D (GPT+GPT+Qwen)")
    
    # Create overall DataFrame
    overall_df = pd.DataFrame([
        baseline_metrics,
        condition_a_metrics,
        condition_b_metrics,
        condition_c_metrics,
        condition_d_metrics
    ])
    
    # If split_tags provided, calculate metrics by sample type
    by_type_rows = []
    if split_tags:
        for split_tag in ['pos_all', 'neg_hard', 'neg_easy']:
            # Filter results by split_tag
            sample_ids = [sid for sid, tag in split_tags.items() if tag == split_tag]
            
            baseline_filtered = [r for r in baseline_results if r['sample_id'] in sample_ids]
            cond_a_filtered = [r for r in condition_a_results if r['sample_id'] in sample_ids]
            cond_b_filtered = [r for r in condition_b_results if r['sample_id'] in sample_ids]
            cond_c_filtered = [r for r in condition_c_results if r['sample_id'] in sample_ids]
            cond_d_filtered = [r for r in condition_d_results if r['sample_id'] in sample_ids]
            
            # Calculate metrics
            baseline_m = calculate_metrics(baseline_filtered, f"Qwen [{split_tag}]")
            cond_a_m = calculate_metrics(cond_a_filtered, f"Cond A [{split_tag}]")
            cond_b_m = calculate_metrics(cond_b_filtered, f"Cond B [{split_tag}]")
            cond_c_m = calculate_metrics(cond_c_filtered, f"Cond C [{split_tag}]")
            cond_d_m = calculate_metrics(cond_d_filtered, f"Cond D [{split_tag}]")
            
            by_type_rows.extend([baseline_m, cond_a_m, cond_b_m, cond_c_m, cond_d_m])
    
    by_type_df = pd.DataFrame(by_type_rows) if by_type_rows else None
    
    return overall_df, by_type_df


def analyze_improvements(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze improvements and answer research questions."""
    
    baseline_acc = comparison_df[comparison_df['condition'] == 'Qwen Baseline']['accuracy'].values[0]
    cond_a_acc = comparison_df[comparison_df['condition'] == 'Condition A (GPT+GPT+GPT)']['accuracy'].values[0]
    cond_b_acc = comparison_df[comparison_df['condition'] == 'Condition B (GPT+Qwen+GPT)']['accuracy'].values[0]
    cond_c_acc = comparison_df[comparison_df['condition'] == 'Condition C (Qwen+GPT+GPT)']['accuracy'].values[0]
    cond_d_acc = comparison_df[comparison_df['condition'] == 'Condition D (GPT+GPT+Qwen)']['accuracy'].values[0]
    
    analysis = {
        'q1_gpt_improvement': {
            'question': 'Does GPT-4 improve over Qwen baseline?',
            'comparison': 'Condition A vs Qwen Baseline',
            'baseline_accuracy': baseline_acc,
            'condition_a_accuracy': cond_a_acc,
            'improvement': cond_a_acc - baseline_acc if (cond_a_acc is not None and baseline_acc is not None) else None,
            'interpretation': None
        },
        'q2_retrieval_quality': {
            'question': 'Is Qwen retrieval limiting performance?',
            'comparison': 'Condition A vs Condition B',
            'condition_a_accuracy': cond_a_acc,
            'condition_b_accuracy': cond_b_acc,
            'gap': cond_a_acc - cond_b_acc if (cond_a_acc is not None and cond_b_acc is not None) else None,
            'interpretation': None
        },
        'q3_analyst_reasoning': {
            'question': 'Is Qwen analyst reasoning limiting performance?',
            'comparison': 'Condition A vs Condition C',
            'condition_a_accuracy': cond_a_acc,
            'condition_c_accuracy': cond_c_acc,
            'gap': cond_a_acc - cond_c_acc if (cond_a_acc is not None and cond_c_acc is not None) else None,
            'interpretation': None
        },
        'q4_integrator_reasoning': {
            'question': 'Is Qwen integrator reasoning limiting performance?',
            'comparison': 'Condition A vs Condition D',
            'condition_a_accuracy': cond_a_acc,
            'condition_d_accuracy': cond_d_acc,
            'gap': cond_a_acc - cond_d_acc if (cond_a_acc is not None and cond_d_acc is not None) else None,
            'interpretation': None
        }
    }
    
    # Add interpretations
    if analysis['q1_gpt_improvement']['improvement'] is not None:
        improvement = analysis['q1_gpt_improvement']['improvement']
        if improvement > 0.05:
            analysis['q1_gpt_improvement']['interpretation'] = "GPT-4 significantly outperforms Qwen (>5% improvement)"
        elif improvement > 0:
            analysis['q1_gpt_improvement']['interpretation'] = "GPT-4 slightly outperforms Qwen"
        else:
            analysis['q1_gpt_improvement']['interpretation'] = "No improvement from GPT-4"
    
    if analysis['q2_retrieval_quality']['gap'] is not None:
        gap = analysis['q2_retrieval_quality']['gap']
        if gap > 0.05:
            analysis['q2_retrieval_quality']['interpretation'] = "Qwen retrieval significantly hurts performance (>5% gap)"
        elif gap > 0:
            analysis['q2_retrieval_quality']['interpretation'] = "Qwen retrieval slightly hurts performance"
        else:
            analysis['q2_retrieval_quality']['interpretation'] = "Retrieval quality is not the issue"
    
    if analysis['q3_analyst_reasoning']['gap'] is not None:
        gap = analysis['q3_analyst_reasoning']['gap']
        if gap > 0.05:
            analysis['q3_analyst_reasoning']['interpretation'] = "Qwen analysts significantly limit performance (>5% gap)"
        elif gap > 0:
            analysis['q3_analyst_reasoning']['interpretation'] = "Qwen analysts slightly limit performance"
        else:
            analysis['q3_analyst_reasoning']['interpretation'] = "Analyst reasoning is not the issue"
    
    if analysis['q4_integrator_reasoning']['gap'] is not None:
        gap = analysis['q4_integrator_reasoning']['gap']
        if gap > 0.05:
            analysis['q4_integrator_reasoning']['interpretation'] = "Qwen integrator significantly limits performance (>5% gap)"
        elif gap > 0:
            analysis['q4_integrator_reasoning']['interpretation'] = "Qwen integrator slightly limits performance"
        else:
            analysis['q4_integrator_reasoning']['interpretation'] = "Integrator reasoning is not the issue"
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare GPT ablation conditions")
    parser.add_argument('--baseline_manifest', type=str,
                       default='manifests/selected_samples_full.parquet',
                       help='Baseline results with Qwen predictions')
    parser.add_argument('--condition_a_dir', type=str,
                       default='results/condition_A_gpt_4o',
                       help='Condition A results directory')
    parser.add_argument('--condition_b_dir', type=str,
                       default='results/condition_B_gpt_4o',
                       help='Condition B results directory')
    parser.add_argument('--condition_c_dir', type=str,
                       default='results/condition_C_gpt_4o',
                       help='Condition C results directory')
    parser.add_argument('--condition_d_dir', type=str,
                       default='results/condition_D_qwen',
                       help='Condition D results directory')
    parser.add_argument('--output', type=str,
                       default='results/ablation_analysis.json',
                       help='Output file for analysis results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ABLATION EXPERIMENT ANALYSIS")
    print("=" * 80)
    
    # Load baseline (Qwen) results from selected samples
    print("\nLoading baseline Qwen results...")
    baseline_df = pd.read_parquet(args.baseline_manifest)
    baseline_results = []
    split_tags = {}  # Map sample_id to split_tag
    
    for _, row in baseline_df.iterrows():
        baseline_results.append({
            'sample_id': row['sample_id'],
            'label': int(row['label']),
            'prediction': int(row['pred_multi_rag_qwen']) if pd.notna(row['pred_multi_rag_qwen']) else None,
            'error': None
        })
        split_tags[row['sample_id']] = row['split_tag']
    
    print(f"Loaded {len(baseline_results)} baseline results")
    print(f"  - pos_all: {sum(1 for t in split_tags.values() if t == 'pos_all')}")
    print(f"  - neg_hard: {sum(1 for t in split_tags.values() if t == 'neg_hard')}")
    print(f"  - neg_easy: {sum(1 for t in split_tags.values() if t == 'neg_easy')}")
    
    # Load condition results
    print("\nLoading Condition A results...")
    condition_a_results = load_condition_results(Path(args.condition_a_dir))
    print(f"Loaded {len(condition_a_results)} Condition A results")
    
    print("\nLoading Condition B results...")
    condition_b_results = load_condition_results(Path(args.condition_b_dir))
    print(f"Loaded {len(condition_b_results)} Condition B results")
    
    print("\nLoading Condition C results...")
    condition_c_results = load_condition_results(Path(args.condition_c_dir))
    print(f"Loaded {len(condition_c_results)} Condition C results")
    
    print("\nLoading Condition D results...")
    condition_d_results = load_condition_results(Path(args.condition_d_dir))
    print(f"Loaded {len(condition_d_results)} Condition D results")
    
    # Compare conditions
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    overall_df, by_type_df = compare_conditions(
        baseline_results, condition_a_results, condition_b_results, condition_c_results,
        condition_d_results, split_tags=split_tags
    )
    
    print("\n" + overall_df.to_string(index=False))
    
    # Display by-type breakdown
    if by_type_df is not None:
        print("\n" + "=" * 80)
        print("PERFORMANCE BY SAMPLE TYPE")
        print("=" * 80)
        print("\n" + by_type_df.to_string(index=False))
    
    # Analyze improvements
    print("\n" + "=" * 80)
    print("RESEARCH QUESTIONS")
    print("=" * 80)
    
    analysis = analyze_improvements(overall_df)
    
    for key, result in analysis.items():
        print(f"\n{result['question']}")
        print(f"  Comparison: {result['comparison']}")
        for k, v in result.items():
            if k not in ['question', 'comparison', 'interpretation']:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        if result['interpretation']:
            print(f"  → {result['interpretation']}")
    
    # Save analysis
    output_data = {
        'overall_comparison': overall_df.to_dict(orient='records'),
        'by_type_comparison': by_type_df.to_dict(orient='records') if by_type_df is not None else None,
        'research_questions': analysis
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
