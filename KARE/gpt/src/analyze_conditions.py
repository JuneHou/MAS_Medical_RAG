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


def check_retriever_called_condition_a(sample_id: str, condition_a_logs_dir: Path) -> bool:
    """
    Check if retriever was called in Condition A by checking for gpt_query field.
    
    Args:
        sample_id: Sample ID (e.g., "10774_5")
        condition_a_logs_dir: Directory containing Condition A log files
        
    Returns:
        True if retriever was called (gpt_query exists and is not None), False otherwise
    """
    log_file = condition_a_logs_dir / f"{sample_id}.json"
    
    if not log_file.exists():
        return False
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
            # Check if gpt_query field exists and is not None/empty
            gpt_query = data.get('gpt_query')
            return gpt_query is not None and gpt_query != ""
    except Exception as e:
        print(f"Warning: Could not read log file {log_file}: {e}")
        return False


def check_retriever_called_baseline(patient_id: str, debate_logs_dir: Path) -> bool:
    """
    Check if retriever was called in baseline Qwen multi-agent debate.
    
    Based on retrieval_analysis.md findings:
    - Multi-agent RAG uses <search> tool calls by integrator
    - Retrieved information appears in <information> tags
    - Retrieval JSON files are saved separately
    
    Args:
        patient_id: Patient ID (e.g., "10117_0")
        debate_logs_dir: Directory containing debate log files
        
    Returns:
        True if retriever was called, False otherwise
    """
    # Check if retrieval JSON file exists (most reliable method)
    json_file = debate_logs_dir / f"retrieve_integrator_combined_{patient_id}.json"
    if json_file.exists():
        return True
    
    # Fallback: Check log file for <search> and <information> tags
    log_file = debate_logs_dir / f"debate_responses_{patient_id}.log"
    
    if not log_file.exists():
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check for both <search> tool call AND <information> response
            has_search_call = "<search>" in content
            has_info_response = "<information>" in content
            return has_search_call and has_info_response
    except Exception as e:
        print(f"Warning: Could not read log file {log_file}: {e}")
        return False


def compare_conditions(
    baseline_results: List[Dict[str, Any]],
    condition_a_results: List[Dict[str, Any]],
    condition_b_results: List[Dict[str, Any]],
    condition_c_results: List[Dict[str, Any]],
    condition_d_results: List[Dict[str, Any]],
    condition_e_results: List[Dict[str, Any]],
    condition_f_results: List[Dict[str, Any]],
    split_tags: Dict[str, str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create comparison table of all conditions, overall and by sample type."""
    
    # Calculate overall metrics for each condition
    baseline_metrics = calculate_metrics(baseline_results, "Qwen Baseline")
    condition_a_metrics = calculate_metrics(condition_a_results, "Condition A (GPT+GPT+GPT)")
    condition_b_metrics = calculate_metrics(condition_b_results, "Condition B (GPT+Qwen+GPT)")
    condition_c_metrics = calculate_metrics(condition_c_results, "Condition C (Qwen+GPT+GPT)")
    condition_d_metrics = calculate_metrics(condition_d_results, "Condition D (GPT+GPT+Qwen)")
    condition_e_metrics = calculate_metrics(condition_e_results, "Condition E (GPT+Qwen+Qwen)")
    condition_f_metrics = calculate_metrics(condition_f_results, "Condition F (Qwen+GPT+Qwen)")
    
    # Create overall DataFrame
    overall_df = pd.DataFrame([
        baseline_metrics,
        condition_a_metrics,
        condition_b_metrics,
        condition_c_metrics,
        condition_d_metrics,
        condition_e_metrics,
        condition_f_metrics
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
            cond_e_filtered = [r for r in condition_e_results if r['sample_id'] in sample_ids]
            cond_f_filtered = [r for r in condition_f_results if r['sample_id'] in sample_ids]
            
            # Calculate metrics
            baseline_m = calculate_metrics(baseline_filtered, f"Qwen [{split_tag}]")
            cond_a_m = calculate_metrics(cond_a_filtered, f"Cond A [{split_tag}]")
            cond_b_m = calculate_metrics(cond_b_filtered, f"Cond B [{split_tag}]")
            cond_c_m = calculate_metrics(cond_c_filtered, f"Cond C [{split_tag}]")
            cond_d_m = calculate_metrics(cond_d_filtered, f"Cond D [{split_tag}]")
            cond_e_m = calculate_metrics(cond_e_filtered, f"Cond E [{split_tag}]")
            cond_f_m = calculate_metrics(cond_f_filtered, f"Cond F [{split_tag}]")
            
            by_type_rows.extend([baseline_m, cond_a_m, cond_b_m, cond_c_m, cond_d_m, cond_e_m, cond_f_m])
    
    by_type_df = pd.DataFrame(by_type_rows) if by_type_rows else None
    
    return overall_df, by_type_df


def analyze_improvements(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze improvements and answer research questions."""
    
    baseline_acc = comparison_df[comparison_df['condition'] == 'Qwen Baseline']['accuracy'].values[0]
    cond_a_acc = comparison_df[comparison_df['condition'] == 'Condition A (GPT+GPT+GPT)']['accuracy'].values[0]
    cond_b_acc = comparison_df[comparison_df['condition'] == 'Condition B (GPT+Qwen+GPT)']['accuracy'].values[0]
    cond_c_acc = comparison_df[comparison_df['condition'] == 'Condition C (Qwen+GPT+GPT)']['accuracy'].values[0]
    cond_d_acc = comparison_df[comparison_df['condition'] == 'Condition D (GPT+GPT+Qwen)']['accuracy'].values[0]
    cond_e_acc = comparison_df[comparison_df['condition'] == 'Condition E (GPT+Qwen+Qwen)']['accuracy'].values[0]
    cond_f_acc = comparison_df[comparison_df['condition'] == 'Condition F (Qwen+GPT+Qwen)']['accuracy'].values[0]
    
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
        },
        'q5_retriever_under_qwen_integrator': {
            'question': 'Does GPT retrieval help when Qwen integrates?',
            'comparison': 'Condition D (GPT+GPT+Qwen) vs Condition E (GPT+Qwen+Qwen)',
            'condition_d_accuracy': cond_d_acc,
            'condition_e_accuracy': cond_e_acc,
            'gap': cond_d_acc - cond_e_acc if (cond_d_acc is not None and cond_e_acc is not None) else None,
            'interpretation': None
        },
        'q6_analyst_under_qwen_integrator': {
            'question': 'Do GPT analysts help when Qwen integrates?',
            'comparison': 'Condition D (GPT+GPT+Qwen) vs Condition F (Qwen+GPT+Qwen)',
            'condition_d_accuracy': cond_d_acc,
            'condition_f_accuracy': cond_f_acc,
            'gap': cond_d_acc - cond_f_acc if (cond_d_acc is not None and cond_f_acc is not None) else None,
            'interpretation': None
        },
        'q7_integrator_under_gpt_upstream': {
            'question': 'Does GPT integrator help when GPT handles upstream tasks?',
            'comparison': 'Condition A (GPT+GPT+GPT) vs Condition D (GPT+GPT+Qwen)',
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
    
    if analysis['q5_retriever_under_qwen_integrator']['gap'] is not None:
        gap = analysis['q5_retriever_under_qwen_integrator']['gap']
        if gap > 0.05:
            analysis['q5_retriever_under_qwen_integrator']['interpretation'] = "GPT retrieval significantly helps with Qwen integrator (>5% improvement)"
        elif gap > 0:
            analysis['q5_retriever_under_qwen_integrator']['interpretation'] = "GPT retrieval slightly helps with Qwen integrator"
        elif gap < -0.05:
            analysis['q5_retriever_under_qwen_integrator']['interpretation'] = "GPT retrieval significantly hurts with Qwen integrator (>5% degradation)"
        elif gap < 0:
            analysis['q5_retriever_under_qwen_integrator']['interpretation'] = "GPT retrieval slightly hurts with Qwen integrator"
        else:
            analysis['q5_retriever_under_qwen_integrator']['interpretation'] = "Retriever choice doesn't matter under Qwen integrator"
    
    if analysis['q6_analyst_under_qwen_integrator']['gap'] is not None:
        gap = analysis['q6_analyst_under_qwen_integrator']['gap']
        if gap > 0.05:
            analysis['q6_analyst_under_qwen_integrator']['interpretation'] = "GPT analysts significantly help with Qwen integrator (>5% improvement)"
        elif gap > 0:
            analysis['q6_analyst_under_qwen_integrator']['interpretation'] = "GPT analysts slightly help with Qwen integrator"
        elif gap < -0.05:
            analysis['q6_analyst_under_qwen_integrator']['interpretation'] = "GPT analysts significantly hurt with Qwen integrator (>5% degradation)"
        elif gap < 0:
            analysis['q6_analyst_under_qwen_integrator']['interpretation'] = "GPT analysts slightly hurt with Qwen integrator"
        else:
            analysis['q6_analyst_under_qwen_integrator']['interpretation'] = "Analyst choice doesn't matter under Qwen integrator"
    
    if analysis['q7_integrator_under_gpt_upstream']['gap'] is not None:
        gap = analysis['q7_integrator_under_gpt_upstream']['gap']
        if gap > 0.05:
            analysis['q7_integrator_under_gpt_upstream']['interpretation'] = "GPT integrator significantly helps with GPT upstream (>5% improvement)"
        elif gap > 0:
            analysis['q7_integrator_under_gpt_upstream']['interpretation'] = "GPT integrator slightly helps with GPT upstream"
        elif gap < -0.05:
            analysis['q7_integrator_under_gpt_upstream']['interpretation'] = "GPT integrator significantly hurts with GPT upstream (>5% degradation)"
        elif gap < 0:
            analysis['q7_integrator_under_gpt_upstream']['interpretation'] = "GPT integrator slightly hurts with GPT upstream"
        else:
            analysis['q7_integrator_under_gpt_upstream']['interpretation'] = "Integrator choice doesn't matter under GPT upstream"
    
    return analysis


def analyze_retrieval_usage(
    baseline_results: List[Dict[str, Any]],
    condition_a_results: List[Dict[str, Any]],
    baseline_debate_logs_dir: Path,
    condition_a_logs_dir: Path
) -> Dict[str, Any]:
    """
    Compare retrieval usage between Condition A and baseline Qwen.
    
    Args:
        baseline_results: Baseline Qwen results
        condition_a_results: Condition A results
        baseline_debate_logs_dir: Directory containing baseline debate logs
        condition_a_logs_dir: Directory containing Condition A logs
        
    Returns:
        Dictionary with retrieval usage statistics and comparison
    """
    print("\n" + "=" * 80)
    print("ANALYZING RETRIEVAL USAGE")
    print("=" * 80)
    
    # Check retrieval for baseline
    baseline_retrieval_count = 0
    baseline_sample_ids = [r['sample_id'] for r in baseline_results]
    
    for sample_id in baseline_sample_ids:
        if check_retriever_called_baseline(sample_id, baseline_debate_logs_dir):
            baseline_retrieval_count += 1
    
    baseline_retrieval_rate = baseline_retrieval_count / len(baseline_sample_ids) if baseline_sample_ids else 0
    
    # Check retrieval for Condition A
    condition_a_retrieval_count = 0
    condition_a_sample_ids = [r['sample_id'] for r in condition_a_results]
    
    for sample_id in condition_a_sample_ids:
        if check_retriever_called_condition_a(sample_id, condition_a_logs_dir):
            condition_a_retrieval_count += 1
    
    condition_a_retrieval_rate = condition_a_retrieval_count / len(condition_a_sample_ids) if condition_a_sample_ids else 0
    
    retrieval_analysis = {
        'baseline_qwen': {
            'total_samples': len(baseline_sample_ids),
            'retrieval_calls': baseline_retrieval_count,
            'retrieval_rate': baseline_retrieval_rate,
            'no_retrieval_calls': len(baseline_sample_ids) - baseline_retrieval_count
        },
        'condition_a_gpt': {
            'total_samples': len(condition_a_sample_ids),
            'retrieval_calls': condition_a_retrieval_count,
            'retrieval_rate': condition_a_retrieval_rate,
            'no_retrieval_calls': len(condition_a_sample_ids) - condition_a_retrieval_count
        },
        'comparison': {
            'retrieval_rate_difference': condition_a_retrieval_rate - baseline_retrieval_rate,
            'retrieval_count_difference': condition_a_retrieval_count - baseline_retrieval_count,
            'interpretation': None
        }
    }
    
    # Add interpretation
    rate_diff = retrieval_analysis['comparison']['retrieval_rate_difference']
    if rate_diff > 0.1:
        retrieval_analysis['comparison']['interpretation'] = f"GPT-4 uses retrieval significantly more ({rate_diff:.1%} more)"
    elif rate_diff > 0:
        retrieval_analysis['comparison']['interpretation'] = f"GPT-4 uses retrieval slightly more ({rate_diff:.1%} more)"
    elif rate_diff < -0.1:
        retrieval_analysis['comparison']['interpretation'] = f"GPT-4 uses retrieval significantly less ({-rate_diff:.1%} less)"
    elif rate_diff < 0:
        retrieval_analysis['comparison']['interpretation'] = f"GPT-4 uses retrieval slightly less ({-rate_diff:.1%} less)"
    else:
        retrieval_analysis['comparison']['interpretation'] = "Similar retrieval usage"
    
    return retrieval_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare GPT ablation conditions")
    parser.add_argument('--baseline_manifest', type=str,
                       default='manifests/selected_samples_full.parquet',
                       help='Baseline results with Qwen predictions')
    parser.add_argument('--condition_a_dir', type=str,
                       default='results_bias/condition_A_gpt_4o',
                       help='Condition A results directory')
    parser.add_argument('--condition_b_dir', type=str,
                       default='results_bias/condition_B_gpt_4o',
                       help='Condition B results directory')
    parser.add_argument('--condition_c_dir', type=str,
                       default='results_bias/condition_C_gpt_4o',
                       help='Condition C results directory')
    parser.add_argument('--condition_d_dir', type=str,
                       default='results_bias/condition_D_qwen',
                       help='Condition D results directory')
    parser.add_argument('--condition_e_dir', type=str,
                       default='results_bias/condition_E_gpt_qwen_qwen',
                       help='Condition E results directory')
    parser.add_argument('--condition_f_dir', type=str,
                       default='results_bias/condition_F_qwen_gpt_qwen',
                       help='Condition F results directory')
    parser.add_argument('--baseline_debate_logs_dir', type=str,
                       default='/data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs',
                       help='Baseline Qwen multi-agent debate logs directory')
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
    
    print("\nLoading Condition E results...")
    condition_e_results = load_condition_results(Path(args.condition_e_dir))
    print(f"Loaded {len(condition_e_results)} Condition E results")
    
    print("\nLoading Condition F results...")
    condition_f_results = load_condition_results(Path(args.condition_f_dir))
    print(f"Loaded {len(condition_f_results)} Condition F results")
    
    # Compare conditions
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    overall_df, by_type_df = compare_conditions(
        baseline_results, condition_a_results, condition_b_results, condition_c_results,
        condition_d_results, condition_e_results, condition_f_results, split_tags=split_tags
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
    
    # Analyze retrieval usage
    retrieval_analysis = analyze_retrieval_usage(
        baseline_results=baseline_results,
        condition_a_results=condition_a_results,
        baseline_debate_logs_dir=Path(args.baseline_debate_logs_dir),
        condition_a_logs_dir=Path(args.condition_a_dir) / "logs"
    )
    
    print("\n" + "=" * 80)
    print("RETRIEVAL USAGE COMPARISON")
    print("=" * 80)
    print(f"\nBaseline Qwen Multi-Agent RAG:")
    print(f"  Total samples: {retrieval_analysis['baseline_qwen']['total_samples']}")
    print(f"  Retrieval calls: {retrieval_analysis['baseline_qwen']['retrieval_calls']}")
    print(f"  Retrieval rate: {retrieval_analysis['baseline_qwen']['retrieval_rate']:.1%}")
    print(f"  No retrieval: {retrieval_analysis['baseline_qwen']['no_retrieval_calls']}")
    
    print(f"\nCondition A (GPT-4 Integrator):")
    print(f"  Total samples: {retrieval_analysis['condition_a_gpt']['total_samples']}")
    print(f"  Retrieval calls: {retrieval_analysis['condition_a_gpt']['retrieval_calls']}")
    print(f"  Retrieval rate: {retrieval_analysis['condition_a_gpt']['retrieval_rate']:.1%}")
    print(f"  No retrieval: {retrieval_analysis['condition_a_gpt']['no_retrieval_calls']}")
    
    print(f"\nComparison:")
    print(f"  Rate difference: {retrieval_analysis['comparison']['retrieval_rate_difference']:+.1%}")
    print(f"  Count difference: {retrieval_analysis['comparison']['retrieval_count_difference']:+d}")
    print(f"  → {retrieval_analysis['comparison']['interpretation']}")
    
    # Save analysis
    output_data = {
        'overall_comparison': overall_df.to_dict(orient='records'),
        'by_type_comparison': by_type_df.to_dict(orient='records') if by_type_df is not None else None,
        'research_questions': analysis,
        'retrieval_usage': retrieval_analysis
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
