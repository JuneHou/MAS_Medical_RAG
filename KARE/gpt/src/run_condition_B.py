#!/usr/bin/env python3
"""
Condition B: GPT analysts + Qwen retrieval + GPT integrator

This condition isolates retrieval/query quality by swapping ONLY the retrieval component.

Flow:
1. Reuse GPT Analyst 1 & 2 outputs from Condition A
2. Run GPT Integrator (initial) - same as Condition A
3. Use QWEN retrieval bundle (query + docs) from existing Qwen RAG logs
4. Run GPT Integrator (final) with Qwen-retrieved evidence

This tests: Is Qwen's retrieval quality limiting performance?
If Condition B << Condition A → Qwen retrieval is problematic
If Condition B ≈ Condition A → Retrieval quality is not the issue
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from kare_data_adapter import KAREDataAdapter
from gpt_utils import (
    GPTClient, AGENT_PROMPTS, initialize_medrag, retrieve_documents,
    parse_tool_call, format_retrieved_docs, extract_probabilities,
    load_qwen_debate_history, run_gpt_integrator_no_search
)


def load_qwen_retrieval_bundle(
    patient_id: str,
    qwen_log_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Load Qwen retrieval bundle from existing logs.
    
    Args:
        patient_id: Patient ID
        qwen_log_dir: Directory containing Qwen debate logs
        
    Returns:
        Dictionary with query and docs, or None
    """
    # Try to load from debate history
    history = load_qwen_debate_history(patient_id, qwen_log_dir)
    
    if not history or not history.get('called_retriever'):
        return None
    
    return {
        'query': history.get('retrieval_query'),
        'docs_text': history.get('retrieval_docs'),
        'called_retriever': True
    }


def process_sample_condition_b(
    sample: Dict[str, Any],
    gpt_client: GPTClient,
    condition_a_dir: Path,
    qwen_log_dir: str
) -> Dict[str, Any]:
    """
    Process single sample for Condition B.
    
    Args:
        sample: Sample data
        gpt_client: GPT client
        condition_a_dir: Directory with Condition A results
        qwen_log_dir: Directory with Qwen debate logs
        
    Returns:
        Result dictionary
    """
    sample_id = sample['sample_id']
    target_context = sample['patient_context']
    
    print(f"\nProcessing {sample_id}...")
    
    result = {
        'sample_id': sample_id,
        'label': int(sample['label']),
        'gpt_analyst1': None,  # Reused from Condition A
        'gpt_analyst2': None,  # Reused from Condition A
        'qwen_query': None,
        'qwen_docs': None,
        'called_retriever': False,
        'gpt_integrator_final': None,
        'mortality_probability': None,
        'survival_probability': None,
        'prediction': None,
        'error': None
    }
    
    try:
        # Step 1: Load GPT analyst outputs from Condition A
        condition_a_file = condition_a_dir / "logs" / f"{sample_id}.json"
        if not condition_a_file.exists():
            raise FileNotFoundError(f"Condition A result not found: {condition_a_file}")
        
        with open(condition_a_file, 'r') as f:
            condition_a_result = json.load(f)
        
        analyst1_output = condition_a_result['gpt_analyst1']
        analyst2_output = condition_a_result['gpt_analyst2']
        
        result['gpt_analyst1'] = analyst1_output
        result['gpt_analyst2'] = analyst2_output
        
        print("  Loaded GPT analyst outputs from Condition A")
        
        # Step 2: Load Qwen retrieval bundle (we use Qwen's query, not GPT's)
        print("  Loading Qwen retrieval bundle...")
        qwen_bundle = load_qwen_retrieval_bundle(sample_id, qwen_log_dir)
        
        if qwen_bundle and qwen_bundle['called_retriever']:
            print(f"  Qwen retrieval found: {qwen_bundle['query'][:100] if qwen_bundle['query'] else 'No query'}...")
            result['qwen_query'] = qwen_bundle['query']
            result['qwen_docs'] = qwen_bundle['docs_text']
            result['called_retriever'] = True
            
            # Step 3: Run GPT Integrator with Qwen-retrieved evidence (no search capability)
            print("  Running GPT Integrator with Qwen evidence...")
            integrator_final = run_gpt_integrator_no_search(
                gpt_client, target_context, analyst1_output, analyst2_output,
                qwen_bundle['docs_text']
            )
            result['gpt_integrator_final'] = integrator_final
            
        else:
            # No Qwen retrieval available - run without retrieval (no search capability)
            print("  No Qwen retrieval available, running without retrieval")
            integrator_final = run_gpt_integrator_no_search(
                gpt_client, target_context, analyst1_output, analyst2_output,
                retrieved_docs=None
            )
            result['gpt_integrator_final'] = integrator_final
        
        # Step 4: Extract probabilities and prediction
        probs = extract_probabilities(integrator_final)
        result['mortality_probability'] = probs['mortality_probability']
        result['survival_probability'] = probs['survival_probability']
        result['prediction'] = probs['prediction']
        
        print(f"  Prediction: {result['prediction']} (mortality={result['mortality_probability']}, survival={result['survival_probability']})")
        
    except Exception as e:
        print(f"  Error: {e}")
        result['error'] = str(e)
    
    # Fallback: if prediction is None (parse failure or exception), use opposite of label
    if result['prediction'] is None:
        result['prediction'] = 1 - int(sample['label'])
        result['mortality_probability'] = 1.0 if result['prediction'] == 1 else 0.0
        result['survival_probability'] = 1.0 - result['mortality_probability']
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Condition B: GPT analysts + Qwen retrieval + GPT integrator")
    parser.add_argument('--manifest', type=str,
                       default='manifests/samples_swap_core.csv',
                       help='Path to sample manifest CSV')
    parser.add_argument('--full_data', type=str,
                       default='manifests/selected_samples_full.parquet',
                       help='Path to full sample data parquet')
    parser.add_argument('--condition_a_dir', type=str,
                       default=None,
                       help='Directory with Condition A results (default: results/condition_A_{model})')
    parser.add_argument('--qwen_log_dir', type=str,
                       default='/data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs',
                       help='Directory with Qwen debate logs')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory for results (default: results/condition_B_{model})')
    parser.add_argument('--gpt_model', type=str,
                       default='gpt-4o',
                       help='GPT model name')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Set default directories with model name if not specified
    model_safe_name = args.gpt_model.replace('-', '_').replace('.', '_')
    if args.output_dir is None:
        args.output_dir = f"results/condition_B_{model_safe_name}"
    if args.condition_a_dir is None:
        args.condition_a_dir = f"results/condition_A_{model_safe_name}"
    
    print("=" * 80)
    print("CONDITION B: GPT Analysts + Qwen Retrieval + GPT Integrator")
    print("=" * 80)
    print(f"Model: {args.gpt_model}")
    print(f"Output: {args.output_dir}")
    print(f"Condition A source: {args.condition_a_dir}")
    
    # Load samples
    print(f"\nLoading samples from {args.full_data}...")
    samples_df = pd.read_parquet(args.full_data)
    print(f"Loaded {len(samples_df)} samples")
    
    # Check Condition A directory
    condition_a_dir = Path(args.condition_a_dir)
    if not condition_a_dir.exists():
        print(f"\nERROR: Condition A directory not found: {condition_a_dir}")
        print("Please run run_condition_A.py first!")
        return
    
    # Check Qwen log directory
    if not Path(args.qwen_log_dir).exists():
        print(f"\nWARNING: Qwen log directory not found: {args.qwen_log_dir}")
        print("Retrieval bundles may not be available for some samples")
    
    # Initialize GPT client
    print(f"\nInitializing GPT client (model: {args.gpt_model})...")
    gpt_client = GPTClient(api_key=args.api_key, model=args.gpt_model)
    
    # Create output directory with logs subdirectory
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    results = []
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Processing samples"):
        sample = row.to_dict()
        
        # Check if already processed
        log_file = logs_dir / f"{sample['sample_id']}.json"
        if log_file.exists():
            print(f"\n  Skipping {sample['sample_id']} (already processed)")
            with open(log_file, 'r') as f:
                result = json.load(f)
            results.append(result)
            continue
        
        # Process sample
        result = process_sample_condition_b(
            sample, gpt_client, condition_a_dir, args.qwen_log_dir
        )
        
        # Save individual result to logs directory
        with open(log_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        results.append(result)
        
        # Rate limiting
        time.sleep(1)
    
    # Save summary as results.json in main directory
    results_file = output_dir / "results.json"
    
    # Create compact summary (don't duplicate verbose outputs from log files)
    compact_results = []
    for r in results:
        compact_results.append({
            'sample_id': r.get('sample_id'),
            'label': r.get('label'),
            'prediction': r.get('prediction'),
            'mortality_probability': r.get('mortality_probability'),
            'survival_probability': r.get('survival_probability'),
            'called_retriever': r.get('called_retriever', False),
            'error': r.get('error')
        })
    
    summary = {
        'condition': 'B',
        'model': args.gpt_model,
        'total_samples': len(results),
        'successful': sum(1 for r in results if r['prediction'] is not None),
        'errors': sum(1 for r in results if r['error'] is not None),
        'called_retriever': sum(1 for r in results if r['called_retriever']),
        'accuracy': None,
        'results': compact_results
    }
    
    # Calculate accuracy
    valid_results = [r for r in results if r['prediction'] is not None]
    if valid_results:
        correct = sum(1 for r in valid_results if r['prediction'] == r['label'])
        summary['accuracy'] = correct / len(valid_results)
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("CONDITION B COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful']}")
    print(f"Errors: {summary['errors']}")
    print(f"Called retriever: {summary['called_retriever']}")
    print(f"Accuracy: {summary['accuracy']:.3f}" if summary['accuracy'] is not None else "Accuracy: N/A")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
