#!/usr/bin/env python3
"""
Test run script for ablation experiments.

Runs 3 samples for each condition (A, B, C) to:
1. Verify cache and log structure
2. Test end-to-end pipeline
3. Estimate total cost before running full experiments

This script helps catch issues early without burning through API credits.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from gpt_utils import GPTClient, initialize_medrag


# OpenAI pricing (as of GPT-4 Turbo)
# https://openai.com/pricing
PRICING = {
    "gpt-4-turbo-preview": {
        "input": 0.01 / 1000,   # $0.01 per 1K input tokens
        "output": 0.03 / 1000   # $0.03 per 1K output tokens
    },
    "gpt-4": {
        "input": 0.03 / 1000,
        "output": 0.06 / 1000
    },
    "gpt-4o": {
        "input": 0.0025 / 1000,  # $2.50 per 1M input tokens
        "output": 0.01 / 1000    # $10.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015 / 1000,  # $0.150 per 1M input tokens
        "output": 0.0006 / 1000   # $0.600 per 1M output tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.0005 / 1000,
        "output": 0.0015 / 1000
    },
    "o3-mini": {
        "input": 0.01 / 1000,   # Placeholder - actual pricing TBD
        "output": 0.03 / 1000
    }
}


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (1 token ≈ 4 characters for English).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def test_condition_a(
    gpt_client: GPTClient,
    samples: List[Dict[str, Any]],
    medrag,
    k: int = 8,
    model_name: str = "gpt-4-turbo-preview"
) -> Dict[str, Any]:
    """
    Test Condition A on 4 samples.
    
    Returns:
        Dictionary with results and cost estimate
    """
    from run_condition_A import process_sample_condition_a
    
    total_input_tokens = 0
    total_output_tokens = 0
    results = []
    
    # Create test output directory
    model_safe_name = model_name.replace('-', '_').replace('.', '_')
    output_dir = Path(__file__).parent.parent / "results" / f"test_condition_A_{model_safe_name}"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("TESTING CONDITION A: GPT+GPT+GPT (Ceiling)")
    print("="*80)
    
    # Also create cache directory for Conditions B & C to use
    cache_dir = Path(__file__).parent.parent / "cache" / "condition_A"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nProcessing sample {i}/{len(samples)}: {sample['sample_id']}")
        
        try:
            result = process_sample_condition_a(sample, gpt_client, medrag, k)
            
            # Save to logs directory (for test results)
            log_file = logs_dir / f"{sample['sample_id']}.json"
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # ALSO save to cache/condition_A/ for Conditions B & C
            cache_file = cache_dir / f"{sample['sample_id']}.json"
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Estimate tokens from result
            if 'gpt_analyst1' in result and result['gpt_analyst1']:
                total_output_tokens += estimate_tokens(result['gpt_analyst1'])
            if 'gpt_analyst2' in result and result['gpt_analyst2']:
                total_output_tokens += estimate_tokens(result['gpt_analyst2'])
            if 'gpt_integrator_initial' in result and result['gpt_integrator_initial']:
                total_output_tokens += estimate_tokens(result['gpt_integrator_initial'])
            if 'gpt_integrator_final' in result and result['gpt_integrator_final']:
                total_output_tokens += estimate_tokens(result['gpt_integrator_final'])
            
            # Input tokens (rough estimate: context + prompts)
            # 2 analyst calls (~2K tokens each) + 2 integrator calls (~3K tokens each)
            total_input_tokens += 2 * 2000 + 2 * 3000
            
            results.append(result)
            
            print(f"  ✓ Prediction: {result.get('prediction', 'N/A')}")
            print(f"  ✓ Mortality Prob: {result.get('mortality_probability', 'N/A')}")
            print(f"  ✓ Retrieval used: {result.get('called_retriever', False)}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({'error': str(e), 'sample_id': sample['sample_id']})
    
    # Save test results.json
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
    
    test_summary = {
        'condition': 'A',
        'model': model_name,
        'test_run': True,
        'total_samples': len(results),
        'successful': sum(1 for r in results if r.get('prediction') is not None),
        'errors': sum(1 for r in results if 'error' in r and r['error']),
        'results': compact_results
    }
    with open(results_file, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Check output directory
    print(f"\n✓ Output directory: {output_dir}")
    print(f"  Log files created: {len(list(logs_dir.glob('*.json')))}")
    print(f"  Results file: {results_file.exists()}")
    
    return {
        'condition': 'A',
        'results': results,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }


def test_condition_b(
    gpt_client: GPTClient,
    samples: List[Dict[str, Any]],
    qwen_log_dir: str
) -> Dict[str, Any]:
    """
    Test Condition B on 4 samples.
    
    Returns:
        Dictionary with results and cost estimate
    """
    from run_condition_B import process_sample_condition_b
    
    total_input_tokens = 0
    total_output_tokens = 0
    results = []
    
    print("\n" + "="*80)
    print("TESTING CONDITION B: GPT+Qwen+GPT (Retrieval Test)")
    print("="*80)
    
    # Check if Condition A outputs exist
    condition_a_dir = Path(__file__).parent.parent / "cache" / "condition_A"
    if not condition_a_dir.exists() or not list(condition_a_dir.glob('*.json')):
        print("\n⚠ WARNING: Condition A outputs not found. Run Condition A first!")
        return {
            'condition': 'B',
            'error': 'Condition A required',
            'input_tokens': 0,
            'output_tokens': 0
        }
    
    # Create test output directory
    output_dir = Path(__file__).parent.parent / "results" / "test_condition_B"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create cache directory for future use
    cache_dir = Path(__file__).parent.parent / "cache" / "condition_B"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nProcessing sample {i}/{len(samples)}: {sample['sample_id']}")
        
        try:
            result = process_sample_condition_b(sample, gpt_client, condition_a_dir, qwen_log_dir)
            
            # Save to logs directory
            log_file = logs_dir / f"{sample['sample_id']}.json"
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Also save to cache
            cache_file = cache_dir / f"{sample['sample_id']}.json"
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Estimate tokens
            if 'gpt_integrator_final' in result and result['gpt_integrator_final']:
                total_output_tokens += estimate_tokens(result['gpt_integrator_final'])
            
            # Input tokens (1 integrator call only - we don't generate query, just use Qwen's)
            total_input_tokens += 3000
            
            results.append(result)
            
            print(f"  ✓ Prediction: {result.get('prediction', 'N/A')}")
            print(f"  ✓ Mortality Prob: {result.get('mortality_probability', 'N/A')}")
            print(f"  ✓ Qwen retrieval loaded: {result.get('called_retriever', False)}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({'error': str(e), 'sample_id': sample['sample_id']})
    
    # Save test results.json
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
    
    test_summary = {
        'condition': 'B',
        'test_run': True,
        'total_samples': len(results),
        'successful': sum(1 for r in results if r.get('prediction') is not None),
        'errors': sum(1 for r in results if 'error' in r and r['error']),
        'results': compact_results
    }
    with open(results_file, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Check output directory
    print(f"\n✓ Output directory: {output_dir}")
    print(f"  Log files created: {len(list(logs_dir.glob('*.json')))}")
    print(f"  Results file: {results_file.exists()}")
    
    return {
        'condition': 'B',
        'results': results,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }


def test_condition_c(
    gpt_client: GPTClient,
    samples: List[Dict[str, Any]],
    qwen_log_dir: str
) -> Dict[str, Any]:
    """
    Test Condition C on 4 samples.
    
    Returns:
        Dictionary with results and cost estimate
    """
    from run_condition_C import process_sample_condition_c
    
    total_input_tokens = 0
    total_output_tokens = 0
    results = []
    
    print("\n" + "="*80)
    print("TESTING CONDITION C: Qwen+GPT+GPT (Analyst Test)")
    print("="*80)
    
    # Check if Condition A outputs exist
    condition_a_dir = Path(__file__).parent.parent / "cache" / "condition_A"
    if not condition_a_dir.exists() or not list(condition_a_dir.glob('*.json')):
        print("\n⚠ WARNING: Condition A outputs not found. Run Condition A first!")
        return {
            'condition': 'C',
            'error': 'Condition A required',
            'input_tokens': 0,
            'output_tokens': 0
        }
    
    # Create test output directory
    output_dir = Path(__file__).parent.parent / "results" / "test_condition_C"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create cache directory for future use
    cache_dir = Path(__file__).parent.parent / "cache" / "condition_C"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nProcessing sample {i}/{len(samples)}: {sample['sample_id']}")
        
        try:
            result = process_sample_condition_c(sample, gpt_client, condition_a_dir, qwen_log_dir)
            
            # Save to logs directory
            log_file = logs_dir / f"{sample['sample_id']}.json"
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Also save to cache
            cache_file = cache_dir / f"{sample['sample_id']}.json"
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Estimate tokens
            if 'gpt_integrator_initial' in result and result['gpt_integrator_initial']:
                total_output_tokens += estimate_tokens(result['gpt_integrator_initial'])
            if 'gpt_integrator_final' in result and result['gpt_integrator_final']:
                total_output_tokens += estimate_tokens(result['gpt_integrator_final'])
            
            # Input tokens (2 integrator calls, reuses GPT retrieval)
            total_input_tokens += 2 * 3000
            
            results.append(result)
            
            print(f"  ✓ Prediction: {result.get('prediction', 'N/A')}")
            print(f"  ✓ Mortality Prob: {result.get('mortality_probability', 'N/A')}")
            print(f"  ✓ Qwen analysts loaded: {result.get('qwen_analysts_loaded', False)}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({'error': str(e), 'sample_id': sample['sample_id']})
    
    # Save test results.json
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
            'qwen_analysts_loaded': r.get('qwen_analysts_loaded', False),
            'error': r.get('error')
        })
    
    test_summary = {
        'condition': 'C',
        'test_run': True,
        'total_samples': len(results),
        'successful': sum(1 for r in results if r.get('prediction') is not None),
        'errors': sum(1 for r in results if 'error' in r and r['error']),
        'results': compact_results
    }
    with open(results_file, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Check output directory
    print(f"\n✓ Output directory: {output_dir}")
    print(f"  Log files created: {len(list(logs_dir.glob('*.json')))}")
    print(f"  Results file: {results_file.exists()}")
    
    return {
        'condition': 'C',
        'results': results,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }


def print_cost_estimate(test_results: List[Dict[str, Any]], model: str, num_samples: int = 100):
    """
    Print cost estimate for full experiment.
    
    Args:
        test_results: List of test results from each condition
        model: GPT model name
        num_samples: Total number of samples in full experiment
    """
    print("\n" + "="*80)
    print("COST ESTIMATE")
    print("="*80)
    
    if model not in PRICING:
        print(f"⚠ Pricing not available for {model}, using gpt-4-turbo-preview rates")
        model = "gpt-4-turbo-preview"
    
    pricing = PRICING[model]
    
    # Calculate test costs
    test_cost_total = 0
    for result in test_results:
        if 'error' in result and not result.get('results'):
            continue
            
        input_tokens = result.get('input_tokens', 0)
        output_tokens = result.get('output_tokens', 0)
        
        cost = (input_tokens * pricing['input']) + (output_tokens * pricing['output'])
        test_cost_total += cost
        
        print(f"\nCondition {result['condition']} (3 samples):")
        print(f"  Input tokens:  {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Cost: ${cost:.4f}")
    
    print(f"\n{'─'*80}")
    print(f"Total test cost (4 samples × conditions): ${test_cost_total:.4f}")
    
    # Extrapolate to full experiment
    if test_cost_total > 0:
        scale_factor = num_samples / 4  # 100 samples vs 4 test samples
        
        print(f"\n{'─'*80}")
        print(f"ESTIMATED FULL EXPERIMENT COST ({num_samples} samples):")
        print(f"{'─'*80}")
        
        for result in test_results:
            if 'error' in result and not result.get('results'):
                continue
                
            input_tokens = result.get('input_tokens', 0)
            output_tokens = result.get('output_tokens', 0)
            
            full_input = input_tokens * scale_factor
            full_output = output_tokens * scale_factor
            full_cost = (full_input * pricing['input']) + (full_output * pricing['output'])
            
            print(f"\nCondition {result['condition']}:")
            print(f"  Estimated input tokens:  {full_input:,.0f}")
            print(f"  Estimated output tokens: {full_output:,.0f}")
            print(f"  Estimated cost: ${full_cost:.2f}")
        
        total_full_cost = test_cost_total * scale_factor
        print(f"\n{'─'*80}")
        print(f"TOTAL ESTIMATED COST: ${total_full_cost:.2f}")
        print(f"{'─'*80}")
        
        # Add safety margin warning
        margin_cost = total_full_cost * 1.5
        print(f"\n⚠ RECOMMENDED BUDGET (with 50% safety margin): ${margin_cost:.2f}")
        print(f"  (Actual costs may vary based on response length and retrieval usage)")


def main():
    parser = argparse.ArgumentParser(description="Test ablation experiments on 3 samples")
    parser.add_argument("--gpt_model", type=str, default="gpt-4-turbo-preview",
                       help="GPT model to use")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--k", type=int, default=8,
                       help="Number of documents to retrieve")
    parser.add_argument("--retriever_device", type=str, default="cuda:0",
                       help="Device for MedRAG retriever (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--qwen_log_dir", type=str,
                       default="/data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs",
                       help="Directory containing Qwen debate logs")
    parser.add_argument("--condition", type=str, choices=['A', 'B', 'C', 'all'], default='all',
                       help="Which condition to test (default: all)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Total samples in full experiment (for cost estimation)")
    
    args = parser.parse_args()
    
    # Initialize GPT client
    print("Initializing GPT client...")
    gpt_client = GPTClient(api_key=args.api_key, model=args.gpt_model)
    
    # Load test samples (first 3 from selected samples)
    manifest_path = Path(__file__).parent.parent / "manifests" / "selected_samples_full.parquet"
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run extract_metadata.py and sample_select.py first!")
        return 1
    
    print(f"Loading test samples from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    
    # Select 4 diverse samples: 2 positive (mortality=1) and 2 negative (survival=0)
    test_samples = []
    
    # Select 2 positive samples from different difficulty tiers
    positive_df = df[df['label'] == 1]
    for split_tag in ['pos_all']:  # All positives have same tag
        candidates = positive_df[positive_df['split_tag'] == split_tag]
        if len(candidates) >= 2:
            # Take first 2 positive samples
            test_samples.extend([candidates.iloc[0].to_dict(), candidates.iloc[1].to_dict()])
            break
    
    # Select 2 negative samples from different difficulty tiers
    negative_df = df[df['label'] == 0]
    for split_tag in ['neg_wrong2', 'neg_wrong1']:  # Moderately hard and edge cases
        candidates = negative_df[negative_df['split_tag'] == split_tag]
        if len(candidates) > 0:
            test_samples.append(candidates.iloc[0].to_dict())
        if len(test_samples) >= 4:
            break
    
    # If we don't have 4 yet, fill with any remaining samples
    if len(test_samples) < 4:
        remaining = df[~df['sample_id'].isin([s['sample_id'] for s in test_samples])]
        for i in range(min(4 - len(test_samples), len(remaining))):
            test_samples.append(remaining.iloc[i].to_dict())
    
    print(f"\nTest samples selected (2 positive, 2 negative):")
    for i, sample in enumerate(test_samples, 1):
        split_tag = sample.get('split_tag', 'N/A')
        label_str = 'mortality' if sample['label'] == 1 else 'survival'
        print(f"  {i}. {sample['sample_id']} (label={sample['label']} [{label_str}], split_tag={split_tag})")
    
    # Initialize MedRAG (only needed for Condition A)
    medrag = None
    if args.condition in ['A', 'all']:
        print("\nInitializing MedRAG...")
        medrag = initialize_medrag(retriever_device=args.retriever_device)
    
    # Run tests
    test_results = []
    
    if args.condition in ['A', 'all']:
        result_a = test_condition_a(gpt_client, test_samples, medrag, args.k, args.gpt_model)
        test_results.append(result_a)
    
    if args.condition in ['B', 'all']:
        result_b = test_condition_b(gpt_client, test_samples, args.qwen_log_dir)
        test_results.append(result_b)
    
    if args.condition in ['C', 'all']:
        result_c = test_condition_c(gpt_client, test_samples, args.qwen_log_dir)
        test_results.append(result_c)
    
    # Print cost estimate
    print_cost_estimate(test_results, args.gpt_model, args.num_samples)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for result in test_results:
        condition = result['condition']
        
        if 'error' in result and not result.get('results'):
            print(f"\nCondition {condition}: ⚠ SKIPPED ({result.get('error', 'Unknown error')})")
            continue
        
        results = result.get('results', [])
        errors = [r for r in results if r.get('error') is not None]
        
        if errors:
            print(f"\nCondition {condition}: ✗ FAILED ({len(errors)}/{len(results)} errors)")
            for err in errors:
                print(f"  - {err.get('error', 'Unknown error')}")
            all_passed = False
        else:
            print(f"\nCondition {condition}: ✓ PASSED ({len(results)}/{len(results)} successful)")
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run full experiments.")
        print("\nNext steps:")
        print("  1. Review cache/ directory structure")
        print("  2. Check log files for expected format")
        print("  3. Verify cost estimate fits budget")
        print("  4. Run full experiments:")
        print("     python src/run_condition_A.py")
        print("     python src/run_condition_B.py")
        print("     python src/run_condition_C.py")
    else:
        print("\n✗ Some tests failed. Fix errors before running full experiments.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
