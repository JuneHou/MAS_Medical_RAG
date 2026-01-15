#!/usr/bin/env python3
"""
Condition A: GPT analysts + GPT retrieval + GPT integrator (Ceiling)

This is the "ceiling" condition where all components use GPT-4.
Tests the upper bound of performance with stronger reasoning.

Flow:
1. Run GPT Analyst 1 (mortality_risk_assessor) on target vs positive similar
2. Run GPT Analyst 2 (protective_factor_analyst) on target vs negative similar  
3. Run GPT Integrator with both analyst outputs
4. GPT Integrator generates <search> query
5. Retrieve documents using MedRAG with GPT query
6. GPT Integrator produces final prediction with retrieved evidence

All prompts are EXACTLY THE SAME as Qwen system for fair comparison.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from kare_data_adapter import KAREDataAdapter
from gpt_utils import (
    GPTClient, AGENT_PROMPTS, initialize_medrag, retrieve_documents,
    parse_tool_call, format_retrieved_docs, extract_probabilities
)


def run_gpt_analyst(
    gpt_client: GPTClient,
    role: str,
    target_context: str,
    similar_context: str
) -> str:
    """
    Run GPT analyst on target vs similar patient.
    
    Args:
        gpt_client: GPT client
        role: Agent role (mortality_risk_assessor or protective_factor_analyst)
        target_context: Target patient context
        similar_context: Similar patient context
        
    Returns:
        Analyst response
    """
    # Get system prompt - EXACT SAME as Qwen
    system_prompt = AGENT_PROMPTS[role]
    
    # Build prompt - EXACT SAME format as Qwen
    prompt = f"""{system_prompt}

## Target Patient:
{target_context}

## Similar Patient:
{similar_context}

Provide your clinical analysis and mortality risk assessment:"""
    
    # Generate response
    response = gpt_client.generate(prompt, max_tokens=2048, temperature=0.7)
    
    return response


def run_gpt_integrator_initial(
    gpt_client: GPTClient,
    target_context: str,
    analyst1_output: str,
    analyst2_output: str
) -> str:
    """
    Run GPT integrator initial turn (before retrieval).
    
    Args:
        gpt_client: GPT client
        target_context: Target patient context
        analyst1_output: Analyst 1 output
        analyst2_output: Analyst 2 output
        
    Returns:
        Integrator initial response (may contain <search> query)
    """
    # Get system prompt - EXACT SAME as Qwen
    system_prompt = AGENT_PROMPTS["balanced_clinical_integrator"]
    
    # Format previous analysis - EXACT SAME as Qwen (_prepare_integrator_history)
    history_text = f"""
## Previous Analysis:

### Analysis 1 (Comparison with mortality case):
{analyst1_output}

### Analysis 2 (Comparison with survival case):
{analyst2_output}

**Note:** The above analyses were conducted without knowledge of outcomes. Use these pattern comparisons to inform your assessment.
"""
    
    # Build prompt
    prompt = f"""{system_prompt}

## Target Patient:
{target_context}

{history_text}

Provide your clinical analysis and mortality risk assessment:"""
    
    # Generate response
    response = gpt_client.generate(prompt, max_tokens=2048, temperature=0.7)
    
    return response


def run_gpt_integrator_final(
    gpt_client: GPTClient,
    target_context: str,
    analyst1_output: str,
    analyst2_output: str,
    initial_response: str,
    retrieved_docs: str
) -> str:
    """
    Run GPT integrator final turn (with retrieved evidence).
    
    Args:
        gpt_client: GPT client
        target_context: Target patient context
        analyst1_output: Analyst 1 output
        analyst2_output: Analyst 2 output
        initial_response: Integrator's initial response
        retrieved_docs: Retrieved documents formatted as string
        
    Returns:
        Integrator final response with prediction
    """
    # Get system prompt
    system_prompt = AGENT_PROMPTS["balanced_clinical_integrator"]
    
    # Format previous analysis
    history_text = f"""
## Previous Analysis:

### Analysis 1 (Comparison with mortality case):
{analyst1_output}

### Analysis 2 (Comparison with survival case):
{analyst2_output}

**Note:** The above analyses were conducted without knowledge of outcomes. Use these pattern comparisons to inform your assessment.
"""
    
    # Add initial response if provided (Condition A), skip if None (Condition B)
    if initial_response:
        history_text += f"""
### Your Initial Analysis:
{initial_response}
"""
    
    # Build prompt with retrieved information - EXACT SAME format as Qwen
    prompt = f"""{system_prompt}

## Target Patient:
{target_context}

{history_text}

<information>
{retrieved_docs}
</information>

Now provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
    
    # Generate response
    response = gpt_client.generate(prompt, max_tokens=2048, temperature=0.7)
    
    return response


def process_sample_condition_a(
    sample: Dict[str, Any],
    gpt_client: GPTClient,
    medrag,
    k: int = 8
) -> Dict[str, Any]:
    """
    Process single sample for Condition A.
    
    Args:
        sample: Sample data
        gpt_client: GPT client
        medrag: MedRAG instance
        k: Number of documents to retrieve
        
    Returns:
        Result dictionary
    """
    sample_id = sample['sample_id']
    target_context = sample['patient_context']
    positive_similar = sample['positive_similars']
    negative_similar = sample['negative_similars']
    
    print(f"\nProcessing {sample_id}...")
    
    result = {
        'sample_id': sample_id,
        'label': int(sample['label']),
        'gpt_analyst1': None,
        'gpt_analyst2': None,
        'gpt_integrator_initial': None,
        'gpt_query': None,
        'gpt_docs': None,
        'called_retriever': False,
        'gpt_integrator_final': None,
        'mortality_probability': None,
        'survival_probability': None,
        'prediction': None,
        'error': None
    }
    
    try:
        # Step 1: Run Analyst 1 (mortality_risk_assessor) - Target vs Positive
        print("  Running Analyst 1 (mortality_risk_assessor)...")
        analyst1_output = run_gpt_analyst(
            gpt_client, "mortality_risk_assessor", target_context, positive_similar
        )
        result['gpt_analyst1'] = analyst1_output
        
        # Step 2: Run Analyst 2 (protective_factor_analyst) - Target vs Negative
        print("  Running Analyst 2 (protective_factor_analyst)...")
        analyst2_output = run_gpt_analyst(
            gpt_client, "protective_factor_analyst", target_context, negative_similar
        )
        result['gpt_analyst2'] = analyst2_output
        
        # Step 3: Run Integrator initial turn
        print("  Running Integrator (initial)...")
        integrator_initial = run_gpt_integrator_initial(
            gpt_client, target_context, analyst1_output, analyst2_output
        )
        result['gpt_integrator_initial'] = integrator_initial
        
        # Step 4: Parse tool call for retrieval
        search_query = parse_tool_call(integrator_initial)
        
        if search_query and medrag:
            print(f"  Retrieval query: {search_query[:100]}...")
            result['gpt_query'] = search_query
            result['called_retriever'] = True
            
            # Step 5: Retrieve documents
            print(f"  Retrieving {k} documents...")
            retrieved_docs = retrieve_documents(medrag, search_query, k=k)
            result['gpt_docs'] = retrieved_docs
            
            # Format documents
            docs_formatted = format_retrieved_docs(retrieved_docs)
            
            # Step 6: Run Integrator final turn with evidence
            print("  Running Integrator (final with evidence)...")
            integrator_final = run_gpt_integrator_final(
                gpt_client, target_context, analyst1_output, analyst2_output,
                integrator_initial, docs_formatted
            )
            result['gpt_integrator_final'] = integrator_final
            
        else:
            # No retrieval requested - use initial response as final
            print("  No retrieval requested")
            integrator_final = integrator_initial
            result['gpt_integrator_final'] = integrator_final
        
        # Step 7: Extract probabilities and prediction
        probs = extract_probabilities(integrator_final)
        result['mortality_probability'] = probs['mortality_probability']
        result['survival_probability'] = probs['survival_probability']
        result['prediction'] = probs['prediction']
        
        print(f"  Prediction: {result['prediction']} (mortality={result['mortality_probability']}, survival={result['survival_probability']})")
        
    except Exception as e:
        print(f"  Error: {e}")
        result['error'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Condition A: GPT analysts + GPT retrieval + GPT integrator")
    parser.add_argument('--manifest', type=str, 
                       default='manifests/samples_swap_core.csv',
                       help='Path to sample manifest CSV')
    parser.add_argument('--full_data', type=str,
                       default='manifests/selected_samples_full.parquet',
                       help='Path to full sample data parquet')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory for results (default: results/condition_A_{model})')
    parser.add_argument('--gpt_model', type=str,
                       default='gpt-4o',
                       help='GPT model name')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (if not set in OPENAI_API_KEY env var)')
    parser.add_argument('--k', type=int, default=8,
                       help='Number of documents to retrieve')
    parser.add_argument('--corpus_name', type=str, default='MedCorp2',
                       help='MedRAG corpus name')
    parser.add_argument('--retriever_name', type=str, default='MedCPT',
                       help='MedRAG retriever name')
    parser.add_argument('--retriever_device', type=str, default='cuda:0',
                       help='Device for MedRAG retriever (e.g., cuda:0, cuda:1, cpu)')
    
    args = parser.parse_args()
    
    # Set output directory with model name if not specified
    if args.output_dir is None:
        model_name = args.gpt_model.replace('-', '_').replace('.', '_')
        args.output_dir = f"results/condition_A_{model_name}"
    
    print("=" * 80)
    print("CONDITION A: GPT Analysts + GPT Retrieval + GPT Integrator")
    print("=" * 80)
    print(f"Model: {args.gpt_model}")
    print(f"Output: {args.output_dir}")
    
    # Load samples
    print(f"\nLoading samples from {args.full_data}...")
    samples_df = pd.read_parquet(args.full_data)
    print(f"Loaded {len(samples_df)} samples")
    
    # Initialize GPT client
    print(f"\nInitializing GPT client (model: {args.gpt_model})...")
    gpt_client = GPTClient(api_key=args.api_key, model=args.gpt_model)
    
    # Initialize MedRAG
    print(f"\nInitializing MedRAG (corpus: {args.corpus_name}, retriever: {args.retriever_name})...")
    medrag = initialize_medrag(
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        retriever_device=args.retriever_device
    )
    
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
        result = process_sample_condition_a(sample, gpt_client, medrag, k=args.k)
        
        # Save individual result to logs directory
        with open(log_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        results.append(result)
        
        # Rate limiting
        time.sleep(1)  # Be nice to the API
    
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
        'condition': 'A',
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
    print("CONDITION A COMPLETE")
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
