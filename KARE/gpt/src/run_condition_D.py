#!/usr/bin/env python3
"""
Condition D: GPT analysts + GPT retrieval + Qwen integrator

This condition isolates integrator reasoning quality by swapping ONLY the integrator.

Flow:
1. Reuse GPT Analyst 1 & 2 outputs from Condition A
2. Reuse GPT retrieval (query + docs) from Condition A (forced retrieval)
3. Run QWEN Integrator with GPT analyst outputs and GPT-retrieved evidence

This tests: Is Qwen's integrator reasoning limiting performance?
If Condition D << Condition A → Qwen integrator reasoning is problematic
If Condition D ≈ Condition A → Integrator quality is not the limiting factor

Note: Uses same vLLM setup as mortality_single_agent_cot.py
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

# Add MedRAG/vLLM paths
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))

from vllm import LLM, SamplingParams
from kare_data_adapter import KAREDataAdapter
from gpt_utils import (
    AGENT_PROMPTS, extract_probabilities
)


class QwenIntegrator:
    """Qwen integrator wrapper - EXACT pattern from mortality_single_agent_cot.py"""
    
    def __init__(self, model_name: str, gpu_id: str):
        """Initialize Qwen integrator - EXACT copy from mortality_single_agent_cot.py"""
        self.model_name = model_name
        self.gpu_id = gpu_id
        
        print(f"Qwen integrator ({model_name}) will use GPU: {self.gpu_id}")
        
        # Only set CUDA_VISIBLE_DEVICES if not already set externally
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
            print(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id}")
        else:
            print(f"CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Initialize VLLM directly - EXACT as mortality_single_agent_cot.py
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enforce_eager=True
        )
        print(f"VLLM initialized for {model_name}")
    
    def __call__(self, prompt: str, max_tokens: int = 32768, temperature: float = 0.7) -> str:
        """Generate response - EXACT pattern from mortality_single_agent_cot.py"""
        # Format prompt for Qwen
        if "qwen" in self.model_name.lower():
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = prompt
        
        # Generate response
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "</s>"],
            repetition_penalty=1.2
        )
        
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        return response_text


# Global integrator instance
_qwen_integrator = None


def run_qwen_integrator(
    target_context: str,
    analyst1_output: str,
    analyst2_output: str,
    retrieved_docs: str,
    max_tokens: int = 32768,
    temperature: float = 0.7
) -> str:
    """
    Run Qwen integrator with GPT analyst outputs and GPT-retrieved evidence.
    
    Args:
        target_context: Target patient context
        analyst1_output: GPT Analyst 1 output
        analyst2_output: GPT Analyst 2 output
        retrieved_docs: GPT-retrieved documents (formatted string)
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        
    Returns:
        Qwen integrator response with probabilities
    """
    global _qwen_integrator
    
    if _qwen_integrator is None:
        raise RuntimeError("Qwen integrator not initialized")
    
    # Get system prompt WITHOUT search capability (same as GPT Conditions B/C)
    system_prompt = AGENT_PROMPTS["balanced_clinical_integrator_no_search"]
    
    # Format previous analysis - EXACT SAME format as GPT
    history_text = f"""
## Previous Analysis:

### Analysis 1 (Comparison with mortality case):
{analyst1_output}

### Analysis 2 (Comparison with survival case):
{analyst2_output}

**Note:** The above analyses were conducted without knowledge of outcomes. Use these pattern comparisons to inform your assessment.
"""
    
    # Build prompt - handle both with and without retrieval
    if retrieved_docs:
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
    else:
        prompt = f"""{system_prompt}

## Target Patient:
{target_context}

{history_text}

Provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
    
    # Call Qwen
    response = _qwen_integrator(prompt, max_tokens=max_tokens, temperature=temperature)
    
    # Extract string response
    if isinstance(response, list):
        response = response[0] if response else ""
        if isinstance(response, dict):
            response = response.get("generated_text", str(response))
    elif isinstance(response, dict):
        response = response.get("generated_text", str(response))
    
    return response.strip()


def process_sample_condition_d(
    sample: Dict[str, Any],
    condition_a_dir: Path
) -> Dict[str, Any]:
    """
    Process single sample for Condition D.
    
    Args:
        sample: Sample data
        condition_a_dir: Directory with Condition A results (for GPT analysts & retrieval)
        
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
        'gpt_query': None,     # Reused from Condition A
        'gpt_docs': None,      # Reused from Condition A
        'called_retriever': False,
        'qwen_integrator': None,
        'mortality_probability': None,
        'survival_probability': None,
        'prediction': None,
        'error': None
    }
    
    try:
        # Step 1: Load GPT analyst outputs and retrieval from Condition A
        print("  Loading GPT outputs from Condition A...")
        condition_a_file = condition_a_dir / "logs" / f"{sample_id}.json"
        if not condition_a_file.exists():
            raise FileNotFoundError(f"Condition A result not found: {condition_a_file}")
        
        with open(condition_a_file, 'r') as f:
            condition_a_result = json.load(f)
        
        analyst1_output = condition_a_result['gpt_analyst1']
        analyst2_output = condition_a_result['gpt_analyst2']
        
        result['gpt_analyst1'] = analyst1_output
        result['gpt_analyst2'] = analyst2_output
        
        # Check if retrieval was used in Condition A
        called_retriever = condition_a_result.get('called_retriever', False)
        
        if called_retriever:
            # Load GPT retrieval results
            gpt_docs = condition_a_result.get('gpt_docs', [])
            if not gpt_docs:
                raise ValueError(f"No GPT documents found in Condition A for {sample_id}")
            
            # Format documents - EXACT SAME format as GPT conditions
            from gpt_utils import format_retrieved_docs
            docs_formatted = format_retrieved_docs(gpt_docs)
            
            result['gpt_query'] = condition_a_result.get('gpt_query')
            result['gpt_docs'] = gpt_docs
            result['called_retriever'] = True
            
            print(f"  Loaded GPT analysts + {len(gpt_docs)} retrieved documents")
        else:
            # No retrieval needed - analysts were confident
            docs_formatted = ""
            result['gpt_query'] = None
            result['gpt_docs'] = []
            result['called_retriever'] = False
            
            print(f"  Loaded GPT analysts (no retrieval needed)")
        
        # Step 2: Run Qwen Integrator with GPT inputs (with or without retrieval)
        if called_retriever:
            print("  Running Qwen Integrator with GPT evidence...")
        else:
            print("  Running Qwen Integrator without retrieval...")
        qwen_response = run_qwen_integrator(
            target_context, analyst1_output, analyst2_output, docs_formatted
        )
        result['qwen_integrator'] = qwen_response
        
        # Step 3: Extract probabilities and prediction
        probs = extract_probabilities(qwen_response)
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
    parser = argparse.ArgumentParser(description="Condition D: GPT analysts + GPT retrieval + Qwen integrator")
    parser.add_argument('--manifest', type=str,
                       default='manifests/samples_swap_core.csv',
                       help='Path to sample manifest CSV')
    parser.add_argument('--full_data', type=str,
                       default='manifests/selected_samples_full.parquet',
                       help='Path to full sample data parquet')
    parser.add_argument('--condition_a_dir', type=str,
                       default='results/condition_A_gpt_4o',
                       help='Directory with Condition A results')
    parser.add_argument('--qwen_model', type=str,
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='Qwen model name for vLLM')
    parser.add_argument('--gpu_id', type=str,
                       default='0',
                       help='GPU ID for Qwen model')
    parser.add_argument('--output_dir', type=str,
                       default='results/condition_D_qwen',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CONDITION D: GPT Analysts + GPT Retrieval + Qwen Integrator")
    print("=" * 80)
    print(f"Qwen model: {args.qwen_model}")
    print(f"GPU: {args.gpu_id}")
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
    
    # Initialize Qwen integrator - EXACT pattern from mortality_single_agent_cot.py
    global _qwen_integrator
    print(f"\nInitializing Qwen integrator...")
    _qwen_integrator = QwenIntegrator(model_name=args.qwen_model, gpu_id=args.gpu_id)
    print("Qwen integrator initialized successfully")
    
    # Create output directory with logs subdirectory
    args.output_dir = args.output_dir
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
        result = process_sample_condition_d(sample, condition_a_dir)
        
        # Save individual result to logs directory
        with open(log_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        results.append(result)
        
        # Small delay between samples
        time.sleep(0.1)
    
    # Save summary as results.json in main directory
    results_file = output_dir / "results.json"
    
    # Create compact summary
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
        'condition': 'D',
        'model': args.qwen_model,
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
    print("CONDITION D COMPLETE")
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
