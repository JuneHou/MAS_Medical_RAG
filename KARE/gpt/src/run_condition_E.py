#!/usr/bin/env python3
"""
Condition E: GPT analysts + Qwen retrieval + Qwen integrator

This condition tests the combination of strong analyst reasoning with Qwen's retrieval and integration.

Flow:
1. Reuse GPT Analyst 1 & 2 outputs from Condition A
2. Reuse QWEN retrieval bundle (query + docs) from existing Qwen RAG logs
3. Run QWEN Integrator (final) with Qwen-retrieved evidence

This tests: Can Qwen integrator work well with GPT analysts and existing Qwen retrieval?
Compared to Condition B (GPT+Qwen+GPT), this swaps only the integrator.
Compared to Condition D (GPT+GPT+Qwen), this swaps only the retrieval.
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
from gpt_utils_bias import (
    AGENT_PROMPTS, extract_probabilities, load_qwen_debate_history
)


class QwenIntegrator:
    """Qwen integrator wrapper - EXACT pattern from mortality_single_agent_cot.py"""
    
    def __init__(self, model_name: str, gpu_id: str):
        """Initialize Qwen integrator"""
        self.model_name = model_name
        self.gpu_id = gpu_id
        
        print(f"Qwen integrator ({model_name}) will use GPU: {self.gpu_id}")
        
        # Only set CUDA_VISIBLE_DEVICES if not already set externally
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
            print(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id}")
        else:
            print(f"CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Initialize VLLM directly
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enforce_eager=True
        )
        print(f"VLLM initialized for {model_name}")
    
    def __call__(self, prompt: str, max_tokens: int = 32768, temperature: float = 0.7) -> str:
        """Generate response"""
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


def load_qwen_retrieval_bundle(
    patient_id: str,
    qwen_log_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Load Qwen retrieval bundle from existing logs.
    EXACT SAME as Condition B.
    
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


def run_qwen_integrator_final(
    target_context: str,
    analyst1_output: str,
    analyst2_output: str,
    retrieved_docs: str,
    qwen_model: str,
    gpu_id: str
) -> str:
    """
    Run Qwen integrator with retrieved evidence (no initial turn needed).
    MATCHES Condition B pattern with Qwen integrator instead of GPT.
    
    Args:
        target_context: Target patient context
        analyst1_output: Analyst 1 output (from GPT)
        analyst2_output: Analyst 2 output (from GPT)
        initial_response: Integrator's initial response
        retrieved_docs: Retrieved documents formatted as string
        qwen_model: Qwen model name
        gpu_id: GPU ID for Qwen
        
    Returns:
        Integrator final response with prediction
    """
    global _qwen_integrator
    
    if _qwen_integrator is None:
        _qwen_integrator = QwenIntegrator(qwen_model, gpu_id)
    
    # Use EXACT prompt from gpt_utils - SAME as Condition A
    system_prompt = AGENT_PROMPTS["balanced_clinical_integrator"]
    
    # Format history - EXACT SAME as Condition B
    history_text = f"""
## Previous Analysis:

### Similar Case with Mortality=1 (positive class) Analysis:
{analyst1_output}

### Similar Case with Survival=0 (negative class) Analysis:
{analyst2_output}

**Note:** The above analyses were conducted without knowledge of outcomes. Use these pattern comparisons to inform your assessment.
"""
    
    # Build prompt with retrieved information - EXACT SAME as Condition B pattern
    if retrieved_docs:
        integrator_prompt = f"""{system_prompt}

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
        # No retrieval available - direct assessment
        integrator_prompt = f"""{system_prompt}

## Target Patient:
{target_context}

{history_text}

Provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
    
    # Generate response
    response = _qwen_integrator(integrator_prompt, max_tokens=4096, temperature=0.7)
    
    return response


def process_sample_condition_e(
    sample: Dict[str, Any],
    condition_a_dir: Path,
    qwen_log_dir: str,
    qwen_model: str,
    gpu_id: str
) -> Dict[str, Any]:
    """
    Process single sample for Condition E.
    MATCHES Condition B pattern with Qwen integrator.
    
    Args:
        sample: Sample data
        condition_a_dir: Directory with Condition A results
        qwen_log_dir: Directory with Qwen debate logs
        qwen_model: Qwen model name
        gpu_id: GPU ID for Qwen
        
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
        'qwen_query': None,    # Reused from Qwen baseline
        'qwen_docs': None,     # Reused from Qwen baseline
        'called_retriever': False,
        'qwen_integrator_final': None,
        'mortality_probability': None,
        'survival_probability': None,
        'prediction': None,
        'error': None
    }
    
    try:
        # Step 1: Load GPT analyst outputs from Condition A
        print(f"  Loading GPT analysts from Condition A...")
        condition_a_log = condition_a_dir / "logs" / f"{sample_id}.json"
        
        if not condition_a_log.exists():
            raise FileNotFoundError(f"Condition A log not found: {condition_a_log}")
        
        with open(condition_a_log, 'r') as f:
            condition_a_data = json.load(f)
        
        analyst1_output = condition_a_data.get('gpt_analyst1')
        analyst2_output = condition_a_data.get('gpt_analyst2')
        
        if not analyst1_output or not analyst2_output:
            raise ValueError(f"Missing analyst outputs in Condition A for {sample_id}")
        
        result['gpt_analyst1'] = analyst1_output
        result['gpt_analyst2'] = analyst2_output
        print("  Loaded GPT analysts from Condition A")
        
        # Step 2: Load Qwen retrieval bundle - SAME as Condition B
        print("  Loading Qwen retrieval bundle...")
        qwen_bundle = load_qwen_retrieval_bundle(sample_id, qwen_log_dir)
        
        if qwen_bundle and qwen_bundle['called_retriever']:
            print(f"  Qwen retrieval found: {qwen_bundle['query'][:100] if qwen_bundle['query'] else 'No query'}...")
            result['qwen_query'] = qwen_bundle['query']
            result['qwen_docs'] = qwen_bundle['docs_text']
            result['called_retriever'] = True
            
            # Step 3: Run Qwen Integrator with Qwen-retrieved evidence
            print("  Running Qwen Integrator with Qwen evidence...")
            integrator_final = run_qwen_integrator_final(
                target_context, analyst1_output, analyst2_output,
                qwen_bundle['docs_text'],
                qwen_model, gpu_id
            )
            result['qwen_integrator_final'] = integrator_final
            
        else:
            # No Qwen retrieval available - run without retrieval
            print("  No Qwen retrieval available, running without retrieval")
            print("  Running Qwen Integrator without retrieval...")
            integrator_final = run_qwen_integrator_final(
                target_context, analyst1_output, analyst2_output,
                None,  # retrieved_docs
                qwen_model, gpu_id
            )
            result['qwen_integrator_final'] = integrator_final
        
        # Step 4: Extract probabilities and prediction - SAME as Condition B
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
    parser = argparse.ArgumentParser(description='Run Condition E: GPT+Qwen+Qwen')
    parser.add_argument('--qwen_model', type=str, 
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='Qwen model name')
    parser.add_argument('--gpu_id', type=str, default='0',
                       help='GPU ID for Qwen')
    parser.add_argument('--qwen_log_dir', type=str,
                       default='/data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs',
                       help='Directory with Qwen debate logs')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup paths
    gpt_dir = Path(__file__).parent.parent
    condition_a_dir = gpt_dir / "results_bias" / "condition_A_gpt_4o"
    output_dir = gpt_dir / "results_bias" / "condition_E_gpt_qwen_qwen"
    log_dir = output_dir / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CONDITION E: GPT Analysts + Qwen Retrieval + Qwen Integrator")
    print("=" * 80)
    print(f"Qwen model: {args.qwen_model}")
    print(f"GPU: {args.gpu_id}")
    print(f"Output: {output_dir}")
    print(f"Condition A source: {condition_a_dir}")
    print(f"Qwen log dir: {args.qwen_log_dir}")
    
    # Verify Condition A exists
    if not condition_a_dir.exists():
        print(f"\nERROR: Condition A directory not found: {condition_a_dir}")
        print("Please run run_condition_A.py first!")
        sys.exit(1)
    
    # Verify Qwen log directory exists
    if not Path(args.qwen_log_dir).exists():
        print(f"\nWARNING: Qwen log directory not found: {args.qwen_log_dir}")
        print("Retrieval bundles may not be available for some samples")
    
    # Load samples - USE SAME MANIFEST AS CONDITION A
    manifest_path = gpt_dir / "manifests" / "selected_samples_full.parquet"
    print(f"\nLoading samples from {manifest_path}...")
    import pandas as pd
    samples_df = pd.read_parquet(manifest_path)
    samples = samples_df.to_dict('records')
    
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Processing only {args.max_samples} samples (test mode)")
    
    print(f"Loaded {len(samples)} samples")
    
    # Process samples
    results = []
    for sample in tqdm(samples, desc="Processing samples"):
        result = process_sample_condition_e(
            sample, condition_a_dir, args.qwen_log_dir,
            args.qwen_model, args.gpu_id
        )
        results.append(result)
        
        # Save individual log
        log_path = log_dir / f"{result['sample_id']}.json"
        with open(log_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Calculate metrics
    valid_results = [r for r in results if r['prediction'] is not None]
    successful = len(valid_results)
    errors = sum(1 for r in results if r.get('error') is not None)
    called_retriever = sum(1 for r in results if r.get('called_retriever', False))
    
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
    
    accuracy = None
    precision = recall = f1 = 0.0
    tp = fp = fn = tn = 0
    if valid_results:
        correct = sum(1 for r in valid_results if r['prediction'] == r['label'])
        accuracy = correct / len(valid_results)
        tp = sum(1 for r in valid_results if r['prediction'] == 1 and r['label'] == 1)
        fp = sum(1 for r in valid_results if r['prediction'] == 1 and r['label'] == 0)
        fn = sum(1 for r in valid_results if r['prediction'] == 0 and r['label'] == 1)
        tn = sum(1 for r in valid_results if r['prediction'] == 0 and r['label'] == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    summary = {
        'condition': 'E',
        'model': 'GPT+Qwen+Qwen',
        'total_samples': len(samples),
        'successful': successful,
        'errors': errors,
        'called_retriever': called_retriever,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'results': compact_results
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("CONDITION E COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful']}")
    print(f"Errors: {summary['errors']}")
    print(f"Called retriever: {summary['called_retriever']}")
    print(f"Accuracy: {summary['accuracy']:.3f}" if summary['accuracy'] is not None else "Accuracy: N/A")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
