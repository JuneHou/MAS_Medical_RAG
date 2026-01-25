#!/usr/bin/env python3
"""
Condition F: Qwen analysts + GPT retrieval + Qwen integrator

This condition tests whether Qwen integrator can work well with Qwen analysts and GPT retrieval.

Flow:
1. Reuse QWEN Analyst 1 & 2 outputs from Qwen baseline logs
2. Reuse GPT retrieval (query + docs) from Condition C when available; else run without retrieval (same fallback as C)
3. Run QWEN Integrator (final) with Qwen analyst outputs and GPT-retrieved evidence (or no evidence)

This tests: Can Qwen analysts + Qwen integrator work well with GPT retrieval?
Compared to Condition C (Qwen+GPT+GPT), this swaps only the integrator.
Compared to Condition D (GPT+GPT+Qwen), this swaps only the analysts.
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
    AGENT_PROMPTS, extract_probabilities, load_qwen_debate_history,
    format_retrieved_docs
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


def run_qwen_integrator_final(
    target_context: str,
    analyst1_output: str,
    analyst2_output: str,
    retrieved_docs: Optional[str],
    qwen_model: str,
    gpu_id: str
) -> str:
    """
    Run Qwen integrator (final) with retrieved evidence.
    Supports no retrieval (retrieved_docs None/empty) - same fallback as Condition C.
    
    Args:
        target_context: Target patient context
        analyst1_output: Analyst 1 output (from Qwen baseline)
        analyst2_output: Analyst 2 output (from Qwen baseline)
        retrieved_docs: Retrieved documents text (from GPT retrieval), or None if no retrieval
        qwen_model: Qwen model name
        gpu_id: GPU ID for Qwen
        
    Returns:
        Integrator final response with prediction
    """
    global _qwen_integrator
    
    if _qwen_integrator is None:
        _qwen_integrator = QwenIntegrator(qwen_model, gpu_id)
    
    # Use EXACT prompt from gpt_utils - same as GPT conditions without search
    system_prompt = AGENT_PROMPTS["balanced_clinical_integrator"].replace(
        "Available tools:\n- <search>query</search>: Retrieve medical evidence. Retrieved information will appear in <information>...</information> tags.\n\nWorkflow:\n1) Compare the Target patient to two similar cases using the two analysis, and write 3-4 key factors contribute to the target patient's next visit.\n2) When you need additional knowledge, call <search>your custom query</search> based on the patient's specific conditions (e.g., <search>sepsis mortality prognosis elderly patients</search>)\n3) After seeing the <information>retrieved evidence</information>, analyze BOTH risky factors AND survival factors.\n4) After reviewing all evidence, provide your final assessment with:",
        "Instructions:\n1) Compare the Target patient to two similar cases using the analysis provided.\n2) Review the retrieved evidence provided in <information> tags below.\n3) Analyze BOTH risky factors AND survival factors.\n4) Provide your final assessment with:"
    )
    
    # Build prompt - with or without retrieval (same fallback as C/E)
    if retrieved_docs:
        integrator_prompt = f"""{system_prompt}

## Target Patient:
{target_context}

## Analysis 1 (Risk Factors):
{analyst1_output}

## Analysis 2 (Protective Factors):
{analyst2_output}

<information>
{retrieved_docs}
</information>

Now provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
    else:
        integrator_prompt = f"""{system_prompt}

## Target Patient:
{target_context}

## Analysis 1 (Risk Factors):
{analyst1_output}

## Analysis 2 (Protective Factors):
{analyst2_output}

Provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
    
    # Generate response
    response = _qwen_integrator(integrator_prompt, max_tokens=4096, temperature=0.7)
    
    return response


def load_qwen_analyst_outputs(
    patient_id: str,
    qwen_log_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Load Qwen analyst outputs from existing Qwen baseline logs.
    
    Args:
        patient_id: Patient ID
        qwen_log_dir: Directory containing Qwen debate logs
        
    Returns:
        Dictionary with analyst1 and analyst2 outputs, or None
    """
    history = load_qwen_debate_history(patient_id, qwen_log_dir)
    
    if not history:
        return None
    
    return {
        'analyst1': history.get('analyst1_output'),
        'analyst2': history.get('analyst2_output')
    }


def process_sample_condition_f(
    sample: Dict[str, Any],
    condition_c_dir: Path,
    qwen_log_dir: str,
    qwen_model: str,
    gpu_id: str
) -> Dict[str, Any]:
    """
    Process single sample for Condition F.
    
    Args:
        sample: Sample data
        condition_c_dir: Directory with Condition C results (for GPT retrieval)
        qwen_log_dir: Directory with Qwen baseline logs (for analyst outputs)
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
        'qwen_analyst1': None,
        'qwen_analyst2': None,
        'gpt_query': None,  # Reused from Condition C
        'gpt_docs': None,   # Reused from Condition C
        'called_retriever': False,
        'qwen_integrator_final': None,
        'mortality_probability': None,
        'survival_probability': None,
        'prediction': None,
        'error': None
    }
    
    try:
        # Step 1: Load Qwen analyst outputs from Qwen baseline
        print("  Loading Qwen analyst outputs...")
        qwen_analysts = load_qwen_analyst_outputs(sample_id, qwen_log_dir)
        
        if not qwen_analysts or not qwen_analysts['analyst1'] or not qwen_analysts['analyst2']:
            raise ValueError(f"Qwen analyst outputs not found for {sample_id}")
        
        result['qwen_analyst1'] = qwen_analysts['analyst1']
        result['qwen_analyst2'] = qwen_analysts['analyst2']
        print("  Loaded Qwen analyst outputs")
        
        # Step 2: Load GPT retrieval from Condition C (same fallback as C: run without retrieval if none)
        print("  Loading GPT retrieval bundle from Condition C...")
        condition_c_log = condition_c_dir / "logs" / f"{sample_id}.json"
        if not condition_c_log.exists():
            raise FileNotFoundError(f"Condition C result not found: {condition_c_log}")
        
        with open(condition_c_log, 'r') as f:
            condition_c_data = json.load(f)
        
        called_retriever = condition_c_data.get('called_retriever', False)
        gpt_docs = condition_c_data.get('gpt_docs') or []
        
        if called_retriever and gpt_docs:
            result['gpt_query'] = condition_c_data.get('gpt_query')
            result['gpt_docs'] = gpt_docs
            result['called_retriever'] = True
            print(f"  GPT retrieval found: {len(gpt_docs)} documents")
            print("  Running Qwen Integrator with GPT evidence...")
            docs_formatted = format_retrieved_docs(gpt_docs)
            integrator_final = run_qwen_integrator_final(
                target_context,
                result['qwen_analyst1'],
                result['qwen_analyst2'],
                docs_formatted,
                qwen_model,
                gpu_id
            )
        else:
            result['gpt_query'] = None
            result['gpt_docs'] = None
            result['called_retriever'] = False
            if called_retriever and not gpt_docs:
                print("  No GPT documents available, running without retrieval")
            else:
                print("  No GPT retrieval in Condition C, running without retrieval")
            print("  Running Qwen Integrator without retrieval...")
            integrator_final = run_qwen_integrator_final(
                target_context,
                result['qwen_analyst1'],
                result['qwen_analyst2'],
                None,
                qwen_model,
                gpu_id
            )
        
        result['qwen_integrator_final'] = integrator_final
        
        # Step 3: Extract probabilities and prediction (same as Condition E / ABCD)
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
    parser = argparse.ArgumentParser(description='Run Condition F: Qwen+GPT+Qwen')
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
    condition_c_dir = gpt_dir / "results_bias" / "condition_C_gpt_4o"
    output_dir = gpt_dir / "results_bias" / "condition_F_qwen_gpt_qwen"
    log_dir = output_dir / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CONDITION F: Qwen Analysts + GPT Retrieval + Qwen Integrator")
    print("=" * 80)
    print(f"Qwen model: {args.qwen_model}")
    print(f"GPU: {args.gpu_id}")
    print(f"Output: {output_dir}")
    print(f"Condition C source: {condition_c_dir}")
    print(f"Qwen log dir: {args.qwen_log_dir}")
    
    # Verify directories exist
    if not condition_c_dir.exists():
        print(f"\nERROR: Condition C directory not found: {condition_c_dir}")
        print("Please run run_condition_C.py first!")
        sys.exit(1)
    
    if not Path(args.qwen_log_dir).exists():
        print(f"\nERROR: Qwen log directory not found: {args.qwen_log_dir}")
        print("Qwen analyst outputs will not be available!")
        sys.exit(1)
    
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
        result = process_sample_condition_f(
            sample, condition_c_dir, args.qwen_log_dir,
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
        'condition': 'F',
        'model': 'Qwen+GPT+Qwen',
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
    print("CONDITION F COMPLETE")
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
