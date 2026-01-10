#!/usr/bin/env python3
"""
Runner script for KARE single-agent experiments (CoT and RAG).
Tests Qwen2.5-7B-Instruct performance following KARE zero_shot_base setting.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from kare_data_adapter import KAREDataAdapter
from mortality_single_agent_cot import MortilitySingleAgentCoT
from mortality_single_agent_rag import MortilitySingleAgentRAG

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate evaluation metrics for the predictions."""
    if not results:
        return {}
    
    # Extract predictions and ground truth
    predictions = []
    ground_truths = []
    none_predictions = 0
    fallback_ids = []
    
    for result in results:
        pred = result.get('prediction')
        gt = result.get('ground_truth')
        patient_id = result.get('patient_id', 'unknown')
        
        # Track fallback predictions
        if result.get('is_fallback', False):
            fallback_ids.append(patient_id)
        
        if pred is None:
            none_predictions += 1
            print(f"Warning: Found None prediction for patient {patient_id}")
            continue
            
        if gt is not None:
            predictions.append(pred)
            ground_truths.append(gt)
    
    if not predictions:
        print(f"Error: No valid predictions found! {none_predictions} predictions were None")
        return {}
    
    print(f"Processing {len(predictions)} valid predictions, {none_predictions} were None")
    print(f"Fallback predictions: {len(fallback_ids)}")
    if fallback_ids:
        print(f"Fallback patient IDs: {', '.join(fallback_ids)}")
    
    # Basic metrics
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / total
    
    # Calculate confusion matrix
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 0)
    
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate macro-F1 (average of F1 for both classes)
    # F1 for class 1 (mortality)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_mortality = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    
    # F1 for class 0 (survival)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_survival = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    
    # Macro-F1 is the average
    macro_f1 = (f1_mortality + f1_survival) / 2
    
    # Calculate prediction and ground truth distributions
    pred_dist = {0: sum(1 for p in predictions if p == 0), 1: sum(1 for p in predictions if p == 1)}
    gt_dist = {0: sum(1 for g in ground_truths if g == 0), 1: sum(1 for g in ground_truths if g == 1)}
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'macro_f1': macro_f1,
        'f1_mortality': f1_mortality,
        'f1_survival': f1_survival,
        'specificity': specificity,
        'total_samples': total,
        'valid_predictions': len(predictions),
        'none_predictions': none_predictions,
        'fallback_predictions': len(fallback_ids),
        'fallback_patient_ids': fallback_ids,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'pred_distribution': pred_dist,
        'gt_distribution': gt_dist
    }

def load_existing_results(output_path: str) -> tuple[List[Dict[str, Any]], set]:
    """Load existing results from JSON file."""
    if not os.path.exists(output_path):
        return [], set()
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        existing_results = data.get('results', [])
        processed_ids = {result['patient_id'] for result in existing_results if result.get('patient_id')}
        
        print(f"Loaded {len(existing_results)} existing results ({len(processed_ids)} unique patients)")
        return existing_results, processed_ids
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()

def save_results(results: List[Dict[str, Any]], 
                metrics: Dict[str, float],
                output_path: str,
                mode: str):
    """Save results and metrics to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': mode,
            'total_samples': len(results)
        },
        'metrics': metrics,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def run_single_agent_evaluation(start_idx: int = 0,
                                num_samples: int = None,
                                output_path: str = None,
                                model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                gpu_ids: str = "6,7",
                                batch_size: int = 10,
                                mode: str = "cot",
                                in_context: str = "zero-shot",
                                corpus_name: str = "MedCorp2",
                                retriever_name: str = "MedCPT",
                                db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus") -> List[Dict[str, Any]]:
    """
    Run KARE single-agent evaluation.
    
    Args:
        start_idx: Starting index
        num_samples: Number of samples to evaluate
        output_path: Output file path
        model_name: Model name for VLLM
        gpu_ids: GPU IDs
        batch_size: Batch size for intermediate saves
        mode: "cot" or "rag"
        in_context: "zero-shot" or "few-shot" (whether to use similar patients)
        corpus_name: MedRAG corpus (only for RAG mode)
        retriever_name: MedRAG retriever (only for RAG mode)
        db_dir: MedRAG database directory (only for RAG mode)
    """
    print(f"Initializing KARE Single-Agent {mode.upper()} Evaluation...")
    
    # Load existing results
    existing_results, processed_patients = load_existing_results(output_path) if output_path else ([], set())
    
    # Initialize components
    try:
        print("Loading KARE data adapter...")
        data_adapter = KAREDataAdapter()
        
        print(f"Initializing single-agent {mode.upper()} system...")
        if mode.lower() == "rag":
            agent_system = MortilitySingleAgentRAG(
                model_name=model_name,
                gpu_ids=gpu_ids,
                in_context=in_context,
                corpus_name=corpus_name,
                retriever_name=retriever_name,
                db_dir=db_dir
            )
        elif mode.lower() == "cot":
            # CoT only needs single GPU
            gpu_list = gpu_ids.split(',')
            agent_system = MortilitySingleAgentCoT(
                model_name=model_name,
                gpu_id=gpu_list[0],  # Use first GPU only
                in_context=in_context
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'cot' or 'rag'")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        return []
    
    # Determine sample range
    total_samples = len(data_adapter.test_data)
    if num_samples is None:
        num_samples = total_samples - start_idx
    else:
        num_samples = min(num_samples, total_samples - start_idx)
    
    end_idx = start_idx + num_samples
    print(f"Processing samples {start_idx} to {end_idx-1} ({num_samples} total)")
    
    # Process samples
    results = existing_results.copy()
    error_count = 0
    skipped_count = 0
    
    try:
        with tqdm(range(start_idx, end_idx), desc="Processing samples") as pbar:
            for i in pbar:
                try:
                    sample = data_adapter.get_test_sample(i)
                    
                    # Skip if already processed
                    if sample['patient_id'] in processed_patients:
                        skipped_count += 1
                        pbar.set_postfix({'Skipped': skipped_count, 'Errors': error_count})
                        continue
                    
                    pbar.set_postfix({
                        'Patient': sample['patient_id'],
                        'GT': sample['ground_truth'],
                        'Errors': error_count
                    })
                    
                    # Run single-agent prediction
                    prediction_result = agent_system.predict_mortality(
                        patient_context=sample['target_context'],  # Changed from 'patient_context' to 'target_context'
                        positive_similars=sample['positive_similars'],
                        negative_similars=sample['negative_similars'],
                        patient_id=sample['patient_id'],
                        output_dir=Path(output_path).parent if output_path else None,
                        ground_truth=sample['ground_truth']
                    )
                    
                    # Combine results
                    result = {
                        'patient_id': sample['patient_id'],
                        'visit_id': sample['visit_id'],
                        'ground_truth': sample['ground_truth'],
                        'prediction': prediction_result['final_prediction'],
                        'is_fallback': prediction_result.get('is_fallback', False),
                        'total_generation_time': prediction_result['total_generation_time']
                    }
                    
                    results.append(result)
                    
                    # Save intermediate results
                    if output_path and (len(results) % batch_size == 0):
                        metrics = calculate_metrics(results)
                        save_results(results, metrics, output_path, mode)
                        print(f"\nIntermediate save: {len(results)} results")
                
                except Exception as e:
                    error_count += 1
                    print(f"\nError processing sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    
    finally:
        # Final save
        if output_path and results:
            metrics = calculate_metrics(results)
            save_results(results, metrics, output_path, mode)
            print(f"\nFinal Results:")
            print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"Precision: {metrics.get('precision', 0):.3f}")
            print(f"Recall: {metrics.get('recall', 0):.3f}")
            print(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="KARE Single-Agent Mortality Prediction")
    parser.add_argument('--start_idx', type=int, default=0, help='Starting sample index')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model name')
    parser.add_argument('--gpus', type=str, default='6,7', help='GPU IDs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--mode', type=str, default='cot', choices=['cot', 'rag'], 
                       help='Mode: cot for Chain-of-Thought, rag for RAG-enhanced')
    parser.add_argument('--in_context', type=str, default='zero-shot', choices=['zero-shot', 'few-shot'],
                       help='In-context mode: zero-shot (no similar patients) or few-shot (with similar patients)')
    
    # RAG parameters
    parser.add_argument('--corpus_name', type=str, default="MedCorp2", help='MedRAG corpus')
    parser.add_argument('--retriever_name', type=str, default="MedCPT", help='MedRAG retriever')
    parser.add_argument('--db_dir', type=str, 
                       default="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
                       help='MedRAG database directory')
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        script_dir = Path(__file__).parent
        clean_model_name = args.model.replace('/', '_').replace('-', '_')
        in_context_suffix = args.in_context.replace('-', '_')
        
        if args.mode == 'rag':
            dir_name = f"single_{args.mode}_mor_{clean_model_name}_{args.retriever_name}_{in_context_suffix}"
        else:
            dir_name = f"single_{args.mode}_mor_{clean_model_name}_{in_context_suffix}"
        
        results_dir = script_dir / "results" / dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(results_dir / "results.json")
        print(f"Auto-generated output path: {args.output}")
    
    # Run evaluation
    results = run_single_agent_evaluation(
        start_idx=args.start_idx,
        num_samples=args.num_samples,
        output_path=args.output,
        model_name=args.model,
        gpu_ids=args.gpus,
        batch_size=args.batch_size,
        mode=args.mode,
        in_context=args.in_context,
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        db_dir=args.db_dir
    )
    
    print(f"\nEvaluation complete. {len(results)} results generated.")

if __name__ == "__main__":
    main()
