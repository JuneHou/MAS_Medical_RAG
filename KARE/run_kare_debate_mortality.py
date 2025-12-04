#!/usr/bin/env python3
"""
Main integration script for KARE multi-agent debate mortality prediction.
Combines KARE's data infrastructure with enhanced debate reasoning.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from kare_data_adapter import KAREDataAdapter
from mortality_debate_rag import MortalityDebateSystem as MortalityDebateRAG
from mortality_debate_cot import MortalityDebateSystem as MortalityDebateCOT

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the predictions.
    
    Args:
        results: List of prediction results
        
    Returns:
        Dictionary of calculated metrics
    """
    if not results:
        return {}
    
    # Extract predictions and ground truth, handling None values
    predictions = []
    ground_truths = []
    none_predictions = 0
    
    for result in results:
        pred = result.get('prediction')
        gt = result.get('ground_truth')
        
        if pred is None:
            none_predictions += 1
            print(f"Warning: Found None prediction for patient {result.get('patient_id', 'unknown')}")
            continue
            
        if gt is not None:
            predictions.append(pred)
            ground_truths.append(gt)
    
    if not predictions:
        print(f"Error: No valid predictions found! {none_predictions} predictions were None")
        return {}
    
    print(f"Debug: Processing {len(predictions)} valid predictions, {none_predictions} were None")
    
    # Basic metrics
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / total
    
    # Calculate confusion matrix for mortality class (1)
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 0)
    
    # Debug: Print confusion matrix
    print(f"Confusion Matrix:")
    print(f"  TP (predicted=1, actual=1): {tp}")
    print(f"  FP (predicted=1, actual=0): {fp}")
    print(f"  FN (predicted=0, actual=1): {fn}")
    print(f"  TN (predicted=0, actual=0): {tn}")
    print(f"  Total: {tp + fp + fn + tn}")
    
    # Check prediction distribution
    pred_0_count = sum(1 for p in predictions if p == 0)
    pred_1_count = sum(1 for p in predictions if p == 1)
    gt_0_count = sum(1 for g in ground_truths if g == 0)
    gt_1_count = sum(1 for g in ground_truths if g == 1)
    
    print(f"Prediction distribution: 0={pred_0_count}, 1={pred_1_count}")
    print(f"Ground truth distribution: 0={gt_0_count}, 1={gt_1_count}")
    
    # Calculate metrics with better handling of edge cases
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
        print("Warning: No positive predictions (tp + fp = 0), precision = 0.0")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
        print("Warning: No positive ground truth labels (tp + fn = 0), recall = 0.0")
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        print("Warning: Both precision and recall are 0, F1 = 0.0")
    
    # Specificity (true negative rate)
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0
        print("Warning: No negative ground truth labels (tn + fp = 0), specificity = 0.0")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'total_samples': total,
        'valid_predictions': len(predictions),
        'none_predictions': none_predictions,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'pred_distribution': {'0': pred_0_count, '1': pred_1_count},
        'gt_distribution': {'0': gt_0_count, '1': gt_1_count}
    }

def save_results(results: List[Dict[str, Any]], 
                metrics: Dict[str, float],
                output_path: str,
                include_debate_history: bool = False):
    """
    Save results and metrics to JSON file.
    
    Args:
        results: List of prediction results
        metrics: Calculated metrics
        output_path: Output file path
        include_debate_history: Whether to include full debate history
    """
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(results),
            'include_debate_history': include_debate_history
        },
        'metrics': metrics,
        'results': []
    }
    
    # Process results for saving
    for result in results:
        save_result = {
            'patient_id': result.get('patient_id'),
            'visit_id': result.get('visit_id'),
            'ground_truth': result.get('ground_truth'),
            'prediction': result.get('prediction'),
            'rounds_completed': result.get('rounds_completed'),
            'total_generation_time': result.get('total_generation_time'),
            'error': result.get('error')
        }
        
        # Include debate history if requested
        if include_debate_history:
            save_result['debate_history'] = result.get('debate_history', [])
        
        save_data['results'].append(save_result)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def run_kare_debate_evaluation(start_idx: int = 0, 
                             num_samples: int = None,
                             output_path: str = None,
                             model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                             integrator_model_name: str = None,
                             gpu_ids: str = "6,7",
                             integrator_gpu: str = None,
                             include_debate_history: bool = False,
                             batch_size: int = 10,
                             debate_mode: str = "rag",
                             corpus_name: str = "MedCorp2",
                             retriever_name: str = "MedCPT", 
                             db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
                             round1_k: int = 8,
                             round3_k: int = 16) -> List[Dict[str, Any]]:
    """
    Run KARE multi-agent debate evaluation.
    
    Args:
        start_idx: Starting index for evaluation
        num_samples: Number of samples to evaluate (None for all)
        output_path: Output file path
        model_name: Model name for VLLM (agents 1-3)
        integrator_model_name: Model name for integrator agent (agent 4). If None, uses model_name
        gpu_ids: GPU IDs to use for main model
        integrator_gpu: GPU ID for integrator model. If None, uses second GPU from gpu_ids
        include_debate_history: Whether to include full debate history in output
        batch_size: Batch size for processing
        debate_mode: Debate mode - "rag" for RAG-enhanced or "cot" for Chain-of-Thought only
        corpus_name: MedRAG corpus name (only used in RAG mode)
        retriever_name: MedRAG retriever name (only used in RAG mode)
        db_dir: MedRAG database directory (only used in RAG mode)
        round1_k: Number of documents to retrieve in Round 1 (for output path naming)
        round3_k: Number of documents to retrieve in Round 3 (for output path naming)
        
    Returns:
        List of evaluation results
    """
    print("Initializing KARE Multi-Agent Debate Evaluation...")
    
    # Initialize components
    try:
        print("Loading KARE data adapter...")
        data_adapter = KAREDataAdapter()
        
        print(f"Initializing mortality debate system in {debate_mode.upper()} mode...")
        # Determine integrator model and GPU
        if integrator_model_name is None:
            integrator_model_name = model_name
        if integrator_gpu is None:
            gpu_list = gpu_ids.split(',')
            integrator_gpu = gpu_list[1] if len(gpu_list) > 1 else gpu_list[0]
        
        # Initialize appropriate debate system based on mode
        if debate_mode.lower() == "rag":
            debate_system = MortalityDebateRAG(
                model_name=model_name, 
                gpu_ids=gpu_ids,
                integrator_model_name=integrator_model_name,
                integrator_gpu=integrator_gpu,
                rag_enabled=True,  # RAG mode always uses retrieval
                corpus_name=corpus_name,
                retriever_name=retriever_name,
                db_dir=db_dir
            )
        elif debate_mode.lower() == "cot":
            debate_system = MortalityDebateCOT(
                model_name=model_name, 
                gpu_ids=gpu_ids,
                integrator_model_name=integrator_model_name,
                integrator_gpu=integrator_gpu
            )
        else:
            raise ValueError(f"Invalid debate_mode: {debate_mode}. Must be 'rag' or 'cot'")
        
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
    print(f"Dataset statistics: {data_adapter.get_statistics()}")
    
    # Process samples
    results = []
    error_count = 0
    
    try:
        with tqdm(range(start_idx, end_idx), desc="Processing samples", ncols=100) as pbar:
            for i in pbar:
                try:
                    # Get sample data
                    sample = data_adapter.get_test_sample(i)
                    
                    # Debug: Check similar patients data length
                    if i < 3:  # Only debug first few samples
                        print(f"\nDEBUG Sample {i}:")
                        print(f"  Patient ID: {sample['patient_id']}")
                        print(f"  Positive similars length: {len(sample.get('positive_similars', ''))}")
                        print(f"  Negative similars length: {len(sample.get('negative_similars', ''))}")
                        print(f"  Positive similars preview: {sample.get('positive_similars', '')[:100]}...")
                    
                    pbar.set_postfix({
                        'Patient': sample['patient_id'],
                        'GT': sample['ground_truth'],
                        'Errors': error_count
                    })
                    
                    # Run debate with separate positive and negative similar patients
                    # Pass the full output_path so the debate system can create logs in the same directory structure
                    
                    debate_result = debate_system.debate_mortality_prediction(
                        patient_context=sample['target_context'],
                        positive_similars=sample['positive_similars'],
                        negative_similars=sample['negative_similars'],
                        medical_knowledge="",  # Can be added later if available
                        patient_id=sample['patient_id'],  # Pass patient ID for logging
                        model_name=model_name,  # Pass model name for structured logging
                        output_dir=output_path  # Pass the full output file path
                    )
                    
                    # Combine results
                    result = {
                        'patient_id': sample['patient_id'],  # KARE temporal format (e.g., \"10188_1\")
                        'base_patient_id': sample['base_patient_id'],  # Original patient ID
                        'visit_id': sample['visit_id'],
                        'visit_index': sample['visit_index'],  # Temporal visit index
                        'ground_truth': sample['ground_truth'],
                        'prediction': debate_result['final_prediction'],
                        'rounds_completed': debate_result['rounds_completed'],
                        'total_generation_time': debate_result['total_generation_time']
                    }
                    
                    # Include debate history if requested
                    if include_debate_history:
                        result['debate_history'] = debate_result['debate_history']
                    
                    results.append(result)
                    
                    # Save intermediate results every batch_size samples
                    if output_path and (len(results) % batch_size == 0):
                        metrics = calculate_metrics(results)
                        save_results(results, metrics, output_path, include_debate_history)
                        
                        # Print intermediate metrics
                        print(f"\nIntermediate Results (n={len(results)}):")
                        print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
                
                except Exception as e:
                    error_count += 1
                    print(f"\nError processing sample {i}: {e}")
                    
                    # Add error result
                    try:
                        sample = data_adapter.get_test_sample(i)
                        results.append({
                            'patient_id': sample['patient_id'],
                            'base_patient_id': sample['base_patient_id'],
                            'visit_id': sample['visit_id'],
                            'visit_index': sample['visit_index'],
                            'ground_truth': sample['ground_truth'],
                            'prediction': None,
                            'error': str(e)
                        })
                    except:
                        pass  # Skip if we can't even get sample data
                    
                    if error_count > 10:  # Stop if too many errors
                        print("Too many errors, stopping evaluation.")
                        break
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    
    print(f"\nEvaluation completed. Processed {len(results)} samples with {error_count} errors.")
    
    # Calculate final metrics
    metrics = calculate_metrics(results)
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Total Samples: {metrics.get('total_samples', 0)}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"Precision: {metrics.get('precision', 0):.3f}")
    print(f"Recall: {metrics.get('recall', 0):.3f}")
    print(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
    print(f"Specificity: {metrics.get('specificity', 0):.3f}")
    
    # Save final results
    if output_path:
        save_results(results, metrics, output_path, include_debate_history)
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="KARE Multi-Agent Debate Mortality Prediction")
    parser.add_argument('--start_idx', type=int, default=0, help='Starting sample index')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--output', type=str, default=None, help='Output file path (if not provided, will be auto-generated based on parameters)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='Model name for VLLM (agents 1-3)')
    parser.add_argument('--integrator_model', type=str, default=None, help='Model name for integrator agent (agent 4). If None, uses same as --model')
    parser.add_argument('--gpus', type=str, default='6,7', help='GPU IDs (comma-separated)')
    parser.add_argument('--integrator_gpu', type=str, default=None, help='GPU ID(s) for integrator model. Single GPU: "1", Multiple GPUs: "1,2" (uses tensor parallelism). If None, uses second GPU from --gpus')
    parser.add_argument('--include_history', action='store_true', help='Include debate history in output')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for intermediate saves')
    parser.add_argument('--mode', type=str, default='rag', choices=['rag', 'cot'], help='Debate mode: rag for RAG-enhanced, cot for Chain-of-Thought only')
    
    # Retrieval parameters for structured output path
    parser.add_argument('--round1_k', type=int, default=8, help='Number of documents to retrieve in Round 1 (for output path naming)')
    parser.add_argument('--round3_k', type=int, default=16, help='Number of documents to retrieve in Round 3 (for output path naming)')
    
    # MedRAG parameters (only used in RAG mode)
    parser.add_argument('--corpus_name', type=str, default="MedCorp2", help='MedRAG corpus name (default: MedCorp2)')
    parser.add_argument('--retriever_name', type=str, default="MedCPT", help='MedRAG retriever name (default: MedCPT)')
    parser.add_argument('--db_dir', type=str, default="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus", help='MedRAG database directory')
    
    args = parser.parse_args()
    
    # Generate automatic output path if not provided
    if args.output is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Clean model name for directory (replace / and - with _)
        clean_model_name = args.model.replace('/', '_').replace('-', '_')
        
        # Create structured directory name based on parameters
        if args.mode.lower() == 'rag':
            dir_name = f"{args.mode}_mor_{clean_model_name}_{args.round1_k}_{args.round3_k}"
        else:  # cot mode doesn't use retrieval parameters
            dir_name = f"{args.mode}_mor_{clean_model_name}"
        
        # Create results directory structure
        results_dir = script_dir / "results" / dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set output file path
        args.output = str(results_dir / "kare_debate_mortality_results.json")
        
        print(f"Auto-generated output path: {args.output}")
    
    # Run evaluation
    results = run_kare_debate_evaluation(
        start_idx=args.start_idx,
        num_samples=args.num_samples,
        output_path=args.output,
        model_name=args.model,
        integrator_model_name=args.integrator_model,
        gpu_ids=args.gpus,
        integrator_gpu=args.integrator_gpu,
        include_debate_history=args.include_history,
        batch_size=args.batch_size,
        debate_mode=args.mode,
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        db_dir=args.db_dir,
        round1_k=args.round1_k,
        round3_k=args.round3_k
    )
    
    print(f"Evaluation complete. {len(results)} results generated.")

if __name__ == "__main__":
    main()