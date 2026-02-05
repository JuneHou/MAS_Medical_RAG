#!/usr/bin/env python3
"""
Main integration script for KARE multi-agent debate mortality prediction using effGen.
Combines KARE's data infrastructure with effGen-based debate reasoning.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import KARE data adapter (reuse from parent directory)
try:
    from kare_data_adapter import KAREDataAdapter
except ImportError:
    print("Error: kare_data_adapter.py not found in parent directory")
    sys.exit(1)

# Import effGen debate systems
try:
    from mortality_debate_effgen_cot import MortalityDebateSystemEffGen as MortalityDebateCOT
    from mortality_debate_effgen_rag import MortalityDebateSystemEffGenRAG as MortalityDebateRAG
except ImportError as e:
    print(f"Error importing effGen debate systems: {e}")
    print("Make sure mortality_debate_effgen_cot.py and mortality_debate_effgen_rag.py are in the same directory")
    sys.exit(1)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the predictions.
    (Reused from original run_kare_debate_mortality.py)
    """
    if not results:
        return {}
    
    # Extract predictions and ground truth
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
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate macro-F1
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_mortality = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_survival = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    
    macro_f1 = (f1_mortality + f1_survival) / 2
    
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
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def load_existing_results(output_path: str) -> tuple:
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
                include_debate_history: bool = False):
    """Save results and metrics to JSON file."""
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(results),
            'include_debate_history': include_debate_history,
            'framework': 'effgen'
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
                             model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                             gpu_ids: str = "0",
                             include_debate_history: bool = False,
                             batch_size: int = 10,
                             debate_mode: str = "cot",
                             corpus_name: str = "MedCorp2",
                             retriever_name: str = "MedCPT", 
                             db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus") -> List[Dict[str, Any]]:
    """
    Run KARE multi-agent debate evaluation using effGen.
    
    Args:
        start_idx: Starting index for evaluation
        num_samples: Number of samples to evaluate (None for all)
        output_path: Output file path
        model_name: Model name
        gpu_ids: GPU IDs to use
        include_debate_history: Whether to include full debate history in output
        batch_size: Batch size for processing
        debate_mode: Debate mode - "rag" for RAG-enhanced or "cot" for Chain-of-Thought only
        corpus_name: MedRAG corpus name (only used in RAG mode)
        retriever_name: MedRAG retriever name (only used in RAG mode)
        db_dir: MedRAG database directory (only used in RAG mode)
        
    Returns:
        List of evaluation results
    """
    print("Initializing KARE Multi-Agent Debate Evaluation (effGen)...")
    
    # Load existing results
    existing_results, processed_patients = load_existing_results(output_path) if output_path else ([], set())
    
    if processed_patients:
        print(f"Found {len(processed_patients)} already processed patients (will skip these)")
    
    # Initialize components
    try:
        print("Loading KARE data adapter...")
        # Use parent directory path for data
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_adapter = KAREDataAdapter(base_path=os.path.join(parent_dir, "data"))
        
        print(f"Initializing mortality debate system in {debate_mode.upper()} mode (effGen)...")
        
        # Initialize appropriate debate system based on mode
        if debate_mode.lower() == "rag":
            debate_system = MortalityDebateRAG(
                model_name=model_name, 
                gpu_ids=gpu_ids,
                corpus_name=corpus_name,
                retriever_name=retriever_name,
                db_dir=db_dir
            )
        elif debate_mode.lower() == "cot":
            debate_system = MortalityDebateCOT(
                model_name=model_name, 
                gpu_ids=gpu_ids
            )
        else:
            raise ValueError(f"Invalid debate_mode: {debate_mode}. Must be 'rag' or 'cot'")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
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
    results = existing_results.copy()
    error_count = 0
    skipped_count = 0
    
    try:
        with tqdm(range(start_idx, end_idx), desc="Processing samples", ncols=100) as pbar:
            for i in pbar:
                try:
                    # Get sample data
                    sample = data_adapter.get_test_sample(i)
                    
                    # Skip if already processed
                    if sample['patient_id'] in processed_patients:
                        skipped_count += 1
                        pbar.set_postfix({
                            'Patient': sample['patient_id'],
                            'Status': 'SKIPPED',
                            'Skipped': skipped_count,
                            'Errors': error_count
                        })
                        continue
                    
                    pbar.set_postfix({
                        'Patient': sample['patient_id'],
                        'GT': sample['ground_truth'],
                        'Skipped': skipped_count,
                        'Errors': error_count
                    })
                    
                    # Run debate
                    debate_result = debate_system.debate_mortality_prediction(
                        patient_context=sample['target_context'],
                        positive_similars=sample['positive_similars'],
                        negative_similars=sample['negative_similars'],
                        medical_knowledge="",
                        patient_id=sample['patient_id'],
                        model_name=model_name,
                        output_dir=output_path,
                        ground_truth=sample['ground_truth']
                    )
                    
                    # Combine results
                    result = {
                        'patient_id': sample['patient_id'],
                        'base_patient_id': sample['base_patient_id'],
                        'visit_id': sample['visit_id'],
                        'visit_index': sample['visit_index'],
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
                    import traceback
                    traceback.print_exc()
                    
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
                        pass
                    
                    # Save after errors
                    if output_path and error_count % 5 == 0:
                        print(f"Saving results after {error_count} errors...")
                        metrics = calculate_metrics(results)
                        save_results(results, metrics, output_path, include_debate_history)
                    
                    if error_count > 50:
                        print(f"WARNING: Too many errors ({error_count}), stopping evaluation.")
                        break
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    
    finally:
        # ALWAYS save results before exiting
        print(f"\nSaving final results before exit...")
        if output_path and results:
            try:
                metrics = calculate_metrics(results)
                save_results(results, metrics, output_path, include_debate_history)
                print(f"Final save completed: {len(results)} results saved.")
            except Exception as e:
                print(f"ERROR during final save: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nEvaluation completed. Total results: {len(results)}, Skipped: {skipped_count}, Errors: {error_count}")
    
    # Calculate and print final metrics
    if results:
        metrics = calculate_metrics(results)
        
        print(f"\nFinal Results:")
        print(f"Total Samples: {metrics.get('total_samples', 0)}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"Precision: {metrics.get('precision', 0):.3f}")
        print(f"Recall: {metrics.get('recall', 0):.3f}")
        print(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
        print(f"Macro-F1: {metrics.get('macro_f1', 0):.3f}")
        print(f"Specificity: {metrics.get('specificity', 0):.3f}")
    else:
        print("\nNo results to display.")
    
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="KARE Multi-Agent Debate Mortality Prediction (effGen)")
    parser.add_argument('--start_idx', type=int, default=0, help='Starting sample index')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model name')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs (comma-separated)')
    parser.add_argument('--include_history', action='store_true', help='Include debate history in output')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for intermediate saves')
    parser.add_argument('--mode', type=str, default='cot', choices=['rag', 'cot'], 
                       help='Debate mode: rag for RAG-enhanced, cot for Chain-of-Thought only')
    
    # MedRAG parameters (only used in RAG mode)
    parser.add_argument('--corpus_name', type=str, default="MedCorp2", help='MedRAG corpus name')
    parser.add_argument('--retriever_name', type=str, default="MedCPT", help='MedRAG retriever name')
    parser.add_argument('--db_dir', type=str, 
                       default="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
                       help='MedRAG database directory')
    
    args = parser.parse_args()
    
    # Generate automatic output path if not provided
    if args.output is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Clean model name for directory
        clean_model_name = args.model.replace('/', '_').replace('-', '_')
        
        # Create structured directory name based on parameters
        if args.mode.lower() == 'rag':
            dir_name = f"effgen_{args.mode}_{clean_model_name}_{args.retriever_name}"
        else:
            dir_name = f"effgen_{args.mode}_{clean_model_name}"
        
        # Create results directory structure
        results_dir = script_dir / "results" / dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set output file path
        args.output = str(results_dir / "results.json")
        
        print(f"Auto-generated output path: {args.output}")
    
    # Run evaluation
    results = run_kare_debate_evaluation(
        start_idx=args.start_idx,
        num_samples=args.num_samples,
        output_path=args.output,
        model_name=args.model,
        gpu_ids=args.gpus,
        include_debate_history=args.include_history,
        batch_size=args.batch_size,
        debate_mode=args.mode,
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        db_dir=args.db_dir
    )
    
    print(f"Evaluation complete. {len(results)} results generated.")


if __name__ == "__main__":
    main()
