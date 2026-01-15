#!/usr/bin/env python3
"""
Extract metadata from KARE dataset and multi-agent debate results.
Creates a unified candidate table for GPT experiments.

This script:
1. Loads the KARE test dataset (996 samples) with stable identifiers
2. Loads three multi-agent log sources (CoT, RAG-Qwen, RAG-R1)
3. Joins them on sample_id (patient_id)
4. Computes correctness flags, retriever-call flags, and complexity features
5. Exports to cache/candidate_table.parquet
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path to import KARE data adapter
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kare_data_adapter import KAREDataAdapter


def load_multiagent_results(result_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load multi-agent results from JSON file.
    
    Args:
        result_path: Path to kare_debate_mortality_results.json
        
    Returns:
        Dictionary mapping patient_id -> result data
    """
    if not os.path.exists(result_path):
        print(f"Warning: Results file not found: {result_path}")
        return {}
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Map by patient_id for easy lookup
    results_map = {}
    for result in results:
        patient_id = result.get('patient_id')
        if patient_id:
            results_map[patient_id] = result
    
    print(f"Loaded {len(results_map)} results from {result_path}")
    return results_map


def check_retriever_called_from_log(patient_id: str, debate_logs_dir: str) -> bool:
    """
    Check if retriever was called during multi-agent debate by reading log file.
    
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
    # Try to read the debate log file (naming pattern: debate_responses_{patient_id}.log)
    log_file = Path(debate_logs_dir) / f"debate_responses_{patient_id}.log"
    
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


def check_retriever_called_from_json(patient_id: str, debate_logs_dir: str) -> bool:
    """
    Alternative method: Check if retrieval JSON file exists.
    
    Args:
        patient_id: Patient ID (e.g., "10117_0")
        debate_logs_dir: Directory containing debate log files
        
    Returns:
        True if retrieval JSON exists, False otherwise
    """
    json_file = Path(debate_logs_dir) / f"retrieve_integrator_combined_{patient_id}.json"
    return json_file.exists()


def estimate_prompt_tokens(text: str) -> int:
    """
    Estimate token count from text (rough approximation: 1 token ≈ 4 chars).
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def extract_candidate_table(
    data_adapter: KAREDataAdapter,
    cot_results_path: str,
    rag_qwen_results_path: str,
    rag_r1_results_path: str,
    output_path: str,
    rag_qwen_logs_dir: str = None,
    rag_r1_logs_dir: str = None
) -> pd.DataFrame:
    """
    Extract candidate table from dataset and multi-agent results.
    
    Args:
        data_adapter: KARE data adapter with test data
        cot_results_path: Path to CoT results JSON
        rag_qwen_results_path: Path to RAG-Qwen results JSON
        rag_r1_results_path: Path to RAG-R1 results JSON
        output_path: Output path for parquet file
        rag_qwen_logs_dir: Path to RAG-Qwen debate logs directory (for retrieval detection)
        rag_r1_logs_dir: Path to RAG-R1 debate logs directory (for retrieval detection)
        
    Returns:
        DataFrame with candidate table
    """
    # Load multi-agent results
    print("Loading multi-agent results...")
    cot_results = load_multiagent_results(cot_results_path)
    rag_qwen_results = load_multiagent_results(rag_qwen_results_path)
    rag_r1_results = load_multiagent_results(rag_r1_results_path)
    
    # Build candidate table
    print("Building candidate table...")
    rows = []
    
    for idx in range(len(data_adapter.test_data)):
        sample = data_adapter.get_test_sample(idx)
        
        # Core identifiers
        sample_id = sample['patient_id']  # e.g., "10117_0"
        label = sample['ground_truth']
        
        # Get predictions from each run
        cot_result = cot_results.get(sample_id, {})
        rag_qwen_result = rag_qwen_results.get(sample_id, {})
        rag_r1_result = rag_r1_results.get(sample_id, {})
        
        pred_cot = cot_result.get('prediction')
        pred_rag_qwen = rag_qwen_result.get('prediction')
        pred_rag_r1 = rag_r1_result.get('prediction')
        
        # Correctness flags (only if prediction is not None)
        wrong_cot = (pred_cot is not None) and (pred_cot != label)
        wrong_rag_qwen = (pred_rag_qwen is not None) and (pred_rag_qwen != label)
        wrong_rag_r1 = (pred_rag_r1 is not None) and (pred_rag_r1 != label)
        
        # Retriever-call flags - check debate logs
        called_retriever_rag_qwen = False
        called_retriever_rag_r1 = False
        
        if rag_qwen_logs_dir:
            called_retriever_rag_qwen = check_retriever_called_from_log(sample_id, rag_qwen_logs_dir)
        
        if rag_r1_logs_dir:
            called_retriever_rag_r1 = check_retriever_called_from_log(sample_id, rag_r1_logs_dir)
        
        # Fallback flags (if prediction is None, it likely failed/fallback)
        fallback_cot = pred_cot is None
        fallback_rag_qwen = pred_rag_qwen is None
        fallback_rag_r1 = pred_rag_r1 is None
        
        # Complexity features
        prompt_len_tokens = estimate_prompt_tokens(sample['target_context'])
        
        # Derived difficulty scores
        wrong_count_3 = sum([wrong_cot, wrong_rag_qwen, wrong_rag_r1])
        rag_called_both = called_retriever_rag_qwen and called_retriever_rag_r1
        
        row = {
            # Core identifiers
            'sample_id': sample_id,
            'base_patient_id': sample['base_patient_id'],
            'visit_id': sample['visit_id'],
            'visit_index': sample['visit_index'],
            'label': label,
            
            # Multi-agent outcomes
            'pred_multi_cot': pred_cot,
            'pred_multi_rag_qwen': pred_rag_qwen,
            'pred_multi_rag_r1': pred_rag_r1,
            
            # Correctness flags
            'wrong_multi_cot': wrong_cot,
            'wrong_multi_rag_qwen': wrong_rag_qwen,
            'wrong_multi_rag_r1': wrong_rag_r1,
            
            # Retriever-call flags
            'called_retriever_rag_qwen': called_retriever_rag_qwen,
            'called_retriever_rag_r1': called_retriever_rag_r1,
            
            # Fallback/parsing flags
            'fallback_multi_cot': fallback_cot,
            'fallback_multi_rag_qwen': fallback_rag_qwen,
            'fallback_multi_rag_r1': fallback_rag_r1,
            
            # Complexity features
            'prompt_len_tokens_target': prompt_len_tokens,
            
            # Derived difficulty scores
            'wrong_count_3': wrong_count_3,
            'rag_called_both': rag_called_both,
            
            # Store raw data for later access
            'patient_context': sample['target_context'],
            'positive_similars': sample['positive_similars'],
            'negative_similars': sample['negative_similars'],
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Print summary statistics
    print("\n=== Candidate Table Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (label=1): {(df['label'] == 1).sum()}")
    print(f"Negative samples (label=0): {(df['label'] == 0).sum()}")
    print(f"\nPrediction availability:")
    print(f"  CoT: {df['pred_multi_cot'].notna().sum()} / {len(df)}")
    print(f"  RAG-Qwen: {df['pred_multi_rag_qwen'].notna().sum()} / {len(df)}")
    print(f"  RAG-R1: {df['pred_multi_rag_r1'].notna().sum()} / {len(df)}")
    print(f"\nWrong predictions:")
    print(f"  CoT: {df['wrong_multi_cot'].sum()}")
    print(f"  RAG-Qwen: {df['wrong_multi_rag_qwen'].sum()}")
    print(f"  RAG-R1: {df['wrong_multi_rag_r1'].sum()}")
    print(f"\nRetriever usage:")
    print(f"  RAG-Qwen: {df['called_retriever_rag_qwen'].sum()}")
    print(f"  RAG-R1: {df['called_retriever_rag_r1'].sum()}")
    print(f"  Both: {df['rag_called_both'].sum()}")
    print(f"\nDifficulty distribution (wrong_count_3):")
    print(df['wrong_count_3'].value_counts().sort_index())
    
    # Save to parquet
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Candidate table saved to: {output_path}")
    
    return df


def main():
    """Main function."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    
    # Multi-agent result paths
    cot_results_path = results_dir / "cot_mor_Qwen_Qwen2.5_7B_Instruct" / "kare_debate_mortality_results.json"
    rag_qwen_results_path = results_dir / "rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8" / "kare_debate_mortality_results.json"
    rag_r1_results_path = results_dir / "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8" / "kare_debate_mortality_results.json"
    
    # Debate logs directories (for retrieval detection)
    rag_qwen_logs_dir = results_dir / "rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8" / "debate_logs"
    rag_r1_logs_dir = results_dir / "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8" / "debate_logs"
    
    # Output path
    output_dir = Path(__file__).parent.parent / "cache"
    output_path = output_dir / "candidate_table.parquet"
    
    print("=== KARE Metadata Extraction ===")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output: {output_path}")
    
    # Check if debate logs exist
    if rag_qwen_logs_dir.exists():
        print(f"✓ RAG-Qwen logs directory found: {rag_qwen_logs_dir}")
    else:
        print(f"⚠ RAG-Qwen logs directory not found: {rag_qwen_logs_dir}")
        print(f"  Retrieval detection for RAG-Qwen will be disabled.")
        rag_qwen_logs_dir = None
    
    if rag_r1_logs_dir.exists():
        print(f"✓ RAG-R1 logs directory found: {rag_r1_logs_dir}")
    else:
        print(f"⚠ RAG-R1 logs directory not found: {rag_r1_logs_dir}")
        print(f"  Retrieval detection for RAG-R1 will be disabled.")
        rag_r1_logs_dir = None
    
    print()
    
    # Load KARE data adapter
    print("Loading KARE test dataset...")
    data_adapter = KAREDataAdapter(base_path=str(data_dir), split="test")
    
    # Extract candidate table
    df = extract_candidate_table(
        data_adapter=data_adapter,
        cot_results_path=str(cot_results_path),
        rag_qwen_results_path=str(rag_qwen_results_path),
        rag_r1_results_path=str(rag_r1_results_path),
        output_path=str(output_path),
        rag_qwen_logs_dir=str(rag_qwen_logs_dir) if rag_qwen_logs_dir else None,
        rag_r1_logs_dir=str(rag_r1_logs_dir) if rag_r1_logs_dir else None
    )
    
    print("\n=== Extraction Complete ===")


if __name__ == "__main__":
    main()
