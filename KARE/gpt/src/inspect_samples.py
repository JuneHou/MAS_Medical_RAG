#!/usr/bin/env python3
"""
Utility script to inspect and analyze the selected 100 samples.
Provides various views and statistics for the GPT swap experiments.
"""

import pandas as pd
import json
from pathlib import Path


def load_selected_samples(manifests_dir: str = None):
    """Load the selected 100 samples with full data."""
    if manifests_dir is None:
        manifests_dir = Path(__file__).parent.parent / "manifests"
    else:
        manifests_dir = Path(manifests_dir)
    
    # Load full data
    full_path = manifests_dir / "selected_samples_full.parquet"
    df = pd.read_parquet(full_path)
    
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics of selected samples."""
    print("=== Selected Samples Summary ===\n")
    print(f"Total samples: {len(df)}")
    print(f"Positives: {(df['label'] == 1).sum()}")
    print(f"Negatives: {(df['label'] == 0).sum()}")
    print()
    
    print("Negative breakdown by difficulty:")
    negatives = df[df['label'] == 0]
    print(negatives['split_tag'].value_counts())
    print()
    
    print("Wrong counts distribution:")
    print(df['wrong_count_3'].value_counts().sort_index())
    print()
    
    print("Retrieval usage in selected samples:")
    print(f"  RAG-Qwen: {df['called_retriever_rag_qwen'].sum()} / {len(df)}")
    print(f"  RAG-R1: {df['called_retriever_rag_r1'].sum()} / {len(df)}")
    print(f"  Both: {df['rag_called_both'].sum()} / {len(df)}")
    print()


def analyze_by_split_tag(df: pd.DataFrame):
    """Analyze samples grouped by split_tag."""
    print("=== Analysis by Split Tag ===\n")
    
    for tag in df['split_tag'].unique():
        subset = df[df['split_tag'] == tag]
        print(f"\n{tag} ({len(subset)} samples):")
        print(f"  Label distribution: {dict(subset['label'].value_counts())}")
        print(f"  Retrieval (Qwen): {subset['called_retriever_rag_qwen'].sum()} / {len(subset)}")
        print(f"  Retrieval (R1): {subset['called_retriever_rag_r1'].sum()} / {len(subset)}")
        print(f"  Avg prompt length: {subset['prompt_len_tokens_target'].mean():.0f} tokens")
        
        # Show prediction accuracy for each model
        if (subset['label'] == 0).all():  # Only for negatives
            print(f"  CoT accuracy: {((subset['pred_multi_cot'] == 0).sum() / len(subset) * 100):.1f}%")
            print(f"  RAG-Qwen accuracy: {((subset['pred_multi_rag_qwen'] == 0).sum() / len(subset) * 100):.1f}%")
            print(f"  RAG-R1 accuracy: {((subset['pred_multi_rag_r1'] == 0).sum() / len(subset) * 100):.1f}%")


def export_for_gpt_experiment(df: pd.DataFrame, output_path: str):
    """
    Export selected samples in a format ready for GPT experiments.
    
    Args:
        df: Selected samples DataFrame
        output_path: Output JSON file path
    """
    records = []
    
    for _, row in df.iterrows():
        record = {
            'sample_id': row['sample_id'],
            'label': int(row['label']),
            'split_tag': row['split_tag'],
            'patient_context': row['patient_context'],
            'positive_similars': row['positive_similars'],
            'negative_similars': row['negative_similars'],
            'metadata': {
                'wrong_count_3': int(row['wrong_count_3']),
                'prompt_len_tokens': int(row['prompt_len_tokens_target']),
                'called_retriever_rag_qwen': bool(row['called_retriever_rag_qwen']),
                'called_retriever_rag_r1': bool(row['called_retriever_rag_r1']),
                'baseline_predictions': {
                    'cot': int(row['pred_multi_cot']) if pd.notna(row['pred_multi_cot']) else None,
                    'rag_qwen': int(row['pred_multi_rag_qwen']) if pd.notna(row['pred_multi_rag_qwen']) else None,
                    'rag_r1': int(row['pred_multi_rag_r1']) if pd.notna(row['pred_multi_rag_r1']) else None,
                }
            }
        }
        records.append(record)
    
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"✓ Exported {len(records)} samples to: {output_path}")


def get_hardest_cases(df: pd.DataFrame, n: int = 10):
    """Get the n hardest cases (highest wrong_count_3, longest prompts)."""
    # Sort by wrong_count_3 (descending), then prompt length (descending)
    hardest = df.sort_values(
        by=['wrong_count_3', 'prompt_len_tokens_target'],
        ascending=[False, False]
    ).head(n)
    
    print(f"\n=== Top {n} Hardest Cases ===\n")
    for idx, row in hardest.iterrows():
        print(f"Sample: {row['sample_id']}")
        print(f"  Label: {row['label']}, Split: {row['split_tag']}")
        print(f"  Wrong count: {row['wrong_count_3']}/3")
        print(f"  Prompt tokens: {row['prompt_len_tokens_target']}")
        print(f"  Retrieval: Qwen={row['called_retriever_rag_qwen']}, R1={row['called_retriever_rag_r1']}")
        print(f"  Predictions: CoT={row['pred_multi_cot']}, Qwen={row['pred_multi_rag_qwen']}, R1={row['pred_multi_rag_r1']}")
        print()
    
    return hardest


def main():
    """Main function."""
    print("Loading selected samples...\n")
    df = load_selected_samples()
    
    # Print summary
    print_summary(df)
    
    # Analyze by split tag
    analyze_by_split_tag(df)
    
    # Show hardest cases
    get_hardest_cases(df, n=10)
    
    # Export for GPT experiments
    output_path = Path(__file__).parent.parent / "manifests" / "gpt_experiment_samples.json"
    export_for_gpt_experiment(df, str(output_path))
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
