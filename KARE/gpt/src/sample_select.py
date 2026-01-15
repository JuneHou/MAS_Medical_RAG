#!/usr/bin/env python3
"""
Sample selection for GPT swap experiments.
Selects diagnostic slice: all positives + hard negatives by consensus wrongness.

Selection strategy:
- 54 positives: all positive samples (label=1)
- 46 negatives: balanced selection for difficulty comparison
  - 23 from hardest available: 1 from N3 (all wrong) + 22 from N2 (2 wrong)
  - 23 from easiest: N0 (all correct)
  
Total: 100 samples for GPT experiments
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd


def select_samples(
    candidate_table_path: str,
    target_positives: int = 54,
    target_negatives: int = 46,
    output_manifest_path: str = None,
    output_metadata_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select samples for GPT swap experiments.
    
    Args:
        candidate_table_path: Path to candidate_table.parquet
        target_positives: Target number of positive samples (default: all = 54)
        target_negatives: Target number of negative samples (default: 46)
        output_manifest_path: Output path for sample manifest CSV
        output_metadata_path: Output path for metadata JSON
        
    Returns:
        Tuple of (selected_samples_df, metadata_dict)
    """
    # Load candidate table
    print(f"Loading candidate table from: {candidate_table_path}")
    df = pd.read_parquet(candidate_table_path)
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (label=1): {(df['label'] == 1).sum()}")
    print(f"Negative samples (label=0): {(df['label'] == 0).sum()}")
    
    # Step 1: Select all positives
    positives = df[df['label'] == 1].copy()
    
    actual_positives = len(positives)
    if actual_positives != target_positives:
        print(f"\nWarning: Found {actual_positives} positives, expected {target_positives}")
    
    positives['split_tag'] = 'pos_all'
    print(f"\n✓ Selected {len(positives)} positive samples (split_tag: pos_all)")
    
    # Step 2: Select negatives by hardness
    negatives_df = df[df['label'] == 0].copy()
    
    # Define negative pools by wrong_count_3
    pool_n3 = negatives_df[
        (negatives_df['wrong_multi_cot'] == True) &
        (negatives_df['wrong_multi_rag_qwen'] == True) &
        (negatives_df['wrong_multi_rag_r1'] == True)
    ].copy()
    
    pool_n2 = negatives_df[negatives_df['wrong_count_3'] == 2].copy()
    pool_n0 = negatives_df[negatives_df['wrong_count_3'] == 0].copy()
    
    print(f"\n=== Negative Pool Sizes ===")
    print(f"N3 (wrong in all 3 runs): {len(pool_n3)}")
    print(f"N2 (wrong in 2 runs): {len(pool_n2)}")
    print(f"N0 (correct in all 3 runs): {len(pool_n0)}")
    
    # Select negatives with balanced strategy: 23 from hardest (N3+N2) + 23 from easiest (N0)
    selected_negatives = []
    target_hard = 23
    target_easy = 23
    
    # Step 1: Select from hardest available (N3 + N2)
    hard_samples = []
    
    # Take all from N3 first
    if len(pool_n3) > 0:
        pool_n3['split_tag'] = 'neg_hard'
        pool_n3_sorted = pool_n3.sort_values(
            by=['prompt_len_tokens_target', 'sample_id'],
            ascending=[False, True]
        )
        hard_samples.append(pool_n3_sorted)
        print(f"\n✓ Selected {len(pool_n3_sorted)} from N3 for hard pool")
    
    # Calculate how many more we need from N2
    current_hard_count = sum(len(s) for s in hard_samples)
    needed_from_n2 = target_hard - current_hard_count
    
    # Take remaining from N2
    if needed_from_n2 > 0 and len(pool_n2) > 0:
        pool_n2['split_tag'] = 'neg_hard'
        pool_n2_sorted = pool_n2.sort_values(
            by=['prompt_len_tokens_target', 'sample_id'],
            ascending=[False, True]
        )
        to_take_n2 = min(needed_from_n2, len(pool_n2_sorted))
        hard_samples.append(pool_n2_sorted.head(to_take_n2))
        print(f"✓ Selected {to_take_n2} from N2 for hard pool")
    
    # Combine hard samples
    if hard_samples:
        selected_negatives.extend(hard_samples)
        total_hard = sum(len(s) for s in hard_samples)
        print(f"  Total hard samples: {total_hard} (target: {target_hard})")
    
    # Step 2: Select from easiest (N0)
    if len(pool_n0) > 0:
        pool_n0['split_tag'] = 'neg_easy'
        # Sort for determinism
        pool_n0_sorted = pool_n0.sort_values(
            by=['rag_called_both', 'prompt_len_tokens_target', 'sample_id'],
            ascending=[False, False, True]
        )
        to_take_n0 = min(target_easy, len(pool_n0_sorted))
        selected_negatives.append(pool_n0_sorted.head(to_take_n0))
        print(f"✓ Selected {to_take_n0} from N0 for easy pool (split_tag: neg_easy)")
    else:
        to_take_n0 = 0
        print(f"⚠ No samples in N0 pool")
    
    # Check if we got the target amount
    total_selected = sum(len(s) for s in selected_negatives)
    if total_selected < target_negatives:
        print(f"\n⚠ Warning: Only selected {total_selected} negatives, target was {target_negatives}")
        print(f"  N3 available: {len(pool_n3)}, N2 available: {len(pool_n2)}, N0 available: {len(pool_n0)}")
    
    # Combine all selected negatives
    if selected_negatives:
        negatives = pd.concat(selected_negatives, ignore_index=True)
    else:
        negatives = pd.DataFrame()
    
    print(f"\n=== Selection Summary ===")
    print(f"Total negatives selected: {len(negatives)} (target: {target_negatives})")
    if len(negatives) < target_negatives:
        print(f"Warning: Only found {len(negatives)} negatives, needed {target_negatives}")
    
    # Combine positives and negatives
    selected = pd.concat([positives, negatives], ignore_index=True)
    
    print(f"\n=== Final Selection ===")
    print(f"Total samples: {len(selected)}")
    print(f"Positives: {len(positives)}")
    print(f"Negatives: {len(negatives)}")
    print(f"\nNegative breakdown by split_tag:")
    if len(negatives) > 0:
        print(negatives['split_tag'].value_counts())
    
    # Create metadata
    metadata = {
        'selection_criteria': {
            'target_positives': target_positives,
            'actual_positives': len(positives),
            'target_negatives': target_negatives,
            'actual_negatives': len(negatives),
            'total_samples': len(selected)
        },
        'negative_pools': {
            'n3_available': len(pool_n3),
            'n2_available': len(pool_n2),
            'n0_available': len(pool_n0),
            'hard_selected': len(negatives[negatives['split_tag'] == 'neg_hard']) if len(negatives) > 0 else 0,
            'hard_target': 23,
            'easy_selected': len(negatives[negatives['split_tag'] == 'neg_easy']) if len(negatives) > 0 else 0,
            'easy_target': 23,
        },
        'selection_rules': {
            'positive_selection': 'All samples with label=1',
            'negative_selection': [
                'Balanced strategy: 23 hard (N3+N2) + 23 easy (N0)',
                'Hard pool: All N3 (wrong in 3 runs) + top N2 (wrong in 2 runs) to reach 23',
                'Easy pool: 23 from N0 (correct in all 3 runs)'
            ],
            'sorting_criteria': [
                'Hard pool: prompt_len_tokens_target (longer first), sample_id (ascending)',
                'Easy pool: rag_called_both, prompt_len_tokens_target, sample_id'
            ]
        },
        'split_tags': {
            'pos_all': 'All positive samples (label=1)',
            'neg_hard': 'Hard negatives (N3 + top N2, wrong in 2-3 runs)',
            'neg_easy': 'Easy negatives (N0, correct in all 3 runs)'
        }
    }
    
    # Save manifest
    if output_manifest_path:
        manifest_df = selected[['sample_id', 'label', 'split_tag']].copy()
        output_dir = Path(output_manifest_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_df.to_csv(output_manifest_path, index=False)
        print(f"\n✓ Manifest saved to: {output_manifest_path}")
    
    # Save metadata
    if output_metadata_path:
        output_dir = Path(output_metadata_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {output_metadata_path}")
    
    # Also save full selected samples for later use
    selected_full_path = Path(output_manifest_path).parent / "selected_samples_full.parquet"
    selected.to_parquet(selected_full_path, index=False)
    print(f"✓ Full selected samples saved to: {selected_full_path}")
    
    return selected, metadata


def main():
    """Main function."""
    # Paths
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / "cache"
    manifest_dir = base_dir / "manifests"
    
    candidate_table_path = cache_dir / "candidate_table.parquet"
    manifest_path = manifest_dir / "samples_swap_core.csv"
    metadata_path = manifest_dir / "samples_swap_core_metadata.json"
    
    print("=== KARE Sample Selection for GPT Swap Experiments ===")
    print(f"Candidate table: {candidate_table_path}")
    print(f"Output manifest: {manifest_path}")
    print(f"Output metadata: {metadata_path}")
    print()
    
    # Check if candidate table exists
    if not candidate_table_path.exists():
        print(f"Error: Candidate table not found at {candidate_table_path}")
        print("Please run extract_metadata.py first to generate the candidate table.")
        return
    
    # Select samples
    selected, metadata = select_samples(
        candidate_table_path=str(candidate_table_path),
        target_positives=54,
        target_negatives=46,
        output_manifest_path=str(manifest_path),
        output_metadata_path=str(metadata_path)
    )
    
    print("\n=== Selection Complete ===")
    print(f"Selected {len(selected)} samples total")
    print(f"  Positives: {(selected['label'] == 1).sum()}")
    print(f"  Negatives: {(selected['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
