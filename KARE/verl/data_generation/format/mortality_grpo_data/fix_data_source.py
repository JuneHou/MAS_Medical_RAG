#!/usr/bin/env python3
"""
Add data_source column to GRPO training data.

VERL's naive reward loop expects data_source in non_tensor_batch.
We'll use the existing source_dir field as data_source.
"""

import pandas as pd
import sys

def fix_parquet_file(input_path: str, output_path: str):
    """Add data_source and reward_model columns to parquet file."""
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Add data_source column (use source_dir or a constant)
    if 'source_dir' in df.columns:
        df['data_source'] = df['source_dir']
        print("Added data_source column from source_dir")
    else:
        df['data_source'] = 'kare_mortality'
        print("Added constant data_source='kare_mortality'")
    
    # Add reward_model column (required by VERL naive reward loop)
    # ground_truth is a placeholder - we only check format, not correctness
    df['reward_model'] = [{"ground_truth": "__FORMAT_ONLY__"} for _ in range(len(df))]
    print("Added reward_model column with __FORMAT_ONLY__ placeholder")
    
    print(f"New columns: {df.columns.tolist()}")
    
    # Save fixed parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"First row data_source: {df.iloc[0]['data_source']}")
    print(f"First row reward_model: {df.iloc[0]['reward_model']}")

if __name__ == "__main__":
    base_dir = "/data/wang/junh/githubs/Debate/KARE/verl/data_generation/mortality_grpo_data"
    
    # Fix train and test files
    for split in ['train', 'test']:
        input_path = f"{base_dir}/{split}.parquet"
        output_path = input_path  # Overwrite original
        
        print(f"\n{'='*60}")
        print(f"Processing {split}.parquet")
        print('='*60)
        fix_parquet_file(input_path, output_path)
    
    print("\n" + "="*60)
    print("✓ Data source field added to all files")
    print("="*60)
