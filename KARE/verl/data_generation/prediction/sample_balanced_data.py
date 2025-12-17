#!/usr/bin/env python3
"""
Sample balanced data: 100 positive (mortality=1) and 100 negative (mortality=0) samples.
This ensures we have balanced representation for GRPO training.
"""

import json
import sys
from pathlib import Path

def sample_balanced_data(
    split: str = 'val',
    n_positive: int = 100,
    n_negative: int = 100,
    output_file: str = None
):
    """
    Sample balanced data from train/val split.
    
    Args:
        split: 'train' or 'val'
        n_positive: Number of positive samples (mortality=1)
        n_negative: Number of negative samples (mortality=0)
        output_file: Output JSON file path
    
    Returns:
        Dictionary with sampled indices and statistics
    """
    # Load split data
    data_path = f"/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_{split}.json"
    
    print(f"Loading {split} split from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples in {split} split: {len(data)}")
    
    # Load similar patients file to check coverage
    similar_patients_file = "/data/wang/junh/datasets/KARE/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json"
    print(f"Loading similar patients from: {similar_patients_file}")
    with open(similar_patients_file, 'r') as f:
        similar_patients = json.load(f)
    
    print(f"Total patients with similar contexts: {len(similar_patients)}")
    
    # Separate by label - only include patients with BOTH positive AND negative similar patients
    positive_indices = []
    negative_indices = []
    skipped_no_similar = 0
    
    for i, sample in enumerate(data):
        label = sample.get('label', 0)
        
        # Construct KARE patient ID
        patient_id = str(sample['patient_id'])
        conditions_visits = sample.get('conditions', [[]])
        procedures_visits = sample.get('procedures', [[]])
        drugs_visits = sample.get('drugs', [[]])
        num_visits = max(len(conditions_visits), len(procedures_visits), len(drugs_visits))
        visit_index = num_visits - 1
        kare_patient_id = f"{patient_id}_{visit_index}"
        
        # Check if patient has BOTH positive and negative similar patients
        if kare_patient_id in similar_patients:
            similar_data = similar_patients[kare_patient_id]
            has_positive = similar_data.get('positive', []) and similar_data['positive'][0] != "None"
            has_negative = similar_data.get('negative', []) and similar_data['negative'][0] != "None"
            
            # Only include if has BOTH positive and negative similar patients
            if has_positive and has_negative:
                if label == 1:
                    positive_indices.append(i)
                else:
                    negative_indices.append(i)
            else:
                skipped_no_similar += 1
        else:
            skipped_no_similar += 1
    
    print(f"\nLabel distribution (with both positive & negative similar patients):")
    print(f"  Positive (mortality=1): {len(positive_indices)} samples available")
    print(f"  Negative (survival=0): {len(negative_indices)} samples available")
    
    # Automatically adjust to get exactly n_positive + n_negative samples
    # If one class has fewer samples, take all from that class and balance from the other
    if len(positive_indices) < n_positive:
        n_positive = len(positive_indices)
        print(f"\n⚠️  Adjusting: Using all {n_positive} positive samples")
    
    if len(negative_indices) < n_negative:
        n_negative = len(negative_indices)
        print(f"\n⚠️  Adjusting: Using all {n_negative} negative samples")
    
    # Sample balanced data
    import random
    random.seed(42)  # For reproducibility
    
    sampled_positive = random.sample(positive_indices, n_positive)
    sampled_negative = random.sample(negative_indices, n_negative)
    
    # Combine and sort
    all_sampled_indices = sorted(sampled_positive + sampled_negative)
    
    print(f"\n✅ Sampled {len(all_sampled_indices)} total samples:")
    print(f"  Positive (mortality=1): {len(sampled_positive)} samples")
    print(f"  Negative (survival=0): {len(sampled_negative)} samples")
    print(f"  All samples have complete similar patient contexts (positive & negative)")
    
    # Create sampled data with full context
    sampled_data = []
    for idx in all_sampled_indices:
        sample = data[idx]
        
        # Add index for reference
        sample_with_idx = {
            'original_index': idx,
            **sample
        }
        sampled_data.append(sample_with_idx)
    
    # Save to output file
    if output_file is None:
        output_file = f"/data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction/{split}_balanced_{n_positive}pos_{n_negative}neg.json"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'split': split,
                'n_positive': len(sampled_positive),
                'n_negative': len(sampled_negative),
                'total_samples': len(all_sampled_indices),
                'positive_indices': sampled_positive,
                'negative_indices': sampled_negative,
                'all_indices': all_sampled_indices
            },
            'samples': sampled_data
        }, f, indent=2)
    
    print(f"\n✅ Saved balanced data to: {output_file}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print sample patient IDs for verification
    print(f"\n📋 Sample patient IDs (first 5 positive, first 5 negative):")
    print(f"\nPositive (mortality=1):")
    for idx in sampled_positive[:5]:
        patient_id = data[idx]['patient_id']
        print(f"  Index {idx}: Patient {patient_id}")
    
    print(f"\nNegative (survival=0):")
    for idx in sampled_negative[:5]:
        patient_id = data[idx]['patient_id']
        print(f"  Index {idx}: Patient {patient_id}")
    
    return {
        'positive_indices': sampled_positive,
        'negative_indices': sampled_negative,
        'all_indices': all_sampled_indices,
        'output_file': output_file
    }


def load_and_verify_similar_patients(sampled_data_file: str):
    """
    Verify that all sampled patients have similar patient contexts.
    Since we now filter during sampling, this should always show 100% coverage.
    """
    print("\n" + "="*80)
    print("Verifying similar patient contexts for sampled data...")
    print("="*80)
    
    # Load sampled data
    with open(sampled_data_file, 'r') as f:
        sampled_info = json.load(f)
    
    samples = sampled_info['samples']
    
    # Load similar patients file
    similar_patients_file = "/data/wang/junh/datasets/KARE/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json"
    
    print(f"Loading similar patients from: {similar_patients_file}")
    with open(similar_patients_file, 'r') as f:
        similar_patients = json.load(f)
    
    print(f"Total patients with similar contexts: {len(similar_patients)}")
    
    # Check coverage (should be 100% now since we filter during sampling)
    complete_coverage = 0
    
    for sample in samples:
        patient_id = str(sample['patient_id'])
        
        # Construct KARE's temporal patient ID format (patient_id_visit_index)
        conditions_visits = sample.get('conditions', [[]])
        procedures_visits = sample.get('procedures', [[]])
        drugs_visits = sample.get('drugs', [[]])
        num_visits = max(len(conditions_visits), len(procedures_visits), len(drugs_visits))
        visit_index = num_visits - 1
        kare_patient_id = f"{patient_id}_{visit_index}"
        
        # Check if similar patients exist with both positive and negative
        if kare_patient_id in similar_patients:
            similar_data = similar_patients[kare_patient_id]
            has_positive = similar_data.get('positive', []) and similar_data['positive'][0] != "None"
            has_negative = similar_data.get('negative', []) and similar_data['negative'][0] != "None"
            
            if has_positive and has_negative:
                complete_coverage += 1
    
    print(f"\n✅ Coverage: {complete_coverage}/{len(samples)} patients have complete similar patient contexts")
    
    if complete_coverage == len(samples):
        print("✅ Perfect! All sampled patients have both positive & negative similar patients!")
    else:
        print(f"⚠️  Warning: {len(samples) - complete_coverage} patients missing complete similar contexts")
    
    return complete_coverage


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample balanced data for GRPO training")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Data split to sample from (default: train for GRPO training)')
    parser.add_argument('--n_positive', type=int, default=100,
                       help='Number of positive samples (mortality=1)')
    parser.add_argument('--n_negative', type=int, default=100,
                       help='Number of negative samples (survival=0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--verify', action='store_true',
                       help='Verify similar patient coverage')
    
    args = parser.parse_args()
    
    # Sample balanced data
    result = sample_balanced_data(
        split=args.split,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        output_file=args.output
    )
    
    # Verify similar patient coverage if requested
    if args.verify:
        load_and_verify_similar_patients(result['output_file'])
    
    print("\n" + "="*80)
    print("✅ Balanced sampling complete!")
    print("="*80)
    print(f"\nNext step: Generate training data using the sampled indices:")
    print(f"python generate_prediction_training_data.py \\")
    print(f"  --split {args.split} \\")
    print(f"  --balanced_file {result['output_file']} \\")
    print(f"  --gpus 6,7")
