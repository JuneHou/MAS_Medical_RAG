#!/usr/bin/env python3
"""
Prepare balanced KARE mortality data for Search-R1 single-agent training.
Loads from balanced JSON file and generates parquet format.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Add KARE paths
sys.path.insert(0, '/data/wang/junh/githubs/Debate/KARE')
from kare_data_adapter import KAREDataAdapter


def load_balanced_json(json_path: str) -> Tuple[List[int], Dict]:
    """
    Load balanced sample indices from JSON file.
    
    Returns:
        (sample_indices, metadata)
    """
    print(f"Loading balanced sample file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    indices = metadata['all_indices']
    
    print(f"Loaded {len(indices)} balanced samples:")
    print(f"  Positive (mortality=1): {metadata['n_positive']}")
    print(f"  Negative (survival=0): {metadata['n_negative']}")
    
    return indices, metadata





def create_single_agent_prompt(sample: Dict, use_probability_format: bool = False) -> str:
    """
    Create Search-R1 prompt for single-agent mortality prediction.
    No debate context - just target patient + similar cases + search capability.
    
    Args:
        sample: Patient sample from KAREDataAdapter with:
                - target_context: formatted patient EHR
                - positive_similars: similar patients with same outcome
                - negative_similars: similar patients with different outcome
        use_probability_format: If True, request probability outputs (Experiment 2)
                                If False, request binary label (Experiment 1)
        
    Returns:
        Search-R1 formatted prompt
    """
    
    # Extract data from KAREDataAdapter sample
    target_context = sample.get('target_context', '')
    positive_similars = sample.get('positive_similars', 'No similar patients available.')
    negative_similars = sample.get('negative_similars', 'No similar patients available.')
    
    if use_probability_format:
        # Experiment 2: Request calibrated probabilities (aligned with downstream task)
        prompt = f"""You are a medical AI Clinical Assistant analyzing mortality and survival probabilities for the NEXT hospital visit.

You must conduct reasoning inside <think> and </think> tags every time you analyze information.
If you need additional medical evidence based on the patient's specific conditions, search for relevant clinical information by writing <search>custom query about prognosis</search> (e.g., "sepsis mortality prognosis elderly patients" or "heart failure survival outcomes").
The search engine will return medical literature between <information> and </information> tags.
You can search multiple times to gather comprehensive evidence.

Instructions:
1) Based on the patient's specific conditions, you may call <search>query</search> with a custom query about their prognosis
2) Review all available information including the target patient data and similar patient cases
3) Analyze BOTH mortality risk factors AND survival/protective factors
4) Be conservative: mortality is rare, so strong evidence is needed for high mortality probability

When confident in your assessment, provide comprehensive clinical reasoning and end with your final answer:
<answer>
MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)
</answer>

IMPORTANT: The two probabilities MUST sum to exactly 1.00

## Target Patient
{target_context}

## Similar Patient Cases
For reference, here are similar patient cases with known outcomes:

### Similar Patients (Same Outcome Expected)
{positive_similars}

### Similar Patients (Different Outcome Expected)
{negative_similars}"""
    else:
        # Experiment 1: Request binary label (original format)
        prompt = f"""You are a medical AI assistant specialized in mortality prediction. \
Your task is to predict whether a patient will DIE (1) or SURVIVE (0) in their next hospital visit.
IMPORTANT: Mortality is rare - only predict death (1) if evidence STRONGLY supports it. When uncertain, predict survival (0).

You must conduct reasoning inside <think> and </think> tags every time you analyze information.
If you need additional medical evidence, search for relevant clinical information by writing <search>query</search>.
The search engine will return medical literature between <information> and </information> tags.
You can search multiple times to gather comprehensive evidence.
When confident in your prediction, provide your final answer as <answer>0</answer> (survival) or <answer>1</answer> (mortality).

## Target Patient
{target_context}

## Similar Patient Cases
For reference, here are similar patient cases with known outcomes:

### Similar Patients (Same Outcome Expected)
{positive_similars}

### Similar Patients (Different Outcome Expected)
{negative_similars}

## Task
Analyze the target patient's clinical data, consider the similar cases, and predict the mortality outcome for the next visit.
You may search for additional medical evidence to support your reasoning."""

    return prompt


def format_sample_for_searchr1(sample: Dict, split: str = 'train', use_probability_format: bool = False) -> Dict:
    """
    Format a single KARE sample for Search-R1 training.
    
    Args:
        sample: KARE data sample from adapter
        split: 'train' or 'val'
        use_probability_format: If True, use Experiment 2 format (probabilities)
        
    Returns:
        Search-R1 formatted dictionary for parquet file
    """
    
    # Create prompt
    prompt_text = create_single_agent_prompt(sample, use_probability_format=use_probability_format)
    
    # Set ability based on experiment type
    ability_name = "kare_mortality_probability" if use_probability_format else "medical-mortality-prediction"
    
    # Format in Search-R1 structure (matching veRL expected format)
    return {
        "data_source": "kare_mortality_single_agent",
        "prompt": [{
            "role": "user",
            "content": prompt_text,
        }],
        "ability": ability_name,
        "reward_model": {
            "style": "rule",  # Rule-based reward: binary EM (Exp 1) or probability (Exp 2)
            "ground_truth": {
                "target": [str(sample['ground_truth'])]  # "0" or "1"
            }
        },
        "extra_info": {
            'split': split,
            'patient_id': sample['patient_id'],
            'visit_id': sample.get('visit_id', ''),
            'original_index': sample.get('original_index', -1)
        }
    }


def create_searchr1_dataset(
    balanced_json_path: str,
    data_split: str = 'train',
    output_dir: str = 'data/kare_mortality_single_agent',
    use_probability_format: bool = False
):
    """
    Create Search-R1 dataset from balanced JSON file.
    
    Args:
        balanced_json_path: Path to balanced JSON file (e.g., train_balanced_100pos_100neg.json)
        use_probability_format: If True, generate Experiment 2 data (probability outputs)
                                If False, generate Experiment 1 data (binary labels)
        data_split: 'train' or 'val' (follows veRL convention)
        output_dir: Output directory for parquet files
    """
    
    print("=" * 80)
    print("KARE Mortality → Search-R1 Single-Agent Data Preparation (Balanced)")
    print("=" * 80)
    
    # Load balanced sample indices
    sample_indices, metadata = load_balanced_json(balanced_json_path)
    
    # Initialize KARE data adapter
    print(f"\nLoading KARE {data_split} data adapter...")
    data_adapter = KAREDataAdapter(
        base_path="/data/wang/junh/datasets/KARE",
        split=data_split
    )
    print(f"  Total samples in {data_split} split: {len(data_adapter.data)}")
    
    # Process samples
    print(f"\nProcessing {len(sample_indices)} balanced samples...")
    formatted_samples = []
    error_count = 0
    
    for idx in tqdm(sample_indices, desc="Formatting samples"):
        try:
            # Get sample from adapter (uses get_test_sample regardless of split)
            sample = data_adapter.get_test_sample(idx)
            
            # Add original index for tracking
            sample['original_index'] = idx
            
            # Format for Search-R1
            formatted = format_sample_for_searchr1(sample, split=data_split, use_probability_format=use_probability_format)
            formatted_samples.append(formatted)
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    print(f"\n✓ Successfully formatted {len(formatted_samples)} samples ({error_count} errors)")
    
    # Handle case where all samples failed
    if len(formatted_samples) == 0:
        print("\n" + "=" * 80)
        print("ERROR: No samples were successfully formatted!")
        print("=" * 80)
        print("Please check:")
        print("1. Balanced JSON indices match the data split")
        print("2. KAREDataAdapter can load the specified split")
        print("3. Sample indices are valid for the dataset")
        return None
    
    # Save as parquet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(formatted_samples)
    
    # Determine output filename based on split (veRL convention: train.parquet, val.parquet)
    if data_split == 'train':
        output_file = output_path / 'train.parquet'
    else:  # val
        output_file = output_path / 'val.parquet'
    
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Total samples: {len(formatted_samples)}")
    
    # Label distribution
    labels = [int(s['reward_model']['ground_truth']['target'][0]) for s in formatted_samples]
    survival_count = labels.count(0)
    mortality_count = labels.count(1)
    
    print(f"\nLabel Distribution:")
    print(f"  Survival (0): {survival_count} ({100*survival_count/len(labels):.1f}%)")
    print(f"  Mortality (1): {mortality_count} ({100*mortality_count/len(labels):.1f}%)")
    
    # Sample prompt preview
    print("\n" + "=" * 80)
    print("Sample Prompt Preview (first sample)")
    print("=" * 80)
    sample_prompt = formatted_samples[0]['prompt'][0]['content']
    print(sample_prompt[:1000] + "..." if len(sample_prompt) > 1000 else sample_prompt)
    
    print("\n" + "=" * 80)
    print("✓ Data preparation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Start MedRAG retrieval server:")
    print(f"   python searchr1/medrag_retrieval_server.py --port 8000")
    print(f"2. Test retrieval:")
    print(f"   python searchr1/test_medrag_server.py")
    print(f"3. Run training:")
    print(f"   bash searchr1/train_searchr1_single_agent.sh")
    print("=" * 80)
    
    return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare balanced KARE data for Search-R1 training'
    )
    parser.add_argument(
        '--balanced_json',
        type=str,
        required=True,
        help='Path to balanced JSON file (e.g., train_balanced_100pos_100neg.json)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Data split: "train" for training, "val" for validation during RL training'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/kare_mortality_single_agent',
        help='Output directory for parquet files'
    )
    parser.add_argument(
        '--probability',
        action='store_true',
        help='Use probability format (Experiment 2) instead of binary labels (Experiment 1)'
    )
    
    args = parser.parse_args()
    
    create_searchr1_dataset(
        balanced_json_path=args.balanced_json,
        data_split=args.split,
        output_dir=args.output_dir,
        use_probability_format=args.probability
    )
