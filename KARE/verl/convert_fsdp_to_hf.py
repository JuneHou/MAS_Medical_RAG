#!/usr/bin/env python3
"""
Convert VERL FSDP checkpoint to HuggingFace format for use as integrator model.

This is a wrapper around VERL's built-in model merger.

Usage:
    python convert_fsdp_to_hf.py --checkpoint_dir /path/to/checkpoint --output_dir /path/to/output
"""

import argparse
import os
import subprocess
import sys


def convert_checkpoint(checkpoint_dir: str, output_dir: str):
    """
    Convert VERL FSDP checkpoint to HuggingFace format using VERL's model merger.
    
    Args:
        checkpoint_dir: Path to FSDP checkpoint (e.g., checkpoints/global_step_57/actor)
        output_dir: Path to save HuggingFace format model
    """
    print(f"Converting FSDP checkpoint: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    # Validate paths
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    fsdp_config = os.path.join(checkpoint_dir, "fsdp_config.json")
    if not os.path.exists(fsdp_config):
        print(f"Error: FSDP config not found: {fsdp_config}")
        print(f"Make sure you're pointing to the 'actor' subdirectory")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use VERL's built-in model merger
    print("\nRunning VERL model merger...")
    cmd = [
        "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", checkpoint_dir,
        "--target_dir", output_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Conversion complete! Model saved to: {output_dir}")
        print(f"\nYou can now use this model with VLLM:")
        print(f"  model_name = '{output_dir}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Conversion failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert VERL FSDP checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to FSDP checkpoint directory (e.g., checkpoints/global_step_57/actor)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save HuggingFace format model")
    
    args = parser.parse_args()
    
    # Convert
    success = convert_checkpoint(args.checkpoint_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
