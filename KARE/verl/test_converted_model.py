#!/usr/bin/env python3
"""
Quick test to verify the converted model works with VLLM.
"""

import os
import sys

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Add paths
sys.path.insert(0, "/data/wang/junh/githubs/mirage_medrag/MedRAG")
sys.path.insert(0, "/data/wang/junh/githubs/mirage_medrag/MedRAG/src")
sys.path.insert(0, "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src")

from run_medrag_vllm import VLLMWrapper

# Test the fine-tuned model
model_path = "/data/wang/junh/githubs/Debate/KARE/verl/models/format_enforcer_7b_step57"

print(f"Loading fine-tuned model from: {model_path}")
print("This may take 1-2 minutes...")

try:
    llm = VLLMWrapper(model_name=model_path, enable_thinking=True)
    print("\n✓ Model loaded successfully!")
    
    # Test with a simple prompt
    test_prompt = [
        {
            "role": "user",
            "content": "Patient has severe sepsis and respiratory failure. What is the mortality probability? End with: MORTALITY PROBABILITY: X.XX"
        }
    ]
    
    print("\nTesting model response...")
    response = llm.generate(test_prompt, use_tqdm=False, return_str=True)
    
    print("\n" + "="*80)
    print("MODEL RESPONSE:")
    print("="*80)
    print(response)
    print("="*80)
    
    # Check if format is present
    import re
    pattern = r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        prob = float(match.group(1))
        if 0.0 <= prob <= 1.0:
            print(f"\n✓ SUCCESS! Model output correct format: MORTALITY PROBABILITY: {prob}")
        else:
            print(f"\n⚠ WARNING: Format found but probability out of range: {prob}")
    else:
        print("\n✗ ERROR: No valid format found in response")
    
except Exception as e:
    print(f"\n✗ ERROR loading or testing model: {e}")
    import traceback
    traceback.print_exc()
