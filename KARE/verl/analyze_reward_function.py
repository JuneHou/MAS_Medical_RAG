#!/usr/bin/env python3
"""
Demonstrate the new Brier-based reward function behavior across probability ranges.
This shows how the reward changes smoothly from -1 to +1.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the reward function to path
sys.path.append('/data/wang/junh/githubs/Debate/KARE/verl')
from reward_score.kare_prediction_reward import compute_score

def analyze_reward_function():
    """Analyze reward function behavior across probability range."""
    
    # Test probabilities from 0.0 to 1.0
    probs = np.linspace(0.0, 1.0, 101)
    
    # Test cases: (ground_truth, assessment_type)
    test_cases = [
        (0, "mortality", "Survival patient, mortality assessment"),
        (1, "mortality", "Mortality patient, mortality assessment"), 
        (0, "survival", "Survival patient, survival assessment"),
        (1, "survival", "Mortality patient, survival assessment")
    ]
    
    print("Brier-based Reward Function Analysis")
    print("=" * 60)
    print("Formula: r' = 1 - 2(p - y)²")
    print("Where p = predicted probability, y = target (0 or 1)")
    print()
    
    for gt, assessment, description in test_cases:
        print(f"{description}:")
        print(f"Ground truth: {gt}, Assessment: {assessment}")
        
        rewards = []
        for prob in probs:
            # Create mock response with the probability
            if assessment == "mortality":
                solution = f"Analysis...\nMORTALITY PROBABILITY: {prob:.2f}"
            else:
                solution = f"Analysis...\nSURVIVAL PROBABILITY: {prob:.2f}"
            
            reward = compute_score(
                solution_str=solution,
                ground_truth=gt,
                extra_info={'assessment_type': assessment}
            )
            rewards.append(reward)
        
        # Find key points
        max_idx = np.argmax(rewards)
        min_idx = np.argmin(rewards)
        
        print(f"  Maximum reward: {rewards[max_idx]:.3f} at p={probs[max_idx]:.2f}")
        print(f"  Minimum reward: {rewards[min_idx]:.3f} at p={probs[min_idx]:.2f}")
        print(f"  Random (p=0.5): {rewards[50]:.3f}")
        print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print()
    
    print("Key Properties:")
    print("- Continuous signal everywhere (no zero gradients)")
    print("- Perfect predictions (p = y) get reward = +1.0") 
    print("- Worst predictions (|p - y| = 1) get reward = -1.0")
    print("- Random predictions (p = 0.5) get reward = +0.5")
    print("- Smooth gradients encourage fine-tuning near optimal")

if __name__ == "__main__":
    analyze_reward_function()