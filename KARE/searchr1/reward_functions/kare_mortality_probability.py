#!/usr/bin/env python3
"""
Custom reward function for Search-R1 KARE mortality probability prediction.

Option A: Positive-Only Reward (Recommended)
- Reward = mortality_prob if GT=1, else (1 - mortality_prob)
- Range: [0.0, 1.0]
- Smoother optimization, better for initial Search-R1 training
"""

import re
import random

# Regex patterns to extract probabilities from <answer> tag
MORT_PROB_RE = re.compile(r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
SURV_PROB_RE = re.compile(r'SURVIVAL PROBABILITY:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)


def extract_probabilities(solution_str):
    """
    Extract mortality and survival probabilities from model output.
    
    Args:
        solution_str: Full model response text
        
    Returns:
        tuple: (mortality_prob, survival_prob) or (None, None) if not found
    """
    if not solution_str:
        return None, None
    
    mort_match = MORT_PROB_RE.search(solution_str)
    surv_match = SURV_PROB_RE.search(solution_str)
    
    mort_prob = float(mort_match.group(1)) if mort_match else None
    surv_prob = float(surv_match.group(1)) if surv_match else None
    
    return mort_prob, surv_prob


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1., **kwargs):
    """
    Option A: Positive-only reward based on probability calibration.
    
    This is the main function called by Search-R1's reward system.
    Compatible with Search-R1's reward function signature.
    
    Reward formula:
    - Mortality cases (GT=1): reward = mortality_prob
    - Survival cases (GT=0): reward = 1 - mortality_prob
    
    Args:
        solution_str: Full model response with <answer>...</answer> tag
        ground_truth: Dict with {'target': ['0']} or {'target': ['1']}
        method: Compatibility parameter (ignored)
        format_score: Score for invalid format (default: 0.0)
        score: Maximum score for valid format (default: 1.0, but overridden by actual prob)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        float: Reward in range [0.0, 1.0]
    """
    # Random sampling for debug logging (1 in 64 chance)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"{'='*60}")
        print(f"KARE Mortality Probability Reward Debug")
        print(f"{'='*60}")
        print(f"Ground truth: {ground_truth}")
    
    # Extract probabilities from <answer> tag
    mort_prob, surv_prob = extract_probabilities(solution_str)
    
    if do_print:
        print(f"Extracted mortality prob: {mort_prob}")
        print(f"Extracted survival prob: {surv_prob}")
    
    # Validation: mortality probability must be present
    if mort_prob is None:
        if do_print:
            print(f"REWARD: {format_score} (no mortality probability found)")
            print(f"Solution string excerpt: {solution_str[:500]}...")
        return format_score  # Invalid format
    
    # Validation: probability must be in valid range [0.0, 1.0]
    if not (0.0 <= mort_prob <= 1.0):
        if do_print:
            print(f"REWARD: {format_score} (mortality probability out of range: {mort_prob})")
        return format_score
    
    # Optional validation: Check if probabilities sum to 1.0 (with tolerance)
    # This encourages the model to output both probabilities correctly
    if surv_prob is not None:
        prob_sum = mort_prob + surv_prob
        if abs(prob_sum - 1.0) > 0.05:  # 5% tolerance
            if do_print:
                print(f"REWARD: {format_score} (probabilities don't sum to 1.0: {mort_prob} + {surv_prob} = {prob_sum})")
            return format_score
    
    # Extract ground truth label from parquet format
    # ground_truth is a dict like {'target': ['0']} or {'target': ['1']}
    try:
        gt_target = ground_truth['target']
        # Handle both list and single value cases
        if isinstance(gt_target, list):
            gt_label = int(gt_target[0])
        else:
            gt_label = int(gt_target)
    except (KeyError, ValueError, IndexError, TypeError) as e:
        if do_print:
            print(f"REWARD: {format_score} (failed to parse ground truth: {e})")
        return format_score
    
    # Compute reward based on ground truth
    if gt_label == 1:  # Mortality case
        # Reward high mortality predictions
        reward = mort_prob
        reason = f"Mortality case: reward = mort_prob = {mort_prob:.3f}"
    else:  # Survival case (gt_label == 0)
        # Reward high survival predictions (= low mortality)
        reward = 1.0 - mort_prob
        reason = f"Survival case: reward = 1 - mort_prob = {reward:.3f}"
    
    if do_print:
        print(f"Ground truth label: {gt_label} ({'mortality' if gt_label == 1 else 'survival'})")
        print(f"Mortality probability: {mort_prob:.3f}")
        print(f"Survival probability: {1.0 - mort_prob:.3f} (calculated)")
        print(f"REWARD: {reward:.3f} ({reason})")
        print(f"{'='*60}\n")
    
    return reward  # Range: [0.0, 1.0]


# Batch version for efficiency (if Search-R1 supports it)
def compute_score_batch(solution_strs, ground_truths, **kwargs):
    """
    Batch version of compute_score for efficiency.
    
    Args:
        solution_strs: List of model output texts
        ground_truths: List of ground truth dicts
        **kwargs: Additional keyword arguments passed to compute_score
        
    Returns:
        List of rewards
    """
    rewards = []
    for solution_str, ground_truth in zip(solution_strs, ground_truths):
        reward = compute_score(solution_str, ground_truth, **kwargs)
        rewards.append(reward)
    return rewards


if __name__ == "__main__":
    # Test cases
    print("Testing KARE Mortality Probability Reward Function (Option A)")
    print("="*70)
    
    test_cases = [
        # (solution_str, ground_truth, expected_reward)
        (
            "<answer>\nMORTALITY PROBABILITY: 0.85\nSURVIVAL PROBABILITY: 0.15\n</answer>",
            {'target': ['1']},
            0.85,
            "High mortality for mortality case"
        ),
        (
            "<answer>\nMORTALITY PROBABILITY: 0.20\nSURVIVAL PROBABILITY: 0.80\n</answer>",
            {'target': ['0']},
            0.80,
            "Low mortality for survival case"
        ),
        (
            "<answer>\nMORTALITY PROBABILITY: 0.30\nSURVIVAL PROBABILITY: 0.70\n</answer>",
            {'target': ['1']},
            0.30,
            "Low mortality for mortality case (poor prediction)"
        ),
        (
            "<answer>\nMORTALITY PROBABILITY: 0.75\nSURVIVAL PROBABILITY: 0.25\n</answer>",
            {'target': ['0']},
            0.25,
            "High mortality for survival case (poor prediction)"
        ),
        (
            "<answer>\nMORTALITY PROBABILITY: 0.55\nSURVIVAL PROBABILITY: 0.45\n</answer>",
            {'target': ['1']},
            0.55,
            "Borderline mortality for mortality case"
        ),
        (
            "<answer>\nNo probability found\n</answer>",
            {'target': ['1']},
            0.0,
            "Invalid format (no probability)"
        ),
        (
            "<answer>\nMORTALITY PROBABILITY: 1.5\n</answer>",
            {'target': ['1']},
            0.0,
            "Invalid range (> 1.0)"
        ),
        (
            "<answer>\nMORTALITY PROBABILITY: 0.60\nSURVIVAL PROBABILITY: 0.50\n</answer>",
            {'target': ['0']},
            0.0,
            "Probabilities don't sum to 1.0"
        ),
    ]
    
    for i, (solution, gt, expected, description) in enumerate(test_cases, 1):
        reward = compute_score(solution, gt)
        status = "✓" if abs(reward - expected) < 0.001 else "✗"
        print(f"\nTest {i}: {description}")
        print(f"  Ground truth: {gt['target'][0]} ({'mortality' if gt['target'][0] == '1' else 'survival'})")
        mort, surv = extract_probabilities(solution)
        print(f"  Extracted: mort={mort}, surv={surv}")
        print(f"  Expected reward: {expected:.3f}")
        print(f"  Actual reward: {reward:.3f}")
        print(f"  Status: {status}")
    
    print("\n" + "="*70)
    print("All tests complete!")
