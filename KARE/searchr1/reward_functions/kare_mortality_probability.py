#!/usr/bin/env python3
"""
Custom reward function for Search-R1 probability-based mortality prediction.

This module implements Option A (Positive-Only Reward):
- Reward = mortality_prob if GT=1, else (1 - mortality_prob)
- Range: [0.0, 1.0]
- Smooth optimization for better GRPO convergence

Expected output format:
<answer>
MORTALITY PROBABILITY: 0.XX
SURVIVAL PROBABILITY: 0.YY
</answer>

Where SURVIVAL = 1.0 - MORTALITY (enforced with 5% tolerance).
"""

import re
from typing import Optional, Tuple


# Regex patterns to extract probabilities from <answer> tag
MORT_PROB_RE = re.compile(
    r'MORTALITY\s+PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)
SURV_PROB_RE = re.compile(
    r'SURVIVAL\s+PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)


def extract_probabilities(solution_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract mortality and survival probabilities from model output.
    
    Args:
        solution_str: Full model response text with <answer> tag
        
    Returns:
        (mortality_prob, survival_prob) tuple
        - Returns (None, None) if probabilities not found
        - Returns (mort, None) if only mortality found
    """
    if not solution_str:
        return None, None
    
    # Extract mortality probability
    mort_match = MORT_PROB_RE.search(solution_str)
    mort_prob = None
    if mort_match:
        try:
            mort_prob = float(mort_match.group(1))
        except (ValueError, AttributeError):
            pass
    
    # Extract survival probability
    surv_match = SURV_PROB_RE.search(solution_str)
    surv_prob = None
    if surv_match:
        try:
            surv_prob = float(surv_match.group(1))
        except (ValueError, AttributeError):
            pass
    
    return mort_prob, surv_prob


def compute_score(solution_str: str, ground_truth: dict, **kwargs) -> float:
    """
    Compute positive-only reward for probability-based mortality prediction.
    
    This is Option A: smooth, positive rewards that incentivize calibration
    without negative penalties (better for Search-R1's GRPO optimization).
    
    Reward formula:
    - Mortality case (GT=1): reward = mortality_prob
    - Survival case (GT=0): reward = 1.0 - mortality_prob
    
    Validation (ALL REQUIRED - penalty of 0.0 reward if violated):
    - Returns 0.0 if mortality probability not found
    - Returns 0.0 if survival probability not found (REQUIRED)
    - Returns 0.0 if either probability out of [0.0, 1.0] range
    - Returns 0.0 if probabilities don't sum to 1.0 (±5% tolerance)
    
    Args:
        solution_str: Full model response text
        ground_truth: Dictionary with {'target': ['0']} or {'target': ['1']}
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Reward in range [0.0, 1.0]
        
    Examples:
        >>> compute_score("MORTALITY PROBABILITY: 0.85\\nSURVIVAL: 0.15", {'target': ['1']})
        0.85  # High mortality for mortality case
        
        >>> compute_score("MORTALITY PROBABILITY: 0.20\\nSURVIVAL: 0.80", {'target': ['0']})
        0.80  # Low mortality for survival case
        
        >>> compute_score("MORTALITY PROBABILITY: 0.55\\nSURVIVAL: 0.45", {'target': ['0']})
        0.45  # Uncertain prediction (near 0.5)
    """
    # Extract probabilities
    mort_prob, surv_prob = extract_probabilities(solution_str)
    
    # Validation 1: Both probabilities must exist
    if mort_prob is None:
        return 0.0  # Missing mortality probability
    
    if surv_prob is None:
        return 0.0  # Missing survival probability - REQUIRED
    
    # Validation 2: Both probabilities must be in valid range
    if not (0.0 <= mort_prob <= 1.0):
        return 0.0  # Mortality prob out of range
    
    if not (0.0 <= surv_prob <= 1.0):
        return 0.0  # Survival prob out of range
    
    # Validation 3: Probabilities MUST sum to 1.0 (with 5% tolerance)
    prob_sum = mort_prob + surv_prob
    if abs(prob_sum - 1.0) > 0.05:
        return 0.0  # Probabilities don't sum to 1.0 - PENALTY
    
    # Extract ground truth label from Search-R1 format
    # ground_truth = {'target': ['0']} or {'target': ['1']}
    try:
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            gt_label = int(ground_truth['target'][0])
        else:
            # Fallback: try direct int conversion
            gt_label = int(ground_truth)
    except (ValueError, TypeError, IndexError, KeyError):
        return 0.0  # Invalid ground truth format
    
    # Compute positive-only reward
    if gt_label == 1:  # Mortality case
        # Reward high mortality predictions
        reward = mort_prob
    else:  # Survival case (gt_label == 0)
        # Reward low mortality predictions (high survival)
        reward = 1.0 - mort_prob
    
    return reward


def compute_score_batch(solution_strs, ground_truths, **kwargs):
    """
    Batch version of compute_score for efficiency.
    
    Args:
        solution_strs: List of model output texts
        ground_truths: List of ground truth dicts
        **kwargs: Additional arguments (ignored)
        
    Returns:
        List of rewards
    """
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths] * len(solution_strs)
    
    return [
        compute_score(sol, gt, **kwargs)
        for sol, gt in zip(solution_strs, ground_truths)
    ]


# Alternative: Symmetric ±1/0/-1 reward (Option B)
def compute_score_symmetric(solution_str: str, ground_truth: dict, **kwargs) -> float:
    """
    Option B: Symmetric reward with penalties for wrong predictions.
    
    Thresholds:
    - High confidence: mort < 0.4 or mort ≥ 0.7
    - Uncertain: mort ∈ [0.4, 0.7)
    
    Returns:
    - +1.0: Correct confident prediction
    - 0.0: Uncertain middle range
    - -1.0: Wrong confident prediction or invalid format
    
    Note: This is NOT the default. Use this if you want stronger
    learning signals with penalties for wrong directions.
    """
    mort_prob, surv_prob = extract_probabilities(solution_str)
    
    # Invalid format gets penalty
    if mort_prob is None or surv_prob is None:
        return -1.0  # Both probabilities REQUIRED
    
    if not (0.0 <= mort_prob <= 1.0) or not (0.0 <= surv_prob <= 1.0):
        return -1.0  # Out of range
    
    # Check probability sum constraint (REQUIRED)
    if abs((mort_prob + surv_prob) - 1.0) > 0.05:
        return -1.0  # Probabilities don't sum to 1.0
    
    # Extract ground truth
    try:
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            gt_label = int(ground_truth['target'][0])
        else:
            gt_label = int(ground_truth)
    except (ValueError, TypeError, IndexError, KeyError):
        return -1.0
    
    # Symmetric reward with thresholds
    if gt_label == 1:  # Mortality case
        if mort_prob >= 0.7:
            return +1.0  # Correctly high mortality
        elif mort_prob >= 0.4:
            return 0.0   # Uncertain
        else:
            return -1.0  # Incorrectly low mortality
    else:  # Survival case (gt_label == 0)
        if mort_prob < 0.4:
            return +1.0  # Correctly low mortality
        elif mort_prob < 0.7:
            return 0.0   # Uncertain
        else:
            return -1.0  # Incorrectly high mortality


if __name__ == "__main__":
    # Test the reward function
    print("Testing reward function...")
    
    # Test case 1: Perfect mortality prediction
    test1 = """<answer>
MORTALITY PROBABILITY: 0.85
SURVIVAL PROBABILITY: 0.15
</answer>"""
    reward1 = compute_score(test1, {'target': ['1']})
    print(f"Test 1 (mort=0.85, GT=1): {reward1:.3f} (expected ~0.85)")
    
    # Test case 2: Perfect survival prediction
    test2 = """<answer>
MORTALITY PROBABILITY: 0.20
SURVIVAL PROBABILITY: 0.80
</answer>"""
    reward2 = compute_score(test2, {'target': ['0']})
    print(f"Test 2 (mort=0.20, GT=0): {reward2:.3f} (expected ~0.80)")
    
    # Test case 3: Wrong direction (mort=0.75 for survival)
    test3 = """<answer>
MORTALITY PROBABILITY: 0.75
SURVIVAL PROBABILITY: 0.25
</answer>"""
    reward3 = compute_score(test3, {'target': ['0']})
    print(f"Test 3 (mort=0.75, GT=0): {reward3:.3f} (expected ~0.25, low)")
    
    # Test case 4: Invalid format
    test4 = "<answer>High risk patient</answer>"
    reward4 = compute_score(test4, {'target': ['1']})
    print(f"Test 4 (invalid): {reward4:.3f} (expected 0.0)")
    
    # Test case 5: Probabilities don't sum to 1.0
    test5 = """<answer>
MORTALITY PROBABILITY: 0.60
SURVIVAL PROBABILITY: 0.50
</answer>"""
    reward5 = compute_score(test5, {'target': ['1']})
    print(f"Test 5 (sum=1.1): {reward5:.3f} (expected 0.0, invalid)")
    
    # Test case 6: Missing survival probability
    test6 = """<answer>
MORTALITY PROBABILITY: 0.75
</answer>"""
    reward6 = compute_score(test6, {'target': ['1']})
    print(f"Test 6 (no survival): {reward6:.3f} (expected 0.0, penalty)")
    
    print("\n✓ Reward function tests complete!")
