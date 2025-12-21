# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prediction accuracy reward function for KARE mortality prediction.

This module provides reward scoring for training models to output accurate
mortality and survival probabilities based on ground truth patient outcomes.

The reward is based on the Brier score for continuous, stable RL training:
- Brier score: r = 1 - (p - y)²
- Rescaled to [-1, +1]: r' = 1 - 2(p - y)²

Where:
- p = predicted probability (mortality or survival based on assessment_type)
- y = ground truth label (0=survival, 1=mortality)

Benefits:
- Continuous signal (non-zero gradients everywhere) 
- Stable for RL (clamped to [-1, +1] range)
- More aggressive penalty for large errors
- Maximum reward (+1) when p = y (perfect prediction)
- Neutral reward (0) for moderate errors (Brier error = 0.25)
- Strong penalty for bad predictions (70% wrong → -0.96 reward)
- Clamped minimum reward (-1) for stability
"""

import re
import json
from typing import Optional


# Patterns for extracting probabilities - MUST MATCH downstream parsing exactly
MORTALITY_PATTERN = re.compile(
    r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)

SURVIVAL_PATTERN = re.compile(
    r'SURVIVAL PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)


def extract_mortality_probability(text: str) -> Optional[float]:
    """
    Extract mortality probability from model output.
    
    Args:
        text: Model-generated text
        
    Returns:
        Extracted probability as float, or None if not found
    """
    match = MORTALITY_PATTERN.search(text)
    if match:
        try:
            prob = float(match.group(1))
            return prob
        except ValueError:
            return None
    return None


def extract_survival_probability(text: str) -> Optional[float]:
    """
    Extract survival probability from model output.
    
    Args:
        text: Model-generated text
        
    Returns:
        Extracted probability as float, or None if not found
    """
    match = SURVIVAL_PATTERN.search(text)
    if match:
        try:
            prob = float(match.group(1))
            return prob
        except ValueError:
            return None
    return None


def compute_score(data_source: str = None,
                  solution_str: str = "",
                  ground_truth: int = None,
                  extra_info: dict = None,
                  **kwargs) -> float:
    """
    Prediction accuracy reward function for KARE mortality prediction.
    
    Args:
        data_source: Data source identifier (e.g., 'kare_mortality_prediction')
        solution_str: Model-generated response text
        ground_truth: Ground truth label (0=survival, 1=mortality)
        extra_info: Dictionary containing 'assessment_type' ('mortality' or 'survival')
        **kwargs: Additional keyword arguments
        
    Returns:
        Reward score: +1.0 (correct), 0.0 (uncertain), or -1.0 (incorrect)
    """
    # Parse extra_info if it's a JSON string
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, TypeError):
            extra_info = {}
    
    if extra_info is None:
        extra_info = {}
    
    # Determine assessment type
    assessment_type = extra_info.get('assessment_type', 'mortality')
    
    # DEBUG: Log first few calls to see what the model is generating
    import os
    debug_file = "/data/wang/junh/githubs/Debate/KARE/verl/logs/reward_debug.log"
    os.makedirs(os.path.dirname(debug_file), exist_ok=True)
    
    # Only log first 50 calls to avoid huge files
    if os.path.exists(debug_file):
        with open(debug_file, 'r') as f:
            num_logged = len(f.readlines())
    else:
        num_logged = 0
    
    if num_logged < 50:
        with open(debug_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Call #{num_logged + 1}\n")
            f.write(f"Assessment Type: {assessment_type}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Response Length: {len(solution_str)} chars\n")
            f.write(f"Response:\n{solution_str}\n")
            f.write(f"{'='*80}\n")
    
    # Extract both probabilities
    mort_prob = extract_mortality_probability(solution_str)
    surv_prob = extract_survival_probability(solution_str)
    
    # Select probability based on assessment type
    if assessment_type == 'mortality':
        prob = mort_prob
    else:  # 'survival'
        prob = surv_prob
    
    # DEBUG: Log extraction results
    if num_logged < 50:
        with open(debug_file, 'a') as f:
            f.write(f"Extracted Mortality Prob: {mort_prob}\n")
            f.write(f"Extracted Survival Prob: {surv_prob}\n")
            f.write(f"Selected Prob ({assessment_type}): {prob}\n")
    
    # If probability not found, return 0.0
    if prob is None:
        if num_logged < 50:
            with open(debug_file, 'a') as f:
                f.write(f"REWARD: 0.0 (probability not found)\n")
        return 0.0
    
    # Ensure probability is in valid range [0.0, 1.0]
    if prob < 0.0 or prob > 1.0:
        if num_logged < 50:
            with open(debug_file, 'a') as f:
                f.write(f"REWARD: 0.0 (probability out of range: {prob})\n")
        return 0.0
    
    # Ground truth must be provided
    if ground_truth is None:
        if num_logged < 50:
            with open(debug_file, 'a') as f:
                f.write(f"REWARD: 0.0 (ground truth is None)\n")
        return 0.0
    
    # Convert ground truth and probability to the same scale for Brier score
    # For mortality assessment: y = ground_truth, p = mortality probability
    # For survival assessment: y = 1 - ground_truth, p = survival probability
    
    if assessment_type == 'mortality':
        # Mortality assessment: higher prob should match mortality outcome (ground_truth = 1)
        y = float(ground_truth)  # 0 for survival, 1 for mortality
        p = prob  # mortality probability
    else:  # survival assessment
        # Survival assessment: higher prob should match survival outcome (ground_truth = 0)
        y = 1.0 - float(ground_truth)  # 1 for survival, 0 for mortality  
        p = prob  # survival probability
    
    # Modified Brier score with more aggressive penalty: r' = 1 - 4(p - y)²
    # This gives better separation:
    # - Perfect prediction (p = y): reward = +1.0
    # - Good prediction (error = 0.01): reward = +0.96
    # - Okay prediction (error = 0.25): reward = 0.0 (neutral)  
    # - Bad prediction (error = 0.49): reward = -0.96
    # - Worst prediction (error = 1.0): reward = -3.0
    # Then clamp to [-1, +1] range for stability
    brier_error = (p - y) ** 2
    raw_reward = 1.0 - 4.0 * brier_error
    reward = max(-1.0, min(1.0, raw_reward))  # Clamp to [-1, +1]
    
    # Create descriptive reason for logging
    if assessment_type == 'mortality':
        if ground_truth == 1:
            reason = f"Mortality case: p={prob:.3f} vs y={y:.0f}, Brier error={brier_error:.3f}"
        else:
            reason = f"Survival case: mort_p={prob:.3f} vs y={y:.0f}, Brier error={brier_error:.3f}"
    else:
        if ground_truth == 0:
            reason = f"Survival case: p={prob:.3f} vs y={y:.0f}, Brier error={brier_error:.3f}"
        else:
            reason = f"Mortality case: surv_p={prob:.3f} vs y={y:.0f}, Brier error={brier_error:.3f}"
    
    # DEBUG: Log final reward
    if num_logged < 50:
        with open(debug_file, 'a') as f:
            f.write(f"REWARD: {reward} ({reason})\n")
    
    return reward


def test_reward_function():
    """Test cases for the prediction reward function."""
    
    print("Testing KARE Prediction Reward Function")
    print("=" * 60)
    
    # Test cases: (solution_str, ground_truth, assessment_type, expected_reward, description)
    # For Brier score: r' = 1 - 2(p - y)²
    test_cases = [
        # Perfect predictions (p = y): reward = +1.0
        ("Analysis...\nMORTALITY PROBABILITY: 0.0", 0, "mortality", 1.0, "Perfect: survival case, 0% mortality"),
        ("Analysis...\nMORTALITY PROBABILITY: 1.0", 1, "mortality", 1.0, "Perfect: mortality case, 100% mortality"),
        ("Analysis...\nSURVIVAL PROBABILITY: 1.0", 0, "survival", 1.0, "Perfect: survival case, 100% survival"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.0", 1, "survival", 1.0, "Perfect: mortality case, 0% survival"),
        
        # Good predictions (close to target) - with aggressive penalty: 1 - 4(0.1)² = 0.96
        ("Analysis...\nMORTALITY PROBABILITY: 0.1", 0, "mortality", 0.96, "Good: survival case, 10% mortality → 0.96"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.9", 1, "mortality", 0.96, "Good: mortality case, 90% mortality → 0.96"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.9", 0, "survival", 0.96, "Good: survival case, 90% survival → 0.96"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.1", 1, "survival", 0.96, "Good: mortality case, 10% survival → 0.96"),
        
        # Random/uncertain predictions (p = 0.5): reward = 0.0 (neutral)
        # Brier error = (0.5 - 0)² = 0.25 or (0.5 - 1)² = 0.25  
        # Reward = 1 - 4(0.25) = 0.0
        ("Analysis...\nMORTALITY PROBABILITY: 0.5", 0, "mortality", 0.0, "Random: survival case, 50% mortality → 0.0"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.5", 1, "mortality", 0.0, "Random: mortality case, 50% mortality → 0.0"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.5", 0, "survival", 0.0, "Random: survival case, 50% survival → 0.0"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.5", 1, "survival", 0.0, "Random: mortality case, 50% survival → 0.0"),
        
        # Bad predictions (far from target) - now much more heavily penalized
        ("Analysis...\nMORTALITY PROBABILITY: 0.8", 0, "mortality", -1.0, "Bad: survival case, 80% mortality → -1.0 (clamped)"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.2", 1, "mortality", -1.0, "Bad: mortality case, 20% mortality → -1.0 (clamped)"),
        
        # Maximally wrong predictions (|p - y| = 1): reward = -1.0
        ("Analysis...\nMORTALITY PROBABILITY: 1.0", 0, "mortality", -1.0, "Worst: survival case, 100% mortality → -1.0"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.0", 1, "mortality", -1.0, "Worst: mortality case, 0% mortality → -1.0"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.0", 0, "survival", -1.0, "Worst: survival case, 0% survival → -1.0"),
        ("Analysis...\nSURVIVAL PROBABILITY: 1.0", 1, "survival", -1.0, "Worst: mortality case, 100% survival → -1.0"),
        
        # Edge cases
        ("Analysis without probability", 0, "mortality", 0.0, "No probability found → 0"),
        ("MORTALITY PROBABILITY: 1.5", 0, "mortality", 0.0, "Invalid probability (>1.0) → 0"),
        ("MORTALITY PROBABILITY: -0.2", 0, "mortality", 0.0, "Invalid probability (<0.0) → 0"),
    ]
    
    passed = 0
    failed = 0
    
    for solution, gt, assessment, expected, description in test_cases:
        extra_info = {'assessment_type': assessment}
        result = compute_score(
            solution_str=solution,
            ground_truth=gt,
            extra_info=extra_info
        )
        
        # Use approximate equality for floating-point comparisons
        is_close = abs(result - expected) < 1e-6
        status = "✓ PASS" if is_close else "✗ FAIL"
        if is_close:
            passed += 1
        else:
            failed += 1
        
        print(f"{status}: {description}")
        print(f"  Expected: {expected}, Got: {result:.6f}")
        if not is_close:
            print(f"  Ground truth: {gt}, Assessment: {assessment}")
            print(f"  Solution: {solution[:80]}...")
            print(f"  Difference: {abs(result - expected):.6f}")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = test_reward_function()
    exit(0 if success else 1)
