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

The reward is outcome-based with symmetric penalties:
- Ground Truth = 0 (Survival):
  - Mortality < 0.4 → +1.0 (correctly low)
  - Mortality [0.4, 0.7) → 0.0 (uncertain)
  - Mortality ≥ 0.7 → -1.0 (incorrectly high)
  - Survival ≥ 0.6 → +1.0 (correctly high)
  - Survival [0.3, 0.6) → 0.0 (uncertain)
  - Survival < 0.3 → -1.0 (incorrectly low)
  
- Ground Truth = 1 (Mortality):
  - Mortality ≥ 0.7 → +1.0 (correctly high)
  - Mortality [0.4, 0.7) → 0.0 (uncertain)
  - Mortality < 0.4 → -1.0 (incorrectly low)
  - Survival < 0.3 → +1.0 (correctly low)
  - Survival [0.3, 0.6) → 0.0 (uncertain)
  - Survival ≥ 0.6 → -1.0 (incorrectly high)
"""

import re
import json
from typing import Optional


# Patterns for extracting probabilities
MORTALITY_PATTERN = re.compile(
    r'MORTALITY\s+PROBABILITY\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    re.IGNORECASE
)

SURVIVAL_PATTERN = re.compile(
    r'SURVIVAL\s+PROBABILITY\s*:\s*([0-9]+(?:\.[0-9]+)?)',
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
    
    # Extract both probabilities
    mort_prob = extract_mortality_probability(solution_str)
    surv_prob = extract_survival_probability(solution_str)
    
    # Select probability based on assessment type
    if assessment_type == 'mortality':
        prob = mort_prob
    else:  # 'survival'
        prob = surv_prob
    
    # If probability not found, return 0.0
    if prob is None:
        return 0.0
    
    # Ensure probability is in valid range [0.0, 1.0]
    if prob < 0.0 or prob > 1.0:
        return 0.0
    
    # Ground truth must be provided
    if ground_truth is None:
        return 0.0
    
    # Symmetric reward based on ground truth and assessment type
    if ground_truth == 0:  # Survival (patient survived)
        if assessment_type == 'mortality':
            # Lower mortality probability is better for survival cases
            if prob < 0.4:
                return 1.0   # Correctly low mortality prediction
            elif prob < 0.7:
                return 0.0   # Uncertain
            else:
                return -1.0  # Incorrectly high mortality prediction
        else:  # survival assessment
            # Higher survival probability is better for survival cases
            if prob >= 0.6:
                return 1.0   # Correctly high survival prediction
            elif prob >= 0.3:
                return 0.0   # Uncertain
            else:
                return -1.0  # Incorrectly low survival prediction
    
    else:  # ground_truth == 1: Mortality (patient died)
        if assessment_type == 'mortality':
            # Higher mortality probability is better for mortality cases
            if prob >= 0.7:
                return 1.0   # Correctly high mortality prediction
            elif prob >= 0.4:
                return 0.0   # Uncertain
            else:
                return -1.0  # Incorrectly low mortality prediction
        else:  # survival assessment
            # Lower survival probability is better for mortality cases
            if prob < 0.3:
                return 1.0   # Correctly low survival prediction
            elif prob < 0.6:
                return 0.0   # Uncertain
            else:
                return -1.0  # Incorrectly high survival prediction


def test_reward_function():
    """Test cases for the prediction reward function."""
    
    print("Testing KARE Prediction Reward Function")
    print("=" * 60)
    
    # Test cases: (solution_str, ground_truth, assessment_type, expected_reward, description)
    test_cases = [
        # Ground Truth = 0 (Survival), Mortality Assessment
        ("Analysis...\nMORTALITY PROBABILITY: 0.25", 0, "mortality", 1.0, "Survival case, low mortality → +1"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.55", 0, "mortality", 0.0, "Survival case, medium mortality → 0"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.85", 0, "mortality", -1.0, "Survival case, high mortality → -1"),
        
        # Ground Truth = 0 (Survival), Survival Assessment
        ("Analysis...\nSURVIVAL PROBABILITY: 0.75", 0, "survival", 1.0, "Survival case, high survival → +1"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.45", 0, "survival", 0.0, "Survival case, medium survival → 0"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.15", 0, "survival", -1.0, "Survival case, low survival → -1"),
        
        # Ground Truth = 1 (Mortality), Mortality Assessment
        ("Analysis...\nMORTALITY PROBABILITY: 0.85", 1, "mortality", 1.0, "Mortality case, high mortality → +1"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.55", 1, "mortality", 0.0, "Mortality case, medium mortality → 0"),
        ("Analysis...\nMORTALITY PROBABILITY: 0.25", 1, "mortality", -1.0, "Mortality case, low mortality → -1"),
        
        # Ground Truth = 1 (Mortality), Survival Assessment
        ("Analysis...\nSURVIVAL PROBABILITY: 0.15", 1, "survival", 1.0, "Mortality case, low survival → +1"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.45", 1, "survival", 0.0, "Mortality case, medium survival → 0"),
        ("Analysis...\nSURVIVAL PROBABILITY: 0.75", 1, "survival", -1.0, "Mortality case, high survival → -1"),
        
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
        
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status}: {description}")
        print(f"  Expected: {expected}, Got: {result}")
        if result != expected:
            print(f"  Ground truth: {gt}, Assessment: {assessment}")
            print(f"  Solution: {solution[:80]}...")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = test_reward_function()
    exit(0 if success else 1)
