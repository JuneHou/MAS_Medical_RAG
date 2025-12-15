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
Binary reward function for KARE mortality prediction format enforcement.

This module provides reward scoring for training models to output mortality 
probabilities in the correct format: "MORTALITY PROBABILITY: X.XX"
where X.XX is a valid float between 0.00 and 1.00.

The reward is purely format-based (not accuracy-based):
- 1.0: Valid format with probability in [0.0, 1.0] on the LAST LINE
- 0.0: Missing format or invalid probability value

IMPORTANT: Only the LAST non-empty line is checked for the format.
This ensures the model learns to end with the required format.
"""

import re
from typing import Optional


# Strict pattern for format validation on last line
# Matches: "MORTALITY PROBABILITY: X.XX" (case-insensitive)
# Captures the numeric value which can be integer or decimal
FORMAT_PATTERN = re.compile(
    r'^\s*MORTALITY\s+PROBABILITY\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$',
    re.IGNORECASE
)


def compute_score(data_source: str = None, 
                  solution_str: str = "", 
                  ground_truth: str = None, 
                  extra_info: dict = None,
                  **kwargs) -> float:
    """
    Binary reward function for KARE mortality format compliance.
    
    Checks if the LAST non-empty line matches "MORTALITY PROBABILITY: X.XX" 
    with a valid probability value in [0.0, 1.0].
    
    This is format-only training:
    - ground_truth is ignored (should be "__FORMAT_ONLY__" placeholder)
    - Only format compliance is rewarded, not accuracy
    
    Args:
        data_source: Data source identifier (ignored)
        solution_str: Model output text to evaluate
        ground_truth: Placeholder (ignored, should be "__FORMAT_ONLY__")
        extra_info: Additional information (ignored)
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        Binary reward: 1.0 if last line has valid format, 0.0 otherwise
        
    Examples:
        >>> compute_score(None, "Some reasoning\\nMORTALITY PROBABILITY: 0.75")
        1.0
        >>> compute_score(None, "Analysis here\\nMORTALITY PROBABILITY: 0.05\\n")
        1.0
        >>> compute_score(None, "The patient has high risk")
        0.0
        >>> compute_score(None, "MORTALITY PROBABILITY: 1.5")  # Invalid value
        0.0
        >>> compute_score(None, "MORTALITY PROBABILITY: 0.65\\nExtra text")  # Not last line
        0.0
    """
    # DEBUG: Log first few calls to understand what's happening
    import os
    debug_file = "/data/wang/junh/githubs/Debate/KARE/verl/reward_debug.log"
    if not os.path.exists(debug_file) or os.path.getsize(debug_file) < 10000:
        with open(debug_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"solution_str (first 500 chars):\n{solution_str[:500]}\n")
            f.write(f"solution_str (last 500 chars):\n{solution_str[-500:]}\n")
            f.write(f"data_source: {data_source}\n")
            f.write(f"ground_truth: {ground_truth}\n")
    
    # Check if solution_str is empty
    if not solution_str:
        return 0.0
    
    # Get all non-empty lines
    lines = [ln.strip() for ln in str(solution_str).splitlines() if ln.strip()]
    if not lines:
        return 0.0
    
    # Check ONLY the last line for format
    last_line = lines[-1]
    match = FORMAT_PATTERN.match(last_line)
    
    if not match:
        return 0.0
    
    # Extract and validate probability value
    try:
        prob = float(match.group(1))
    except (ValueError, AttributeError):
        return 0.0
    
    # Reward only if probability is in valid range [0.0, 1.0]
    return 1.0 if (0.0 <= prob <= 1.0) else 0.0


# Additional helper for batch scoring (if needed)
def compute_score_batch(solution_strs, ground_truths=None, format_score: float = 0.0, score: float = 1.0):
    """
    Batch version of compute_score for efficiency.
    
    Args:
        solution_strs: List of model output texts
        ground_truths: Not used (included for API compatibility)
        format_score: Score for invalid format
        score: Score for valid format
        
    Returns:
        List of binary rewards
    """
    return [compute_score(solution_str=s, ground_truth=None, format_score=format_score, score=score) for s in solution_strs]
