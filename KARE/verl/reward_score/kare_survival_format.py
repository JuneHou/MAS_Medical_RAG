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
Binary reward function for KARE survival prediction format enforcement.

This module provides reward scoring for training models to output survival 
probabilities in the correct format: "SURVIVAL PROBABILITY: X.XX"
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
# Matches: "SURVIVAL PROBABILITY: X.XX" (case-insensitive)
# Captures the numeric value which can be integer or decimal
FORMAT_PATTERN = re.compile(
    r'^\s*SURVIVAL\s+PROBABILITY\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$',
    re.IGNORECASE
)


def compute_score(data_source: str = None, 
                  solution_str: str = "", 
                  ground_truth: str = None, 
                  extra_info: dict = None,
                  **kwargs) -> float:
    """
    Binary reward function for KARE survival format compliance.
    
    Checks if the LAST non-empty line matches "SURVIVAL PROBABILITY: X.XX" 
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
        >>> compute_score(None, "Some reasoning\\nSURVIVAL PROBABILITY: 0.75")
        1.0
        >>> compute_score(None, "Analysis here\\nSURVIVAL PROBABILITY: 0.95\\n")
        1.0
        >>> compute_score(None, "The patient has good prognosis")
        0.0
        >>> compute_score(None, "SURVIVAL PROBABILITY: 1.5")  # Invalid value
        0.0
        >>> compute_score(None, "SURVIVAL PROBABILITY: 0.85\\nExtra text")  # Not last line
        0.0
    """
    # DEBUG: Log first few calls to understand what's happening
    debug_log_path = "/data/wang/junh/githubs/Debate/KARE/verl/reward_debug_survival.log"
    try:
        import os
        if not os.path.exists(debug_log_path) or os.path.getsize(debug_log_path) < 10240:  # First 10KB
            with open(debug_log_path, "a") as f:
                f.write("="*80 + "\n")
                f.write(f"solution_str (first 500 chars):\n{str(solution_str)[:500]}\n")
                f.write(f"solution_str (last 500 chars):\n{str(solution_str)[-500:]}\n")
                f.write(f"data_source: {data_source}\n")
                f.write(f"ground_truth: {ground_truth}\n\n")
    except Exception:
        pass  # Ignore debug logging errors
    
    if not solution_str:
        return 0.0
    
    # Split into lines and get non-empty lines
    lines = [ln.strip() for ln in str(solution_str).splitlines() if ln.strip()]
    
    if not lines:
        return 0.0
    
    # Check ONLY the last non-empty line
    last_line = lines[-1]
    
    match = FORMAT_PATTERN.match(last_line)
    
    if not match:
        return 0.0
    
    # Extract and validate probability
    try:
        prob = float(match.group(1))
        
        # Valid if in [0.0, 1.0]
        if 0.0 <= prob <= 1.0:
            return 1.0
        else:
            return 0.0
            
    except (ValueError, IndexError):
        return 0.0


# Test function for development
def test_reward_function():
    """Test the reward function with various inputs."""
    test_cases = [
        ("SURVIVAL PROBABILITY: 0.75", 1.0),
        ("Some analysis\nSURVIVAL PROBABILITY: 0.85", 1.0),
        ("SURVIVAL PROBABILITY: 0.00", 1.0),
        ("SURVIVAL PROBABILITY: 1.00", 1.0),
        ("No format here", 0.0),
        ("SURVIVAL PROBABILITY: 1.5", 0.0),  # Out of range
        ("SURVIVAL PROBABILITY: -0.1", 0.0),  # Negative
        ("SURVIVAL PROBABILITY: 0.65\nExtra text", 0.0),  # Not last line
        ("  SURVIVAL PROBABILITY: 0.42  ", 1.0),  # With whitespace
        ("survival probability: 0.99", 1.0),  # Case insensitive
    ]
    
    print("Testing SURVIVAL format reward function:")
    for text, expected in test_cases:
        result = compute_score(solution_str=text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {text[:50]:<50} Expected: {expected} Got: {result}")


if __name__ == "__main__":
    test_reward_function()
