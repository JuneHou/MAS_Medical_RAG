#!/usr/bin/env python3
"""
Verification script to ensure exact parity between vllm and effGen implementations.
Compares prompts, hyperparameters, and logic between the two versions.
"""

import os
import sys
import re
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
effgen_path = Path(__file__).parent
parent_path = effgen_path.parent

def extract_string_literal(content, var_name):
    """Extract a multi-line string literal from Python code."""
    # Look for var_name = """...""" or var_name = "..."
    pattern = rf'{var_name}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern = rf'{var_name}\s*=\s*"(.*?)"'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None

def extract_hyperparameter(content, param_name):
    """Extract a hyperparameter value from generation call."""
    # Look for param_name=value in generate or SamplingParams calls
    patterns = [
        rf'{param_name}=([0-9.]+)',
        rf'{param_name}\s*=\s*([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            return float(value) if '.' in value else int(value)
    
    return None

def compare_strings(name, original, effgen):
    """Compare two strings and report differences."""
    if original is None and effgen is None:
        print(f"  ⚠ {name}: Both not found")
        return False
    elif original is None:
        print(f"  ✗ {name}: Original not found")
        return False
    elif effgen is None:
        print(f"  ✗ {name}: effGen not found")
        return False
    elif original == effgen:
        print(f"  ✓ {name}: Exact match ({len(original)} chars)")
        return True
    else:
        print(f"  ✗ {name}: MISMATCH")
        print(f"    Original: {len(original)} chars")
        print(f"    effGen:   {len(effgen)} chars")
        
        # Show first difference
        for i, (c1, c2) in enumerate(zip(original, effgen)):
            if c1 != c2:
                print(f"    First diff at position {i}:")
                print(f"      Original: ...{original[max(0,i-20):i+20]}...")
                print(f"      effGen:   ...{effgen[max(0,i-20):i+20]}...")
                break
        
        return False

def compare_values(name, original, effgen):
    """Compare two values and report differences."""
    if original is None and effgen is None:
        print(f"  ⚠ {name}: Both not found")
        return False
    elif original is None:
        print(f"  ✗ {name}: Original not found")
        return False
    elif effgen is None:
        print(f"  ✗ {name}: effGen not found")
        return False
    elif original == effgen:
        print(f"  ✓ {name}: {original}")
        return True
    else:
        print(f"  ✗ {name}: MISMATCH - Original: {original}, effGen: {effgen}")
        return False

def main():
    print("=" * 80)
    print("EXACT PARITY VERIFICATION")
    print("Comparing vllm vs effGen implementations")
    print("=" * 80)
    
    # Read original vllm implementation
    original_path = parent_path / "mortality_single_agent_rag.py"
    if not original_path.exists():
        print(f"\n✗ ERROR: Original file not found: {original_path}")
        return False
    
    with open(original_path, 'r') as f:
        original_content = f.read()
    
    print(f"\n✓ Original vllm: {original_path}")
    
    # Read effGen implementation
    effgen_path_file = effgen_path / "mortality_single_agent_effgen_rag.py"
    if not effgen_path_file.exists():
        print(f"\n✗ ERROR: effGen file not found: {effgen_path_file}")
        return False
    
    with open(effgen_path_file, 'r') as f:
        effgen_content = f.read()
    
    print(f"✓ effGen impl:    {effgen_path_file}")
    
    # Compare task_description
    print("\n" + "=" * 80)
    print("1. TASK DESCRIPTION")
    print("=" * 80)
    
    original_task = extract_string_literal(original_content, 'self.task_description')
    effgen_task = extract_string_literal(effgen_content, 'self.task_description')
    task_match = compare_strings("task_description", original_task, effgen_task)
    
    # Compare retrieval_instruction
    print("\n" + "=" * 80)
    print("2. RETRIEVAL INSTRUCTION")
    print("=" * 80)
    
    original_retrieval_inst = extract_string_literal(original_content, 'self.retrieval_instruction')
    effgen_retrieval_inst = extract_string_literal(effgen_content, 'self.retrieval_instruction')
    retrieval_inst_match = compare_strings("retrieval_instruction", original_retrieval_inst, effgen_retrieval_inst)
    
    # Compare hyperparameters
    print("\n" + "=" * 80)
    print("3. HYPERPARAMETERS")
    print("=" * 80)
    
    params_to_check = [
        ('temperature', 0.7),
        ('top_p', 0.9),
        ('max_tokens', 32768),
        ('repetition_penalty', 1.2),
    ]
    
    params_match = True
    for param_name, expected_value in params_to_check:
        # For effGen, check if the value is hardcoded in generate calls
        # Look for the parameter in both SamplingParams (original) and generate() (effGen)
        original_value = extract_hyperparameter(original_content, param_name)
        if param_name == 'max_tokens':
            # effGen uses max_new_tokens instead
            effgen_value = extract_hyperparameter(effgen_content, 'max_new_tokens')
        else:
            effgen_value = extract_hyperparameter(effgen_content, param_name)
        
        if compare_values(param_name, expected_value, effgen_value):
            pass  # Match
        else:
            params_match = False
    
    # Check retrieval logic patterns
    print("\n" + "=" * 80)
    print("4. RETRIEVAL LOGIC")
    print("=" * 80)
    
    retrieval_checks = [
        ("MedCorp2 splitting logic", r"k_medcorp\s*=\s*k\s*//\s*2\s*\+\s*k\s*%\s*2"),
        ("UMLS splitting logic", r"k_umls\s*=\s*k\s*//\s*2"),
        ("Query truncation", r"query\[:200\]"),
        ("RRF parameter", r"rrf_k\s*=\s*60"),
        ("Document score formatting", r"Score:\s*\{[^}]*:.3f\}"),
    ]
    
    retrieval_match = True
    for check_name, pattern in retrieval_checks:
        original_has = bool(re.search(pattern, original_content))
        effgen_has = bool(re.search(pattern, effgen_content))
        
        if original_has and effgen_has:
            print(f"  ✓ {check_name}: Found in both")
        elif not original_has and not effgen_has:
            print(f"  ⚠ {check_name}: Not found in either")
        else:
            print(f"  ✗ {check_name}: Found in {'original' if original_has else 'effGen'} only")
            retrieval_match = False
    
    # Check prediction parsing patterns
    print("\n" + "=" * 80)
    print("5. PREDICTION PARSING")
    print("=" * 80)
    
    prediction_patterns = [
        (r'#\s*Prediction\s*#\[\s:\]\*\(\[01\]\)', "# Prediction # pattern"),
        (r'Prediction\[:\s\]\+\(\[01\]\)', "Prediction: pattern"),
        (r'\\\*\\\*Prediction\\\*\\\*\[:\s\]\+\(\[01\]\)', "**Prediction** pattern"),
        (r'prediction\[:\s\]\+\(\[01\]\)', "prediction: pattern"),
    ]
    
    parsing_match = True
    for pattern, description in prediction_patterns:
        original_has = bool(re.search(pattern, original_content))
        effgen_has = bool(re.search(pattern, effgen_content))
        
        if original_has and effgen_has:
            print(f"  ✓ {description}: Found in both")
        else:
            print(f"  ✗ {description}: Mismatch")
            parsing_match = False
    
    # Check fallback logic
    print("\n" + "=" * 80)
    print("6. FALLBACK LOGIC")
    print("=" * 80)
    
    fallback_checks = [
        ("Opposite of ground truth", "1 - ground_truth"),
        ("Default to survival", "prediction.*=.*0"),
        ("is_fallback flag", "is_fallback.*=.*True"),
    ]
    
    fallback_match = True
    for check_name, pattern in fallback_checks:
        original_has = bool(re.search(pattern, original_content))
        effgen_has = bool(re.search(pattern, effgen_content))
        
        if original_has and effgen_has:
            print(f"  ✓ {check_name}: Found in both")
        else:
            print(f"  ✗ {check_name}: Mismatch")
            fallback_match = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    checks = [
        ("Task Description", task_match),
        ("Retrieval Instruction", retrieval_inst_match),
        ("Hyperparameters", params_match),
        ("Retrieval Logic", retrieval_match),
        ("Prediction Parsing", parsing_match),
        ("Fallback Logic", fallback_match),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ EXACT PARITY VERIFIED")
        print("\nThe effGen implementation matches the original vllm version!")
    else:
        print("✗ PARITY CHECK FAILED")
        print("\nSome components do not match. Please review the differences above.")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
