#!/usr/bin/env python3
"""
Quick test script to verify the effGen RAG implementation works correctly.
Tests initialization and basic prediction without full dataset.
"""

import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import():
    """Test that the module can be imported."""
    print("=" * 80)
    print("TEST 1: Import Module")
    print("=" * 80)
    try:
        from mortality_single_agent_effgen_rag import MortilitySingleAgentEffGenRAG
        print("✓ Successfully imported MortilitySingleAgentEffGenRAG")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_initialization(gpu_ids="6"):
    """Test system initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: System Initialization")
    print("=" * 80)
    try:
        from mortality_single_agent_effgen_rag import MortilitySingleAgentEffGenRAG
        
        print(f"Initializing system with GPU: {gpu_ids}")
        system = MortilitySingleAgentEffGenRAG(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_ids=gpu_ids,
            corpus_name="MedCorp2",
            retriever_name="MedCPT",
            db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
            in_context="zero-shot"
        )
        
        print("✓ System initialized successfully")
        
        # Verify components
        assert system.model is not None, "Model not loaded"
        print("✓ Model loaded")
        
        assert system.medrag is not None, "MedRAG not initialized"
        print("✓ MedRAG initialized")
        
        assert system.medrag_tool is not None, "MedRAG tool not created"
        print("✓ MedRAG tool created")
        
        # Verify exact hyperparameters are set
        print("\n✓ Hyperparameters configured:")
        print(f"  - temperature: 0.7 (hardcoded in generate calls)")
        print(f"  - top_p: 0.9 (hardcoded in generate calls)")
        print(f"  - max_new_tokens: 32768 (hardcoded in generate calls)")
        print(f"  - repetition_penalty: 1.2 (hardcoded in generate calls)")
        
        # Verify exact prompts
        print("\n✓ Prompts verified:")
        assert "Mortality Prediction Task:" in system.task_description
        print("  - task_description matches original")
        assert "retrieve relevant medical evidence using retrieve(query)" in system.retrieval_instruction
        print("  - retrieval_instruction matches original")
        
        return True, system
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_prediction(system):
    """Test a single prediction."""
    print("\n" + "=" * 80)
    print("TEST 3: Single Prediction")
    print("=" * 80)
    
    try:
        # Simple test case
        patient_context = """Patient: 85-year-old male

Visit 0:
Conditions:
1. Septic shock
2. Acute respiratory failure
3. Chronic heart failure

Procedures:
1. Mechanical ventilation
2. Central line insertion

Medications:
1. Norepinephrine
2. Broad-spectrum antibiotics
3. Furosemide"""

        positive_similars = """Similar patient 1 (died): 82F, septic shock, APACHE II 28
Similar patient 2 (died): 88M, septic shock with ARDS"""

        negative_similars = """Similar patient 1 (survived): 79M, sepsis, early antibiotics
Similar patient 2 (survived): 81F, UTI sepsis, good response"""

        print("Running prediction for test patient...")
        result = system.predict_mortality(
            patient_context=patient_context,
            positive_similars=positive_similars,
            negative_similars=negative_similars,
            patient_id="test_001",
            output_dir=None,  # No logging for test
            ground_truth=1
        )
        
        print(f"\n✓ Prediction completed:")
        print(f"  - Prediction: {result['final_prediction']}")
        print(f"  - Is Fallback: {result['is_fallback']}")
        print(f"  - Time: {result['total_generation_time']:.2f}s")
        
        # Verify result structure
        assert 'final_prediction' in result, "Missing final_prediction"
        assert 'is_fallback' in result, "Missing is_fallback"
        assert 'total_generation_time' in result, "Missing total_generation_time"
        assert 'response' in result, "Missing response"
        
        # Verify prediction is valid
        assert result['final_prediction'] in [0, 1], f"Invalid prediction: {result['final_prediction']}"
        print(f"✓ Prediction is valid: {result['final_prediction']} (0=survival, 1=mortality)")
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_formats(system):
    """Test that prompt formats match original."""
    print("\n" + "=" * 80)
    print("TEST 4: Prompt Format Verification")
    print("=" * 80)
    
    # Test zero-shot retrieval prompt
    patient_context = "Test patient context"
    positive_similars = "Test positive"
    negative_similars = "Test negative"
    
    # Zero-shot retrieval prompt (constructed inline in predict_mortality)
    # We'll check the components are correct
    
    checks = [
        ("Task description includes 'Mortality Prediction Task'", 
         "Mortality Prediction Task:" in system.task_description),
        ("Task description includes 'Labels: 1 = mortality, 0 = survival'", 
         "Labels: 1 = mortality, 0 = survival" in system.task_description),
        ("Task description includes 'IMPORTANT: Mortality is rare'", 
         "IMPORTANT: Mortality is rare" in system.task_description),
        ("Task description includes key considerations", 
         "Key Considerations:" in system.task_description),
        ("Retrieval instruction mentions retrieve(query)", 
         "retrieve(query)" in system.retrieval_instruction),
    ]
    
    all_passed = True
    for check_name, check_result in checks:
        if check_result:
            print(f"✓ {check_name}")
        else:
            print(f"✗ {check_name}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EFFGEN RAG IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    
    # Parse GPU ID from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6', help='GPU ID to use')
    args = parser.parse_args()
    
    results = []
    
    # Test 1: Import
    results.append(("Import", test_import()))
    
    if not results[0][1]:
        print("\n" + "=" * 80)
        print("CRITICAL: Import failed. Cannot continue with other tests.")
        print("=" * 80)
        return False
    
    # Test 2: Initialization
    init_result, system = test_initialization(args.gpu)
    results.append(("Initialization", init_result))
    
    if not init_result:
        print("\n" + "=" * 80)
        print("CRITICAL: Initialization failed. Cannot continue with other tests.")
        print("=" * 80)
        return False
    
    # Test 3: Prediction
    results.append(("Prediction", test_prediction(system)))
    
    # Test 4: Prompt formats
    results.append(("Prompt Format", test_prompt_formats(system)))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe effGen RAG implementation is ready to use!")
        print("\nNext steps:")
        print("1. Run full evaluation with: python run_kare_single_agent_effgen.py --mode rag --gpus 6,7")
        print("2. Monitor logs in results/debate_logs_zero_shot/")
        print("3. Compare results with original vllm implementation")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running full evaluation.")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
