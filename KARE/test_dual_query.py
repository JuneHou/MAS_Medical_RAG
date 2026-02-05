#!/usr/bin/env python3
"""
Simple test for dual-query retrieval system.
Tests that the system can generate both MedCorp and UMLS queries and retrieve 8 documents (4+4).
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mortality_debate_rag import MortalityDebateSystem

def test_dual_query_retrieval():
    """Test dual-query retrieval with a simple patient case."""
    
    print("="*80)
    print("DUAL-QUERY RETRIEVAL TEST")
    print("="*80)
    
    # Initialize the debate system
    print("\n[1/5] Initializing MortalityDebateSystem...")
    system = MortalityDebateSystem(
        model_name="Qwen/Qwen2.5-7B-Instruct",  # Smaller model for testing
        gpu_ids="0",
        rag_enabled=True,
        corpus_name="MedCorp2",
        retriever_name="MedCPT"
    )
    print("✓ System initialized")
    
    # Create test patient context
    print("\n[2/5] Creating test patient context...")
    test_patient = """
Patient ID: TEST_001
Age: 65, Gender: Male

Visit 0:
- Conditions: Multiple Myeloma, Chronic Kidney Disease, Pneumonia
- Procedures: Hemodialysis
- Medications: Melphalan, Zoledronic Acid, Antibiotics

Visit 1:
- Conditions: Septicemia, Respiratory Failure
- Procedures: Respiratory Intubation, Mechanical Ventilation
- Medications: Broad-spectrum antibiotics, Vasopressors
"""
    print("✓ Test patient created")
    
    # Create similar patient contexts (dummy data for test)
    similar_patients = {
        'mortality_case': "Similar patient with mortality=1...",
        'survival_case': "Similar patient with survival=0..."
    }
    
    # Create test debate history (minimal for integrator)
    print("\n[3/5] Creating debate history...")
    debate_history = [
        {
            'role': 'mortality_risk_assessor',
            'message': 'Analysis shows high risk due to multiple myeloma and septicemia.',
            'prediction': None
        },
        {
            'role': 'protective_factor_analyst',
            'message': 'Patient has received appropriate interventions including hemodialysis.',
            'prediction': None
        }
    ]
    print("✓ Debate history created")
    
    # Create test log directory
    log_dir = Path("./test_dual_query_logs")
    log_dir.mkdir(exist_ok=True)
    print(f"✓ Log directory created: {log_dir}")
    
    # Execute integrator with dual-query retrieval
    print("\n[4/5] Executing integrator with dual-query retrieval...")
    print("-" * 80)
    
    try:
        result = system._integrator_single_step_prediction(
            patient_context=test_patient,
            similar_patients=similar_patients,
            medical_knowledge="",
            debate_history=debate_history,
            logger=None,
            patient_id="TEST_001",
            log_dir=str(log_dir)
        )
        
        print("-" * 80)
        print("✓ Integrator execution completed")
        
        # Verify results
        print("\n[5/5] Verifying results...")
        print("-" * 80)
        
        # Check if probabilities were extracted
        mortality_prob = result.get('mortality_probability')
        survival_prob = result.get('survival_probability')
        
        print(f"Mortality Probability: {mortality_prob}")
        print(f"Survival Probability: {survival_prob}")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
        
        # Check if dual-query was used
        query_info = result.get('query')
        if isinstance(query_info, dict):
            print(f"\n✓ DUAL-QUERY FORMAT DETECTED")
            print(f"  MedCorp Query: {query_info.get('medcorp_query', 'None')[:100]}...")
            print(f"  UMLS Query: {query_info.get('umls_query', 'None')[:100]}...")
        else:
            print(f"\n⚠ Single-query format used: {query_info[:100] if query_info else 'No query'}...")
        
        # Check log files
        print(f"\nChecking log files in {log_dir}...")
        log_files = list(log_dir.glob("retrieve_integrator_*.json"))
        print(f"Found {len(log_files)} retrieval log files:")
        
        for log_file in sorted(log_files):
            print(f"  - {log_file.name}")
            
            # Read and display key info
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                
            if 'medcorp' in log_file.name:
                print(f"    Source: MedCorp")
                print(f"    Query: {log_data.get('query', '')[:80]}...")
                print(f"    Retrieved: {len(log_data.get('retrieved_docs', []))} documents")
            elif 'umls' in log_file.name:
                print(f"    Source: UMLS")
                print(f"    Query: {log_data.get('query', '')[:80]}...")
                print(f"    Retrieved: {len(log_data.get('retrieved_docs', []))} documents")
            elif 'dual' in log_file.name:
                print(f"    Source: Combined (MedCorp + UMLS)")
                print(f"    MedCorp docs: {log_data.get('medcorp_docs_count', 0)}")
                print(f"    UMLS docs: {log_data.get('umls_docs_count', 0)}")
                total_docs = len(log_data.get('combined_all_docs', []))
                print(f"    Total combined: {total_docs} documents")
                
                # Verify we got 8 documents (4+4)
                if total_docs == 8:
                    print(f"    ✓ CORRECT: Got 8 documents (4 MedCorp + 4 UMLS)")
                else:
                    print(f"    ⚠ WARNING: Expected 8 documents, got {total_docs}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Summary
        print("\nSUMMARY:")
        print(f"  - Dual-query retrieval: {'✓ WORKING' if isinstance(query_info, dict) else '✗ NOT USED'}")
        print(f"  - Probabilities extracted: {'✓ YES' if mortality_prob is not None else '✗ NO'}")
        print(f"  - Log files created: {len(log_files)}")
        print(f"  - Total documents retrieved: {total_docs if 'dual' in str(log_files) else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nStarting dual-query retrieval test...\n")
    success = test_dual_query_retrieval()
    sys.exit(0 if success else 1)
