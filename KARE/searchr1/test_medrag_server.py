#!/usr/bin/env python3
"""
Test MedRAG retrieval server for Search-R1.
Verifies server is running and retrieval works correctly.
"""

import requests
import json
import sys


def test_retrieval_server(url: str = "http://127.0.0.1:8000"):
    """Test MedRAG retrieval server"""
    
    print("=" * 80)
    print("Testing MedRAG Retrieval Server for Search-R1")
    print("=" * 80)
    
    # Test 1: Health check
    print("\n[1/4] Health check...")
    try:
        response = requests.get(f"{url}/")
        print(f"  ✓ Server status: {response.json()}")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        print("\nMake sure server is running:")
        print("  python searchr1/medrag_retrieval_server.py --port 8000")
        sys.exit(1)
    
    # Test 2: Single query retrieval
    print("\n[2/4] Testing single query retrieval...")
    test_query = "sepsis mortality risk factors in elderly ICU patients"
    
    payload = {
        "queries": [test_query],
        "topk": 5,
        "return_scores": True
    }
    
    try:
        response = requests.post(f"{url}/retrieve", json=payload)
        result = response.json()
        
        docs = result['result'][0]
        scores = result.get('scores', [[]])[0]
        
        print(f"  ✓ Retrieved {len(docs)} documents")
        print(f"  Query: '{test_query[:60]}...'")
        
        if docs:
            print(f"\n  Top result:")
            # Search-R1 format: doc['document']['title'], doc['document']['contents']
            top_doc = docs[0]['document']
            print(f"    Title: {top_doc.get('title', 'N/A')[:80]}")
            print(f"    Content preview: {top_doc.get('contents', '')[:200]}...")
            if scores:
                print(f"    Score: {scores[0]:.4f}")
    except Exception as e:
        print(f"  ❌ Single query test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Batch query retrieval
    print("\n[3/4] Testing batch query retrieval...")
    test_queries = [
        "heart failure prognosis NYHA class IV",
        "pneumonia outcomes immunocompromised patients",
        "acute kidney injury mortality predictors"
    ]
    
    payload = {
        "queries": test_queries,
        "topk": 3,
        "return_scores": False
    }
    
    try:
        response = requests.post(f"{url}/retrieve", json=payload)
        result = response.json()
        
        print(f"  ✓ Processed {len(result['result'])} queries")
        for i, docs in enumerate(result['result']):
            print(f"    Query {i+1}: Retrieved {len(docs)} documents")
    except Exception as e:
        print(f"  ❌ Batch query test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 4: Server stats
    print("\n[4/4] Checking server stats...")
    try:
        response = requests.get(f"{url}/stats")
        stats = response.json()
        print(f"  ✓ Server stats: {stats}")
    except Exception as e:
        print(f"  ❌ Stats check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ All tests passed! MedRAG server is ready for Search-R1 training")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Keep server running in background: nohup python searchr1/medrag_retrieval_server.py &")
    print("  2. Generate training data: python searchr1/data_generation/prepare_searchr1_balanced_data.py \\")
    print("       --balanced_json verl/data_generation/prediction/train_balanced_100pos_100neg.json \\")
    print("       --split train")
    print("  3. Start Search-R1 training: bash searchr1/train_searchr1_single_agent.sh")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MedRAG retrieval server')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000',
                        help='Server URL (default: http://127.0.0.1:8000)')
    
    args = parser.parse_args()
    
    test_retrieval_server(args.url)
