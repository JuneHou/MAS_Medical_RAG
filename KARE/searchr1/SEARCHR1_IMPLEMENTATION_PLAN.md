# Search-R1 Implementation Plan for KARE Mortality Prediction
**Single-Agent Baseline with MedRAG Retrieval**

**Date**: December 23, 2025  
**Goal**: Teach Qwen2.5-7B-Instruct to generate proper retrieve queries and use MedRAG at appropriate times for mortality prediction

---

## Overview

### Current System → Target System

**Current (5-agent debate)**:
```
Target Agent → Opponent Agents (3) → Integrator Agent
     ↓              ↓                        ↓
  Context      Debate rounds          Final prediction
```

**Target (Single-agent Search-R1)**:
```
Single Agent receives:
  - Target patient EHR
  - 2 similar patient EHRs
     ↓
Multi-turn retrieval with MedRAG
     ↓
Final mortality prediction (0 or 1)
```

### Key Simplifications

✅ **Remove**: 5-agent debate, debate rounds, target agent context  
✅ **Keep**: Integrator-style prompt, similar patients, MedRAG retrieval  
✅ **Add**: Multi-turn <think>-<search>-<answer> pattern, RL training for query optimization

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Search-R1 Single-Agent System                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: Target EHR + 2 Similar Patients                │
│              │                                          │
│              v                                          │
│  ┌────────────────────────────┐                        │
│  │   Qwen2.5-7B-Instruct      │                        │
│  │  (RL-trained with PPO)     │                        │
│  └────────────────────────────┘                        │
│              │                                          │
│              v                                          │
│  Multi-turn interaction:                               │
│    <think>Analyze patient risk...</think>              │
│    <search>sepsis mortality elderly ICU</search>       │
│              │                                          │
│              v                                          │
│  ┌────────────────────────────┐                        │
│  │   MedRAG Retriever         │                        │
│  │  (MedCPT + MedCorp2)       │                        │
│  └────────────────────────────┘                        │
│              │                                          │
│              v                                          │
│    <information>Retrieved docs...</information>        │
│    <think>Based on evidence...</think>                 │
│    <answer>1</answer>  (or more search rounds)         │
│                                                         │
└─────────────────────────────────────────────────────────┘

Reward: Binary accuracy (1.0 if prediction matches label, 0.0 otherwise)
```

---

## Phase 1: Data Preparation (Week 1)

### Step 1.1: Extract Integrator Prompt Structure

**Current integrator agent prompt** (from your debate system):
```python
# Locate in mortality_debate_rag.py
# Lines ~1560-1600: debate_mortality_prediction()
# The integrator agent receives:
# - Target patient context
# - Debate history (SKIP THIS)
# - Similar patients context
```

**Simplified single-agent prompt** (without debate):
```python
def create_single_agent_prompt(target_ehr, similar_patients):
    """
    Create Search-R1 prompt for single-agent mortality prediction.
    Mimics integrator agent but without debate context.
    """
    
    prompt = f"""You are a medical AI assistant specialized in mortality prediction. \
Your task is to predict whether a patient will DIE (1) or SURVIVE (0) in their next hospital visit.

You must conduct reasoning inside <think> and </think> tags every time you analyze information.
If you need additional medical evidence, search for relevant clinical information by writing <search>query</search>.
The search engine will return medical literature between <information> and </information> tags.
You can search multiple times to gather comprehensive evidence.
When confident in your prediction, provide your final answer as <answer>0</answer> (survival) or <answer>1</answer> (mortality).

## Target Patient
{target_ehr}

## Similar Patient Cases
For reference, here are two similar patient cases with known outcomes:

### Similar Patient 1 (Positive Example - Same Outcome Expected)
{similar_patients['positive']}

### Similar Patient 2 (Negative Example - Different Outcome Expected)
{similar_patients['negative']}

## Task
Analyze the target patient's clinical data, consider the similar cases, and predict the mortality outcome for the next visit.
You may search for additional medical evidence to support your reasoning."""

    return prompt
```

### Step 1.2: Create Data Formatter

Create `/data/wang/junh/githubs/Debate/KARE/scripts/prepare_searchr1_data.py`:

```python
#!/usr/bin/env python3
"""
Prepare KARE mortality data for Search-R1 single-agent training.
Based on integrator agent prompt structure without debate context.
"""

import json
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add KARE paths
sys.path.insert(0, '/data/wang/junh/githubs/Debate/KARE')

def load_kare_mortality_data(data_dir: str = '/data/wang/junh/datasets/KARE') -> Tuple[List[Dict], List[Dict]]:
    """
    Load KARE mortality prediction data.
    Returns train and test splits.
    
    Returns:
        (train_samples, test_samples)
        Each sample has:
        - patient_id: str
        - target_ehr: str (formatted EHR data)
        - positive_similar: str (similar patient with same outcome)
        - negative_similar: str (similar patient with different outcome)
        - label: int (0=survival, 1=mortality)
    """
    # TODO: Implement based on your KARE data structure
    # This depends on how your debate system loads data
    # You mentioned there's a data adapter in mortality_debate_rag.py
    
    # Placeholder - adapt to your actual data loading
    train_samples = []
    test_samples = []
    
    # Load from your KARE dataset
    # Example structure (modify as needed):
    # train_file = Path(data_dir) / "train_mortality.json"
    # with open(train_file) as f:
    #     train_data = json.load(f)
    #     for item in train_data:
    #         train_samples.append({
    #             'patient_id': item['patient_id'],
    #             'target_ehr': format_ehr(item['ehr_data']),
    #             'positive_similar': item['positive_similar_patient'],
    #             'negative_similar': item['negative_similar_patient'],
    #             'label': item['mortality_label']
    #         })
    
    return train_samples, test_samples


def format_ehr_data(ehr_dict: Dict) -> str:
    """
    Format EHR dictionary into readable text.
    Based on your current system's formatting.
    """
    sections = []
    
    # Demographics
    if 'demographics' in ehr_dict:
        demo = ehr_dict['demographics']
        sections.append(f"**Demographics**")
        sections.append(f"Age: {demo.get('age', 'N/A')}")
        sections.append(f"Gender: {demo.get('gender', 'N/A')}")
        sections.append("")
    
    # Diagnoses
    if 'diagnoses' in ehr_dict:
        sections.append(f"**Diagnoses**")
        for diag in ehr_dict['diagnoses'][:10]:  # Top 10
            sections.append(f"- {diag}")
        sections.append("")
    
    # Procedures
    if 'procedures' in ehr_dict:
        sections.append(f"**Procedures**")
        for proc in ehr_dict['procedures'][:10]:
            sections.append(f"- {proc}")
        sections.append("")
    
    # Medications
    if 'medications' in ehr_dict:
        sections.append(f"**Medications**")
        for med in ehr_dict['medications'][:10]:
            sections.append(f"- {med}")
        sections.append("")
    
    # Lab Results
    if 'labs' in ehr_dict:
        sections.append(f"**Laboratory Results**")
        for lab_name, value in list(ehr_dict['labs'].items())[:10]:
            sections.append(f"- {lab_name}: {value}")
        sections.append("")
    
    # Vital Signs
    if 'vitals' in ehr_dict:
        sections.append(f"**Vital Signs**")
        for vital_name, value in ehr_dict['vitals'].items():
            sections.append(f"- {vital_name}: {value}")
        sections.append("")
    
    return '\n'.join(sections)


def create_single_agent_prompt(sample: Dict) -> str:
    """
    Create Search-R1 prompt for single-agent mortality prediction.
    Mimics integrator agent structure without debate.
    """
    
    prompt = f"""You are a medical AI assistant specialized in mortality prediction. \
Your task is to predict whether a patient will DIE (1) or SURVIVE (0) in their next hospital visit.

You must conduct reasoning inside <think> and </think> tags every time you analyze information.
If you need additional medical evidence, search for relevant clinical information by writing <search>query</search>.
The search engine will return medical literature between <information> and </information> tags.
You can search multiple times to gather comprehensive evidence.
When confident in your prediction, provide your final answer as <answer>0</answer> (survival) or <answer>1</answer> (mortality).

## Target Patient
{sample['target_ehr']}

## Similar Patient Cases
For reference, here are two similar patient cases with known outcomes:

### Similar Patient 1 (Positive Example - Same Outcome Expected)
{sample['positive_similar']}

### Similar Patient 2 (Negative Example - Different Outcome Expected)
{sample['negative_similar']}

## Task
Analyze the target patient's clinical data, consider the similar cases, and predict the mortality outcome for the next visit.
You may search for additional medical evidence to support your reasoning."""

    return prompt


def format_sample_for_searchr1(sample: Dict, split: str = 'train') -> Dict:
    """
    Format a single KARE sample for Search-R1 training.
    
    Args:
        sample: KARE data sample with target_ehr, similar patients, label
        split: 'train' or 'test'
        
    Returns:
        Search-R1 formatted dictionary for parquet file
    """
    
    # Create prompt
    prompt_text = create_single_agent_prompt(sample)
    
    # Format in Search-R1 structure
    return {
        "data_source": "kare_mortality_single_agent",
        "prompt": [{
            "role": "user",
            "content": prompt_text,
        }],
        "ability": "medical-mortality-prediction",
        "reward_model": {
            "style": "rule",  # Rule-based reward: binary accuracy
            "ground_truth": {
                "target": [str(sample['label'])]  # "0" or "1"
            }
        },
        "extra_info": {
            'split': split,
            'patient_id': sample['patient_id'],
        }
    }


def create_searchr1_dataset(
    data_dir: str = '/data/wang/junh/datasets/KARE',
    output_dir: str = 'data/kare_mortality_single_agent'
):
    """
    Create Search-R1 training and validation datasets.
    """
    
    print("=" * 60)
    print("KARE Mortality → Search-R1 Single-Agent Data Preparation")
    print("=" * 60)
    
    # Load KARE data
    print("\n[1/4] Loading KARE mortality data...")
    train_samples, test_samples = load_kare_mortality_data(data_dir)
    print(f"  ✓ Loaded {len(train_samples)} training samples")
    print(f"  ✓ Loaded {len(test_samples)} test samples")
    
    # Format for Search-R1
    print("\n[2/4] Formatting training data for Search-R1...")
    train_formatted = []
    for idx, sample in enumerate(train_samples):
        formatted = format_sample_for_searchr1(sample, split='train')
        train_formatted.append(formatted)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_samples)} training samples...")
    
    print("\n[3/4] Formatting test data for Search-R1...")
    test_formatted = []
    for idx, sample in enumerate(test_samples):
        formatted = format_sample_for_searchr1(sample, split='test')
        test_formatted.append(formatted)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_samples)} test samples...")
    
    # Save as parquet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[4/4] Saving to {output_dir}...")
    
    train_df = pd.DataFrame(train_formatted)
    test_df = pd.DataFrame(test_formatted)
    
    train_path = output_path / 'train.parquet'
    test_path = output_path / 'test.parquet'
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"  ✓ Saved training data: {train_path}")
    print(f"  ✓ Saved test data: {test_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Training samples: {len(train_formatted)}")
    print(f"Test samples: {len(test_formatted)}")
    
    # Label distribution
    train_labels = [s['reward_model']['ground_truth']['target'][0] for s in train_formatted]
    test_labels = [s['reward_model']['ground_truth']['target'][0] for s in test_formatted]
    
    train_survival = train_labels.count('0')
    train_mortality = train_labels.count('1')
    test_survival = test_labels.count('0')
    test_mortality = test_labels.count('1')
    
    print(f"\nLabel Distribution:")
    print(f"  Training:")
    print(f"    - Survival (0): {train_survival} ({100*train_survival/len(train_labels):.1f}%)")
    print(f"    - Mortality (1): {train_mortality} ({100*train_mortality/len(train_labels):.1f}%)")
    print(f"  Test:")
    print(f"    - Survival (0): {test_survival} ({100*test_survival/len(test_labels):.1f}%)")
    print(f"    - Mortality (1): {test_mortality} ({100*test_mortality/len(test_labels):.1f}%)")
    
    print("\n✓ Data preparation complete!")
    print(f"Next step: Wrap MedRAG as retrieval server")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare KARE data for Search-R1 training')
    parser.add_argument('--data_dir', type=str, default='/data/wang/junh/datasets/KARE',
                        help='KARE data directory')
    parser.add_argument('--output_dir', type=str, default='data/kare_mortality_single_agent',
                        help='Output directory for Search-R1 formatted data')
    
    args = parser.parse_args()
    
    create_searchr1_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
```

**TODO for you**: Adapt `load_kare_mortality_data()` to match your actual KARE data structure.

---

## Phase 2: MedRAG Retrieval Server (Week 1)

### Step 2.1: Wrap MedRAG as FastAPI Server

Create `/data/wang/junh/githubs/Debate/KARE/scripts/medrag_retrieval_server.py`:

```python
#!/usr/bin/env python3
"""
MedRAG Retrieval Server for Search-R1.
Wraps your existing MedRAG+MedCPT+MedCorp2 as FastAPI server.
"""

import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import logging

# Add MedRAG paths
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))

from medrag import MedRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="MedRAG Retrieval Server for Search-R1")

# Global MedRAG instance
medrag_retriever = None


class QueryRequest(BaseModel):
    """Request schema for retrieval"""
    queries: List[str]
    topk: Optional[int] = 8
    return_scores: bool = False


class RetrievalResponse(BaseModel):
    """Response schema for retrieval"""
    result: List[List[Dict[str, str]]]
    scores: Optional[List[List[float]]] = None


def initialize_medrag(
    retriever_name: str = "MedCPT",
    corpus_name: str = "MedCorp2",
    db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"
):
    """Initialize MedRAG retriever once at startup"""
    global medrag_retriever
    
    logger.info("=" * 60)
    logger.info("Initializing MedRAG Retrieval Server")
    logger.info("=" * 60)
    logger.info(f"Retriever: {retriever_name}")
    logger.info(f"Corpus: {corpus_name}")
    logger.info(f"DB Directory: {db_dir}")
    
    try:
        medrag_retriever = MedRAG(
            llm_name="placeholder",  # Not used for retrieval-only
            rag=True,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir
        )
        logger.info("✓ MedRAG initialized successfully")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to initialize MedRAG: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize MedRAG on server startup"""
    initialize_medrag()


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "MedRAG Retrieval Server",
        "retriever": "MedCPT",
        "corpus": "MedCorp2"
    }


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_endpoint(request: QueryRequest):
    """
    Search-R1 compatible retrieval endpoint.
    
    Args:
        request: QueryRequest with queries list and topk
        
    Returns:
        RetrievalResponse with retrieved documents
    """
    if medrag_retriever is None:
        raise HTTPException(status_code=500, detail="MedRAG not initialized")
    
    try:
        logger.info(f"Received retrieval request: {len(request.queries)} queries, topk={request.topk}")
        
        results = []
        scores = []
        
        for query in request.queries:
            logger.debug(f"Processing query: {query[:100]}...")
            
            # Use your existing MedRAG retrieval
            # Adjust based on your MedRAG version's API
            retrieved_docs = medrag_retriever.retrieve(
                question=query,
                k=request.topk or 8
            )
            
            # Format for Search-R1
            # Expected format: List[Dict] with 'title', 'text', 'contents' keys
            formatted_docs = []
            for doc in retrieved_docs:
                # Adapt based on your MedRAG's return format
                # MedRAG typically returns: {'title': ..., 'content': ...}
                formatted_docs.append({
                    'title': doc.get('title', ''),
                    'text': doc.get('content', doc.get('text', '')),
                    'contents': doc.get('content', doc.get('text', ''))
                })
            
            results.append(formatted_docs)
            
            if request.return_scores:
                # Extract scores if available
                doc_scores = [doc.get('score', 0.0) for doc in retrieved_docs]
                scores.append(doc_scores)
        
        logger.info(f"✓ Retrieved {sum(len(r) for r in results)} total documents")
        
        if request.return_scores:
            return RetrievalResponse(result=results, scores=scores)
        else:
            return RetrievalResponse(result=results)
    
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.get("/stats")
def stats():
    """Get retriever statistics"""
    if medrag_retriever is None:
        raise HTTPException(status_code=500, detail="MedRAG not initialized")
    
    return {
        "retriever": "MedCPT",
        "corpus": "MedCorp2",
        "status": "ready"
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MedRAG Retrieval Server for Search-R1')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--retriever', type=str, default='MedCPT', help='Retriever name')
    parser.add_argument('--corpus', type=str, default='MedCorp2', help='Corpus name')
    parser.add_argument('--db_dir', type=str, 
                        default='/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus',
                        help='Database directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting MedRAG Retrieval Server for Search-R1")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Retriever: {args.retriever}")
    print(f"Corpus: {args.corpus}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
```

### Step 2.2: Test MedRAG Server

Create `/data/wang/junh/githubs/Debate/KARE/scripts/test_medrag_server.py`:

```python
#!/usr/bin/env python3
"""Test MedRAG retrieval server"""

import requests
import json

def test_retrieval_server(url: str = "http://127.0.0.1:8000"):
    """Test MedRAG retrieval server"""
    
    print("=" * 60)
    print("Testing MedRAG Retrieval Server")
    print("=" * 60)
    
    # Health check
    print("\n[1/3] Health check...")
    response = requests.get(f"{url}/")
    print(f"  Status: {response.json()}")
    
    # Test retrieval
    print("\n[2/3] Testing retrieval...")
    test_queries = [
        "sepsis mortality risk factors in elderly ICU patients",
        "heart failure prognosis NYHA class IV",
        "pneumonia outcomes in immunocompromised patients"
    ]
    
    payload = {
        "queries": test_queries,
        "topk": 5,
        "return_scores": True
    }
    
    response = requests.post(f"{url}/retrieve", json=payload)
    result = response.json()
    
    print(f"  Retrieved documents for {len(result['result'])} queries")
    for i, docs in enumerate(result['result']):
        print(f"\n  Query {i+1}: '{test_queries[i][:60]}...'")
        print(f"    Retrieved {len(docs)} documents")
        if docs:
            print(f"    Top result: {docs[0].get('title', 'N/A')[:80]}...")
    
    # Stats
    print("\n[3/3] Server stats...")
    response = requests.get(f"{url}/stats")
    print(f"  {response.json()}")
    
    print("\n" + "=" * 60)
    print("✓ MedRAG server test complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000',
                        help='Server URL')
    args = parser.parse_args()
    
    test_retrieval_server(args.url)
```

---

## Phase 3: Search-R1 Training Configuration (Week 2)

### Step 3.1: Training Script

Create `/data/wang/junh/githubs/Debate/KARE/scripts/train_searchr1_single_agent.sh`:

```bash
#!/bin/bash
# Train Search-R1 for KARE mortality prediction (single-agent baseline)

set -e  # Exit on error

echo "=========================================="
echo "Search-R1 Single-Agent Training for KARE"
echo "=========================================="

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='kare-mortality-single-agent-searchr1-v1'
export DATA_DIR='data/kare_mortality_single_agent'
export WANDB_PROJECT='KARE-Mortality-Search-R1'
export VLLM_ATTENTION_BACKEND=XFORMERS

# Paths
SEARCH_R1_ROOT="/data/wang/junh/githubs/Search-R1"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# Check retriever server
echo ""
echo "[1/3] Checking MedRAG retriever server..."
if ! curl -s "${RETRIEVER_URL%/retrieve}/" > /dev/null; then
    echo "ERROR: MedRAG server not running on port 8000"
    echo "Start it with:"
    echo "  cd /data/wang/junh/githubs/Debate/KARE/scripts"
    echo "  python medrag_retrieval_server.py --port 8000"
    exit 1
fi
echo "✓ MedRAG server is running"

# Check data
echo ""
echo "[2/3] Checking training data..."
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.parquet"
    echo "Run data preparation first:"
    echo "  python scripts/prepare_searchr1_data.py --output_dir $DATA_DIR"
    exit 1
fi
echo "✓ Training data found"

# Training
echo ""
echo "[3/3] Starting Search-R1 training..."
echo "  Model: $BASE_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Data: $DATA_DIR"
echo "  Retriever: $RETRIEVER_URL"
echo ""

cd "$SEARCH_R1_ROOT"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=800 \
    data.max_start_length=3072 \
    data.max_obs_length=600 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path="$BASE_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=20 \
    trainer.total_training_steps=2000 \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=8

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Checkpoints: verl_checkpoints/$EXPERIMENT_NAME"
echo "=========================================="
```

### Step 3.2: Quick Test Script (Small Scale)

Create `/data/wang/junh/githubs/Debate/KARE/scripts/test_searchr1_training.sh`:

```bash
#!/bin/bash
# Quick test run with 10 samples to verify setup

set -e

echo "=========================================="
echo "Search-R1 Quick Test (10 samples)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='kare-test-searchr1'
export DATA_DIR='data/kare_mortality_single_agent'
export VLLM_ATTENTION_BACKEND=XFORMERS

SEARCH_R1_ROOT="/data/wang/junh/githubs/Search-R1"
cd "$SEARCH_R1_ROOT"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=10 \
    data.val_data_num=5 \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    trainer.total_epochs=2 \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=5

echo "✓ Test complete!"
```

---

## Phase 4: Evaluation & Analysis (Week 3)

### Step 4.1: Inference Script

Create `/data/wang/junh/githubs/Debate/KARE/scripts/evaluate_searchr1.py`:

```python
#!/usr/bin/env python3
"""
Evaluate Search-R1 trained model on KARE mortality prediction.
"""

import sys
import torch
from pathlib import Path
from typing import List, Dict
import pandas as pd
import json
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import argparse

# Add Search-R1 paths
sys.path.insert(0, '/data/wang/junh/githubs/Search-R1')

# TODO: Import Search-R1 inference utilities
# from search_r1.llm_agent.generation import LLMGenerationManager


def load_model(checkpoint_path: str):
    """Load Search-R1 trained model"""
    # TODO: Implement based on Search-R1's model loading
    print(f"Loading model from {checkpoint_path}...")
    # model = load_search_r1_model(checkpoint_path)
    # return model
    pass


def extract_prediction(response: str) -> int:
    """
    Extract mortality prediction from model response.
    
    Args:
        response: Model response with <answer>0</answer> or <answer>1</answer>
        
    Returns:
        0 (survival) or 1 (mortality)
    """
    import re
    
    # Extract answer tag
    match = re.search(r'<answer>\s*([01])\s*</answer>', response)
    if match:
        return int(match.group(1))
    
    # Fallback: look for keywords
    response_lower = response.lower()
    if 'mortality' in response_lower or 'die' in response_lower:
        return 1
    elif 'survival' in response_lower or 'survive' in response_lower:
        return 0
    
    # Default to 0 if unclear
    return 0


def evaluate_model(
    model,
    test_data_path: str,
    retriever_url: str = "http://127.0.0.1:8000/retrieve",
    max_turns: int = 3
) -> Dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with metrics and predictions
    """
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_parquet(test_data_path)
    
    print(f"Evaluating on {len(df)} samples...")
    
    predictions = []
    ground_truths = []
    responses = []
    search_queries = []
    
    for idx, row in df.iterrows():
        # Get prompt
        prompt = row['prompt'][0]['content']
        ground_truth = int(row['reward_model']['ground_truth']['target'][0])
        
        # Generate response with Search-R1
        # TODO: Implement inference call
        # response = model.generate(
        #     prompt=prompt,
        #     max_turns=max_turns,
        #     retriever_url=retriever_url
        # )
        response = "<think>Analysis...</think><answer>0</answer>"  # Placeholder
        
        # Extract prediction
        pred = extract_prediction(response)
        
        predictions.append(pred)
        ground_truths.append(ground_truth)
        responses.append(response)
        
        # Extract search queries from response
        import re
        queries = re.findall(r'<search>(.*?)</search>', response, re.DOTALL)
        search_queries.append(queries)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Try AUROC if we have probabilities (for now, binary)
    try:
        auroc = roc_auc_score(ground_truths, predictions)
    except:
        auroc = None
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average='binary'
    )
    
    # Analyze search behavior
    num_searches = [len(q) for q in search_queries]
    avg_searches = sum(num_searches) / len(num_searches)
    
    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_searches_per_sample': avg_searches,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'responses': responses,
        'search_queries': search_queries
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    if results['auroc']:
        print(f"AUROC: {results['auroc']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"\nAverage searches per sample: {results['avg_searches_per_sample']:.2f}")
    print("=" * 60)


def analyze_search_patterns(results: Dict):
    """Analyze learned search patterns"""
    print("\n" + "=" * 60)
    print("Search Pattern Analysis")
    print("=" * 60)
    
    all_queries = []
    for queries in results['search_queries']:
        all_queries.extend(queries)
    
    print(f"Total searches performed: {len(all_queries)}")
    
    # Most common search terms
    from collections import Counter
    words = []
    for query in all_queries:
        words.extend(query.lower().split())
    
    common_words = Counter(words).most_common(20)
    print("\nMost common search terms:")
    for word, count in common_words:
        print(f"  {word}: {count}")
    
    # Sample queries
    print("\nSample search queries:")
    for query in all_queries[:10]:
        print(f"  - {query}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Search-R1 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str,
                        default='data/kare_mortality_single_agent/test.parquet',
                        help='Test data path')
    parser.add_argument('--retriever_url', type=str,
                        default='http://127.0.0.1:8000/retrieve',
                        help='Retriever server URL')
    parser.add_argument('--max_turns', type=int, default=3,
                        help='Max search turns')
    parser.add_argument('--output', type=str, default='results/searchr1_eval.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Evaluate
    results = evaluate_model(
        model=model,
        test_data_path=args.test_data,
        retriever_url=args.retriever_url,
        max_turns=args.max_turns
    )
    
    # Print results
    print_results(results)
    analyze_search_patterns(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    results_json = {
        'accuracy': results['accuracy'],
        'auroc': results['auroc'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'avg_searches_per_sample': results['avg_searches_per_sample'],
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
```

---

## Implementation Timeline

### Week 1: Data & Retrieval Setup

**Day 1-2**: Data Preparation
- [ ] Adapt `load_kare_mortality_data()` to your KARE structure
- [ ] Run `python scripts/prepare_searchr1_data.py`
- [ ] Verify train/test parquet files

**Day 3**: MedRAG Server Setup
- [ ] Test MedRAG import and initialization
- [ ] Start `python scripts/medrag_retrieval_server.py --port 8000`
- [ ] Run `python scripts/test_medrag_server.py`

**Day 4-5**: Data Validation
- [ ] Inspect generated prompts (check similarity to integrator agent)
- [ ] Verify label distribution
- [ ] Test retrieval with sample queries

### Week 2: Training

**Day 1**: Environment Setup
- [ ] Clone Search-R1 if not already done: `git clone https://github.com/PeterGriffinJin/Search-R1.git`
- [ ] Install dependencies: `cd Search-R1 && pip install -r requirements.txt`
- [ ] Test Search-R1 with their examples

**Day 2**: Quick Test Run
- [ ] Run `bash scripts/test_searchr1_training.sh` (10 samples)
- [ ] Verify training loop works
- [ ] Check retrieval integration
- [ ] Monitor GPU usage and memory

**Day 3-5**: Full Training
- [ ] Launch `bash scripts/train_searchr1_single_agent.sh`
- [ ] Monitor training metrics on W&B
- [ ] Watch for:
  - Reward increasing over time
  - Search queries becoming more relevant
  - Accuracy improving on validation set

### Week 3: Evaluation & Analysis

**Day 1-2**: Model Evaluation
- [ ] Implement inference in `evaluate_searchr1.py`
- [ ] Run evaluation on test set
- [ ] Calculate accuracy, AUROC, F1

**Day 3**: Search Pattern Analysis
- [ ] Analyze learned search queries
- [ ] Compare to your manual retrieval in debate system
- [ ] Identify effective vs ineffective queries

**Day 4-5**: Comparison & Integration
- [ ] Compare single-agent Search-R1 vs current 5-agent debate
- [ ] If promising, plan integration with debate system
- [ ] Document findings

---

## Expected Outcomes

### Baseline Performance (Single-Agent)

Without debate, expect:
- **Accuracy**: 60-70% (vs your debate system's current performance)
- **Search behavior**: 1-3 searches per sample on average
- **Learned queries**: Medical terminology, risk factors, outcome predictors

### Search-R1 Improvements

After RL training, model should learn to:
1. **Generate relevant medical queries**: "sepsis SOFA score mortality ICU elderly"
2. **Search at appropriate times**: Before making high-risk predictions
3. **Multi-turn refinement**: Initial broad search → specific follow-up
4. **Balance speed vs accuracy**: Not searching for obvious cases

### Comparison to Debate System

| Metric | 5-Agent Debate | Single-Agent Search-R1 |
|--------|----------------|------------------------|
| **Complexity** | High (5 agents, 3 rounds) | Low (1 agent, 3 turns) |
| **Interpretability** | Medium (debate history) | High (<think> tags) |
| **Training Time** | N/A (prompt-based) | ~1-2 days |
| **Inference Speed** | Slow (multiple LLM calls) | Fast (1 agent) |
| **Retrieval** | Manual in tools | **Learned with RL** |

---

## Next Steps After Search-R1

If single-agent Search-R1 works well, you can:

### Option 1: Enhance Single Agent
- Add more similar patients (2 → 5)
- Increase max_turns (3 → 5)
- Fine-tune reward function (binary → AUROC-based)

### Option 2: Hybrid Debate + Search-R1
- Use Search-R1 for **query generation** in debate agents
- Replace manual retrieval tools with RL-learned queries
- Keep debate structure for reasoning

### Option 3: Multi-Agent Search-R1
- Train separate Search-R1 models for each debate role
- Target agent: risk identification queries
- Opponent agents: protective factor queries
- Integrator: comprehensive synthesis queries

---

## Troubleshooting

### Issue: MedRAG server fails to start
**Solution**: Check paths and dependencies
```bash
cd /data/wang/junh/githubs/mirage_medrag/MedRAG
pip install -r requirements.txt
python -c "from medrag import MedRAG; print('✓ Import successful')"
```

### Issue: Search-R1 training OOM
**Solution**: Reduce batch sizes
```bash
# In train script, adjust:
data.train_batch_size=128  # from 256
actor_rollout_ref.actor.ppo_micro_batch_size=16  # from 32
```

### Issue: Model doesn't learn to search
**Solution**: Check reward signal
- Verify retrieval server is responding
- Check if <information> blocks are properly formatted
- Ensure reward function is not saturated (all 0 or all 1)

### Issue: Too many/too few searches
**Solution**: Adjust prompting or add search budget reward
```python
# In reward function:
num_searches = len(re.findall(r'<search>', response))
search_penalty = -0.05 * max(0, num_searches - 3)  # Penalize >3 searches
reward = accuracy_reward + search_penalty
```

---

## Summary

This plan gives you:
✅ **Single-agent baseline** (simpler than 5-agent debate)  
✅ **Integrator-style prompt** (reuses your current structure)  
✅ **MedRAG integration** (wraps your existing retriever)  
✅ **RL-learned queries** (Search-R1's core innovation)  
✅ **Clear timeline** (3 weeks to first results)  

The key advantage over your current debate system: **Learn which queries help mortality prediction through RL**, rather than manually designing retrieval strategies.

Good luck with implementation! Start with Phase 1 data preparation and let me know if you hit any roadblocks.
