# AgentGym-RL Deep Search Analysis for Retrieve Query Post-Training

## Overview

Based on my analysis of the AgentGym-RL codebase, here's what I found about using Deep Search for post-training retrieve queries in your KARE mortality debate system.

## Key Findings

### 1. **Deep Search Environment (SearchQA)**

AgentGym-RL includes a **SearchQA** environment built on Search-R1, which is a RAG-based environment for multi-turn retrieval and reasoning tasks.

**Location**: `/data/wang/junh/githubs/AgentGym/agentenv-searchqa/`

**Key Components**:
- **Environment Wrapper** (`env_wrapper.py`): Handles question-answering with retrieval
- **Retriever** (`retriever.py`): Implements dense retrieval using models like E5, BGE, DPR
- **Server** (`server.py`): Provides HTTP API for environment interaction

### 2. **How SearchQA Works**

The SearchQA environment follows this interaction pattern:

```
1. Agent receives a question
2. Agent can perform two actions:
   - <search>query text</search> → Retrieves relevant documents
   - <answer>answer text</answer> → Provides final answer
3. Environment returns:
   - For search: <information>retrieved_doc</information>
   - For answer: Reward (1.0 if correct, 0.0 if incorrect)
4. Multi-turn interaction until correct answer or max turns
```

**Example Prompt Format** (from `env_wrapper.py` line 203):
```
You must reason inside <think>...</think> first. 
If you do not have enough knowledge, issue a <search>...</search> and then STOP. 
Wait for external input wrapped in <information>...</information>. 
After receiving information, reason again in <think>. 
If confident, output your final answer in <answer>...</answer>.

Question: {question}
```

### 3. **RL Training Infrastructure**

**Training Script**: `/data/wang/junh/githubs/AgentGym-RL/examples/train/AgentGym-RL/searchqa_train.sh`

**Key Configuration**:
- Uses PPO/GRPO for RL optimization
- Multi-turn rollouts with environment interaction
- Reward based on exact match (EM) score
- Configurable max rounds (default 5)

**Critical Parameters**:
```bash
algorithm.rounds_ctrl.type=fixed           # or scaling_inter_stepwise
algorithm.rounds_ctrl.rounds=5             # max interaction turns
actor_rollout_ref.agentgym.task_name=searchqa
actor_rollout_ref.rollout.max_tokens=512   # per turn generation
data.max_response_length=8192              # total trajectory length
```

### 4. **Retrieval Query Generation Training**

The SearchQA environment specifically trains agents to:
1. **Generate effective search queries** within `<search>...</search>` tags
2. **Decide when to retrieve** vs. when to answer directly
3. **Reason with retrieved information** to formulate answers
4. **Multi-turn refinement** if initial retrieval is insufficient

This is **directly relevant** to your mortality debate system's `retrieve()` tool calls!

---

## Can You Reuse This for Your KARE Mortality Debate?

### ✅ **YES - With Modifications**

Your current `mortality_debate_rag.py` file already has a similar structure to AgentGym's SearchQA:

| **Your System** | **AgentGym SearchQA** |
|-----------------|----------------------|
| `retrieve(query)` tool | `<search>query</search>` action |
| MedRAG corpus retrieval | Wikipedia corpus retrieval |
| Mortality prediction task | QA task |
| 3-round debate structure | Multi-turn interaction |

### Key Differences:

1. **Task Type**: 
   - AgentGym: Open-domain QA
   - Your system: Clinical mortality prediction

2. **Reward Signal**:
   - AgentGym: Exact match (binary)
   - Your system: Mortality prediction accuracy

3. **Retrieval Corpus**:
   - AgentGym: Wikipedia
   - Your system: MedCorp2 medical corpus

---

## Recommended Approach: Adapting AgentGym for KARE

### Option 1: **Create Custom KARE Environment for AgentGym**

Following AgentGym's modular design, create a new environment:

```
/data/wang/junh/githubs/AgentGym/agentenv-kare-mortality/
├── agentenv_kare_mortality/
│   ├── __init__.py
│   ├── env_wrapper.py       # KARE mortality environment
│   ├── retriever.py          # MedRAG integration
│   ├── server.py             # HTTP API server
│   └── reward_score/
│       └── mortality_em.py   # Mortality prediction reward
├── setup.sh
└── pyproject.toml
```

**Benefits**:
- ✅ Reuse AgentGym's RL infrastructure (PPO, GRPO, rollout handling)
- ✅ Leverage ScalingInter-RL for progressive training
- ✅ Built-in logging and visualization tools
- ✅ Multi-process environment parallelization

**Implementation Steps**:

1. **Create KARE Environment Server** (similar to `SearchQAEnvServer`):
   ```python
   class KAREMortalityEnvServer:
       def __init__(self):
           self.data_adapter = KAREDataAdapter()
           self.medrag = MedRAG(...)  # Your existing MedRAG setup
       
       def observation(self, env_idx):
           patient_data = self.env[env_idx]
           return f"""You are analyzing a patient for mortality risk.
           Available tools:
           - retrieve(query): Retrieve medical evidence
           
           Patient Context: {patient_data['context']}
           
           Provide your analysis and prediction."""
       
       def step(self, env_idx, response: str):
           # Parse retrieve() calls or prediction
           if "retrieve(" in response:
               query = self._parse_retrieve(response)
               docs = self.medrag.retrieve(query, k=8)
               observation = f"<evidence>{docs}</evidence>"
               reward = 0
               done = False
           elif "prediction:" in response:
               pred = self._parse_prediction(response)
               reward = 1.0 if pred == ground_truth else 0.0
               done = True
               observation = "Task complete"
           return observation, reward, done, None
   ```

2. **Modify Training Script** (`kare_mortality_train.sh`):
   ```bash
   task_name="kare_mortality"
   env_server_url="http://127.0.0.1:36010"  # Your KARE server
   
   # Use AgentGym-RL's PPO/GRPO training
   python3 -m verl.agent_trainer.main_ppo \
       algorithm.adv_estimator=grpo \
       algorithm.rounds_ctrl.type=scaling_inter_stepwise \
       algorithm.rounds_ctrl.rounds=3 \
       algorithm.rounds_ctrl.steps_scaling_inter=50 \
       data.train_file=kare_mortality_train.json \
       actor_rollout_ref.agentgym.task_name=kare_mortality \
       actor_rollout_ref.agentgym.env_addr=${env_server_url} \
       ...
   ```

3. **Training Data Format** (JSON file):
   ```json
   [
       {
           "item_id": "kare_mortality_0",
           "prompt": "Analyze this patient for mortality risk..."
       },
       ...
   ]
   ```

### Option 2: **Use Your Current System with Logging for Offline RL**

If you want to avoid full AgentGym integration:

1. **Add detailed logging** to your current `mortality_debate_rag.py`
2. **Collect interaction trajectories** (retrieve queries, responses, rewards)
3. **Use offline RL** (DPO, AgentEvol) to train on collected data

**Modifications to your current file**:

```python
# In mortality_debate_rag.py, add trajectory logging
class MortalityDebateSystem:
    def __init__(self, ..., enable_trajectory_logging=False):
        self.enable_trajectory_logging = enable_trajectory_logging
        self.trajectories = []
    
    def _execute_tool_call(self, tool_name, query, qid=None, log_dir=None):
        # Log retrieve queries for later RL training
        if self.enable_trajectory_logging:
            self.trajectories.append({
                'action': 'retrieve',
                'query': query,
                'timestamp': time.time()
            })
        
        # ... existing code ...
    
    def save_trajectories(self, output_path):
        """Save trajectories for offline RL training"""
        with open(output_path, 'w') as f:
            json.dump(self.trajectories, f, indent=2)
```

---

## Specific Code Files to Study

### For Deep Search Implementation:

1. **Environment Interaction Pattern**:
   - `/data/wang/junh/githubs/AgentGym/agentenv-searchqa/agentenv_searchqa/env_wrapper.py`
   - Lines 131-199: `step()` method showing action parsing and reward computation

2. **Retriever Implementation**:
   - `/data/wang/junh/githubs/AgentGym/agentenv-searchqa/agentenv_searchqa/retriever.py`
   - Lines 1-100: Encoder and BaseRetriever classes

3. **RL Training Rollout**:
   - `/data/wang/junh/githubs/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py`
   - Shows how vLLM generates sequences with environment interaction

4. **Training Configuration**:
   - `/data/wang/junh/githubs/AgentGym-RL/examples/train/AgentGym-RL/searchqa_train.sh`
   - Complete training setup for SearchQA

---

## Recommended Next Steps

### Immediate (Test Deep Search First):

1. **Setup SearchQA Environment**:
   ```bash
   cd /data/wang/junh/githubs/AgentGym/agentenv-searchqa
   conda env create -f environment.yml
   conda activate agentenv-searchqa
   pip install -e .
   bash ./setup.sh
   ```

2. **Launch SearchQA Server**:
   ```bash
   searchqa --host 0.0.0.0 --port 36005
   ```

3. **Test with Simple Example**:
   - Use AgentGym's evaluation scripts to see how SearchQA works
   - Understand the interaction pattern before adapting

### Short-term (Adapt for KARE):

1. **Create KARE Environment Prototype**:
   - Copy `agentenv-searchqa` structure
   - Replace QA logic with mortality prediction
   - Integrate your existing `KAREDataAdapter`

2. **Test without RL First**:
   - Verify environment server works correctly
   - Test retrieve queries manually
   - Validate reward computation

3. **Add RL Training**:
   - Use AgentGym-RL's training scripts as template
   - Start with small dataset (100-500 samples)
   - Monitor training stability

### Long-term (Production):

1. **Scale Up Training**:
   - Use ScalingInter-RL for progressive training
   - Experiment with different reward shaping
   - Fine-tune retrieval query generation

2. **Evaluate Improvements**:
   - Compare RL-trained retrieve queries vs. hand-crafted
   - Measure impact on mortality prediction accuracy
   - Analyze retrieved document relevance

---

## Key Takeaways

✅ **AgentGym-RL's SearchQA is highly relevant** to your retrieve query post-training goals

✅ **Your current `mortality_debate_rag.py` can be adapted** to work with AgentGym's infrastructure

✅ **Main advantage**: Automated RL training of better retrieve queries through environment interaction

✅ **Main challenge**: Need to create custom KARE mortality environment following AgentGym's API

✅ **Alternative**: Use your current system + offline RL on collected trajectories (simpler but less powerful)

---

## Questions to Consider

1. **Do you want to train retrieve query generation?** 
   - If yes → Use AgentGym-RL framework
   - If no → Keep current hand-crafted queries

2. **Training data size?**
   - Large (>10K samples) → Online RL (AgentGym-RL)
   - Small (<1K samples) → Offline RL (DPO on logged data)

3. **Infrastructure availability?**
   - Can you run environment servers? → Full AgentGym integration
   - Limited resources? → Lightweight logging approach

Let me know which direction you'd like to pursue, and I can provide more specific implementation code!
