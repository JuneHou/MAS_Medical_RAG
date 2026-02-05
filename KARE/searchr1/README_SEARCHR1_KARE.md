# Search-R1 for KARE Mortality Prediction

This document explains how Search-R1 Reinforcement Learning training works for the KARE mortality prediction task.

---

## How Search-R1 Works: Architecture Overview

### 1. Input Data: Parquet Format

**Training Data Structure** (`train.parquet`, `val.parquet`):

```python
{
    "prompt": [  # Chat format, required by Search-R1
        {
            "role": "user",
            "content": "<Patient EHR data>\n<Similar cases>\nPredict mortality risk..."
        }
    ],
    "data_source": "kare_mortality_prediction",
    "ground_truth": "0" or "1",  # 0=survival, 1=mortality
    "extra_info": {
        "patient_id": "12345_1",
        "visit_id": "1",
        "assessment_type": "mortality"
    }
}
```

**How it's processed:**
- Each training step loads **batch_size=8** samples from parquet
- Prompts are tokenized and fed to the agent
- Agent decides whether to `<search>` for evidence or `<answer>` immediately
- Ground truth is used for reward calculation (but NOT shown to the model during generation)

---

### 2. Multi-Turn Trajectory System

#### **Turn-Based Generation Loop** (`max_turns=2`)

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Initial State (8 samples active)                    │
│   Input: Patient context prompt                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Turn 1: Agent Decision                                      │
│   Model generates: <search>query</search> OR <answer>X</answer>│
│                                                              │
│   Sample 1: <search>mortality risk diabetes</search>        │
│      → Calls retriever → Gets 4 documents → Continue        │
│   Sample 2: <answer>Low risk, patient stable</answer>       │
│      → Done=True → Finished (removed from active batch)     │
│   Sample 3: <search>sepsis elderly prognosis</search>       │
│      → Calls retriever → Gets 5 documents → Continue        │
│   ...                                                        │
│                                                              │
│   Active after Turn 1: [6 samples]  (2 finished early)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Turn 2: Continue with 6 active samples                      │
│   Input: Original prompt + <information>docs</information>  │
│   Model generates: <search>...</search> OR <answer>...</answer>│
│                                                              │
│   All 6 samples: <answer>Risk assessment...</answer>        │
│      → Done=True → All finished                             │
│                                                              │
│   Active after Turn 2: [0 samples]  (all finished)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
                     Training step complete
```

#### **Key Trajectory Mechanics:**

**Active Mask Tracking:**
```python
active_mask = [True, True, True, True, True, True, True, True]  # 8 samples
# After Turn 1:
active_mask = [True, False, True, True, False, True, True, True]  # 6 active, 2 done
# After Turn 2:
active_mask = [False, False, False, False, False, False, False, False]  # All done
```

**Why samples finish at different turns:**
- **Model's decision:** Agent outputs `<answer>` tag → `done=True`
- **Invalid format:** No valid `<search>` or `<answer>` tags → gets error message, continues
- **Max turns reached:** After turn 2, all remaining samples are forced to answer

**Sequence Length Growth:**
```
Turn 1: prompt_tokens (3.5k) + response_tokens (1k) = 4.5k total
Turn 2: turn1_sequence (4.5k) + search_results (2k) + response (1k) = 7.5k total
Turn 3: turn1+2 (7.5k) + more_results (2k) + response (1k) = 10.5k total (if allowed)
```

This is why we set `max_turns=2` to prevent sequences from growing too large (>18k tokens).

---

### 3. Rollout Process (Policy Sampling)

**Rollout = Generating sequences using the current policy**

```python
# Pseudocode for one rollout step
for each_turn in range(max_turns):
    # 1. Generate responses for active samples
    responses = vllm_model.generate(
        prompts=active_samples,
        temperature=1.0,  # Stochastic sampling for exploration
        max_tokens=3072
    )
    
    # 2. Parse actions from responses
    for response in responses:
        if '<answer>' in response:
            action = 'answer'
            done = True  # Sample finished
        elif '<search>' in response:
            action = 'search'
            # Extract query and call retriever
            search_results = retriever.retrieve(query, topk=5)
            next_observation = f'<information>{search_results}</information>'
            done = False  # Continue to next turn
        else:
            action = 'invalid'
            next_observation = 'Invalid format error message'
            done = False
    
    # 3. Update active mask (remove finished samples)
    active_mask = [not done for sample in samples]
    
    # 4. Concatenate search results to prompt for next turn
    prompts = [prompt + response + observation for ...]
```

**Rollout Metrics Collected:**
- `turns_stats`: How many turns each sample took (1, 2, or max)
- `valid_action_stats`: How many valid `<search>` or `<answer>` tags
- `valid_search_stats`: How many successful searches
- `active_mask`: Which samples are still generating

---

### 4. Environment & Action Space

#### **Environment = Search-R1's custom agent loop**

Unlike typical RL environments (Atari games, robotics), Search-R1's environment is **text-based reasoning**:

| Component | Definition | Example |
|-----------|-----------|---------|
| **State** | Current prompt + conversation history | Patient EHR + previous search results |
| **Action Space** | `{search, answer}` | `<search>diabetes mortality</search>` |
| **Observation** | Search results or episode termination | Retrieved medical documents |
| **Done** | Episode termination flag | `<answer>` tag detected or max_turns reached |

#### **Action Parsing:**

```python
def postprocess_predictions(predictions):
    """Parse model output into structured actions"""
    for prediction in predictions:
        pattern = r'<(search|answer)>(.*?)</\1>'
        match = re.search(pattern, prediction)
        
        if match:
            action = match.group(1)  # 'search' or 'answer'
            content = match.group(2)  # Query text or final answer
        else:
            action = None  # Invalid format
            content = ''
    
    return actions, contents
```

#### **Environment Response:**

```python
def execute_predictions(predictions, active_mask):
    """Execute actions and return observations"""
    for action in predictions:
        if action == 'answer':
            next_obs = ''
            done = True  # Episode finished
            valid = True
            is_search = False
            
        elif action == 'search':
            # Call MedRAG retriever via HTTP
            search_results = batch_search(query, topk=5)
            next_obs = f'<information>{search_results}</information>'
            done = False  # Continue episode
            valid = True
            is_search = True
            
        else:  # Invalid format
            next_obs = 'Invalid action. Use <search>query</search> or <answer>text</answer>'
            done = False  # Give another chance
            valid = False
            is_search = False
    
    return next_obs, done, valid, is_search
```

---

### 5. Reward Function: Binary Label Match

Search-R1 uses a **sparse reward** system - you only get rewarded at the **end of the episode** (no penalties for number of search actions, invalid intermediate responses, or sequence length).

**How it works:**
- Search-R1's default reward extracts the final `<answer>...</answer>` tag
- Compares extracted answer with ground truth label ("0" or "1")  
- Returns **1.0** for exact match, **0.0** for mismatch

```python
# Search-R1's qa_em.py default behavior
def compute_score_em(solution_str, ground_truth):
    """
    Binary exact match reward.
    
    Args:
        solution_str: Full model response with <answer>X</answer> tag
        ground_truth: {'target': ['0']} or {'target': ['1']}
    
    Returns:
        1.0 if extracted answer matches ground_truth, 0.0 otherwise
    """
    # Extract last <answer>...</answer> tag
    answer = extract_solution(solution_str)
    
    if answer is None:
        return 0.0  # No valid answer tag found
    
    # Exact string match
    if em_check(answer, ground_truth['target']):
        return 1.0  # Correct prediction
    else:
        return 0.0  # Wrong prediction
```

**Example:**

| Sample | Ground Truth | Final Answer | Extracted | Reward |
|--------|--------------|--------------|-----------|--------|
| 1 | "1" (Mortality) | `<answer>1</answer>` | "1" | **1.0** ✓ |
| 2 | "0" (Survival) | `<answer>0</answer>` | "0" | **1.0** ✓ |
| 3 | "1" (Mortality) | `<answer>0</answer>` | "0" | **0.0** ✗ |
| 4 | "0" (Survival) | `<answer>survival</answer>` | "survival" | **0.0** ✗ |

**Pros:**
- ✅ Simple binary signal (easy to optimize)
- ✅ No custom reward code needed
- ✅ Works with Search-R1 out-of-box

**Cons:**
- ❌ No calibration incentive (doesn't care about probability quality)
- ❌ Binary predictions only (ignores uncertainty)
- ❌ No differentiation between near-miss vs totally wrong

---

#### **Why No Intermediate Penalties?**

Search-R1's sparse reward focuses **only on final answer quality**:

1. **Search efficiency learned implicitly:** 
   - More searches → longer sequences → slower training → natural pressure to be efficient
   - Model learns optimal search depth from reward signal alone

2. **GRPO advantage normalization handles inefficiency:**
   - Samples that search unnecessarily get lower relative advantage
   - Example: Sample A searches 3x and gets reward=0.8, Sample B answers immediately and gets reward=0.85
   - Advantage(A) = 0.8 - mean(0.8, 0.85) = -0.025 (penalized)
   - Advantage(B) = 0.85 - mean(0.8, 0.85) = +0.025 (rewarded)

3. **Valid action tracking (monitoring only):**
   ```python
   metrics['env/ratio_of_valid_action'] = 1.0  # All actions properly formatted
---

#### **What the Model Learns:**

- ✅ Predict correct outcome (0 or 1)
- ✅ Use search to improve prediction accuracy
- ✅ Format outputs correctly (`<answer>0</answer>` or `<answer>1</answer>`)

**Training signal flow:**
```
Episode → Final Answer → Extract answer from <answer> tag → Compare with ground_truth → Reward (0 or 1) → Advantage Estimation → Policy Gradient
                                              ↓
                            Higher reward for correct predictions
```

---

### 6. GRPO Training Loop

**Group Relative Policy Optimization** (GRPO) is used instead of PPO with a critic:

```python
# Simplified training step
for epoch in range(5):
    for batch in dataloader:  # 8 samples per batch
        # 1. ROLLOUT: Generate trajectories with current policy
        trajectories = rollout_manager.run_llm_loop(batch)
        # Output: {responses, actions, rewards, turns_stats, active_mask}
        
        # 2. REWARD: Compute rewards for completed episodes
        rewards = []
        for sample in trajectories:
            final_answer = sample['final_response']
            ground_truth = sample['ground_truth']
            reward = compute_reward(final_answer, ground_truth)
            rewards.append(reward)
        
        # 3. ADVANTAGE: Group-based normalization (no critic!)
        mean_reward = np.mean(rewards)
        advantages = [r - mean_reward for r in rewards]
        # Example: rewards=[0.9, 0.3, 0.7, 0.5]
        #          advantages=[0.3, -0.3, 0.1, -0.1]
        
        # 4. POLICY UPDATE: Maximize log_prob * advantage
        for ppo_epoch in range(1):  # Single PPO epoch
            old_log_probs = ref_model.compute_log_prob(trajectories)
            new_log_probs = actor_model.compute_log_prob(trajectories)
            
            # PPO clipped objective
            ratio = exp(new_log_probs - old_log_probs)
            clipped_ratio = clip(ratio, 0.8, 1.2)
            loss = -min(ratio * advantages, clipped_ratio * advantages)
            
            # KL divergence penalty (stay close to base model)
            kl_loss = kl_divergence(new_policy, base_model)
            total_loss = loss + 0.001 * kl_loss
            
            # Backward pass with gradient offloading
            total_loss.backward()
            optimizer.step()
```

**Key Differences from Standard PPO:**
- ❌ No value network (critic)
- ✅ Advantage = reward - group_mean (simpler, more stable)
- ✅ Single rollout per prompt (no n=4 sampling like typical GRPO)
- ✅ Multi-turn episodes (variable length trajectories)

---

### 7. Key Metrics Explained

From your training logs:

```python
# Step 4 metrics
{
    # Sequence lengths (tokens)
    "global_seqlen/mean": 8997.5,      # Average total sequence length
    "global_seqlen/max": 10657,        # Longest sequence this step
    "global_seqlen/min": 7237,         # Shortest sequence this step
    
    # Agent behavior
    "env/number_of_actions/mean": 2.0, # Average actions per sample (search + answer)
    "env/number_of_actions/max": 3,    # Some samples took 3 actions
    "env/number_of_actions/min": 1,    # Some answered immediately
    "env/finish_ratio": 1.0,           # All samples completed successfully
    "env/ratio_of_valid_action": 1.0,  # All actions properly formatted
    "env/number_of_valid_search": 1.0, # Average 1 search per sample
    
    # Rewards
    "critic/rewards/mean": 0.375,      # Average reward (3/8 correct?)
    "critic/rewards/max": 1.0,         # Perfect prediction
    "critic/rewards/min": 0.0,         # Wrong prediction
    
    # Policy gradient
    "actor/pg_loss": -0.241,           # Negative = maximizing reward
    "actor/kl_loss": 0.002,            # Very low = close to base model
    "actor/entropy_loss": 0.998,       # High entropy = good exploration
    
    # Advantages (GRPO)
    "critic/advantages/mean": -0.065,  # Slight negative (below group mean)
    "critic/advantages/max": 1.208,    # Best performer in batch
    "critic/advantages/min": -0.725,   # Worst performer in batch
}
```

**What these metrics tell you:**

1. **Efficient search behavior:** Average 1 search per sample (not searching excessively)
2. **Good exploration:** High entropy (0.998) means diverse outputs
3. **Stable training:** Low KL divergence (0.002) prevents forgetting base knowledge
4. **Variable performance:** Wide advantage range (-0.725 to 1.208) shows learning signal

---

## Summary: The Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load 8 samples from train.parquet                            │
│    → Each has patient context + ground_truth (hidden from model)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Multi-turn rollout (max_turns=2)                             │
│    Turn 1: Model decides <search> or <answer> for each sample   │
│    Turn 2: Active samples continue with retrieval results       │
│    → Trajectories: [8,6,0] = 2 answered turn 1, 6 in turn 2     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Compute rewards (sparse, end-of-episode)                     │
│    → Extract answer from final <answer> tag                      │
│    → Compare with ground_truth: reward 1.0 (match) or 0.0 (mismatch)│
│    → No penalties for searching or invalid actions               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. GRPO advantage estimation                                    │
│    → advantages = rewards - mean(rewards)                       │
│    → Samples above mean get positive gradient                   │
│    → Samples below mean get negative gradient                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Policy update (PPO-style)                                    │
│    → Maximize log_prob * advantage (policy gradient)            │
│    → Add KL penalty to stay close to base model                 │
│    → Use FSDP + gradient offloading for 7B model                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     Repeat for 5 epochs
```

**The model learns to:**
- 🎯 Predict mortality accurately (maximize reward)
- 🔍 Search when uncertain (implicit efficiency from advantage comparison)
- ⚡ Answer quickly when confident (shorter sequences = higher relative advantage)
- 📝 Format outputs correctly (invalid actions delay reward)

All without explicit penalties - the sparse reward + GRPO advantage normalization naturally encourages efficient, accurate behavior!

---

## Implementation (Binary Label Match)

**Status:** ✅ Current setup (Search-R1 default)

**Data:** Generated at `searchr1/data/kare_mortality_single_agent/`

**Prompt format:** Model outputs binary prediction in `<answer>` tag:
```
<answer>1</answer>  # For mortality prediction
<answer>0</answer>  # For survival prediction
```

**Training command:** Use `train_searchr1_single_agent.sh` (or `test_searchr1_training.sh` for a quick test run).

---
