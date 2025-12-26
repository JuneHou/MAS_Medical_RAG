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
│      → Calls retriever → Gets 5 documents → Continue        │
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

### 5. Reward Functions: Two Experimental Approaches

Search-R1 uses a **sparse reward** system - you only get rewarded at the **end of the episode** (no penalties for number of search actions, invalid intermediate responses, or sequence length).

---

#### **Experiment 1: Binary Label Match (Search-R1 Default)**

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

#### **Experiment 2: Probability-Based Calibration Reward (Custom)**

**Target output format:**
```
<answer>
MORTALITY PROBABILITY: 0.XX
SURVIVAL PROBABILITY: 0.YY
</answer>
```

Where `SURVIVAL = 1.0 - MORTALITY` (enforced constraint).

**Reward Design Options:**

##### **Option A: Positive-Only Reward (Recommended for Search-R1)**

Uses positive rewards scaled by prediction quality:

```python
def compute_score_prob_positive(solution_str, ground_truth):
    """
    Positive-only reward based on probability calibration.
    
    Reward formula:
    - Mortality cases (GT=1): reward = mortality_prob
    - Survival cases (GT=0): reward = survival_prob = 1 - mortality_prob
    
    Range: [0.0, 1.0]
    """
    # Extract probabilities from <answer> tag
    mort_prob = extract_mortality_probability(solution_str)
    
    if mort_prob is None:
        return 0.0  # Invalid format
    
    if not (0.0 <= mort_prob <= 1.0):
        return 0.0  # Out of range
    
    # Ground truth from parquet
    gt_label = int(ground_truth['target'][0])  # 0 or 1
    
    if gt_label == 1:  # Mortality case
        # Reward high mortality predictions
        reward = mort_prob
    else:  # Survival case (gt_label == 0)
        # Reward high survival predictions
        reward = 1.0 - mort_prob
    
    return reward  # Range: [0.0, 1.0]
```

**Example:**

| Sample | GT | Final Answer | Mort Prob | Surv Prob | Reward |
|--------|----|--------------|-----------|-----------|----|
| 1 | 1 (Mortality) | MORT: 0.85, SURV: 0.15 | 0.85 | 0.15 | **0.85** ✓ |
| 2 | 0 (Survival) | MORT: 0.20, SURV: 0.80 | 0.20 | 0.80 | **0.80** ✓ |
| 3 | 1 (Mortality) | MORT: 0.30, SURV: 0.70 | 0.30 | 0.70 | **0.30** ✗ (low) |
| 4 | 0 (Survival) | MORT: 0.75, SURV: 0.25 | 0.75 | 0.25 | **0.25** ✗ (low) |

**Pros:**
- ✅ Incentivizes probability calibration
- ✅ Smooth gradient (easier optimization)
- ✅ No negative rewards (stable GRPO training)

**Cons:**
- ❌ No strong penalty for wrong direction (mort=0.55 on survival still gets 0.45 reward)

---

##### **Option B: Symmetric ±1/0/-1 Reward (Stronger Signal)**

Uses penalties for wrong predictions and neutral zone for uncertainty:

```python
def compute_score_prob_symmetric(solution_str, ground_truth):
    """
    Symmetric reward with penalties for wrong predictions.
    
    Thresholds:
    - High confidence: mort < 0.4 or mort ≥ 0.7
    - Uncertain: mort ∈ [0.4, 0.7)
    
    Returns: +1 (correct direction), 0 (uncertain), -1 (wrong direction)
    """
    mort_prob = extract_mortality_probability(solution_str)
    
    if mort_prob is None or not (0.0 <= mort_prob <= 1.0):
        return -1.0  # Invalid format gets penalty
    
    gt_label = int(ground_truth['target'][0])
    
    if gt_label == 1:  # Mortality case
        if mort_prob >= 0.7:
            return +1.0  # Correctly high mortality
        elif mort_prob >= 0.4:
            return 0.0   # Uncertain
        else:
            return -1.0  # Incorrectly low mortality
    
    else:  # Survival case (gt_label == 0)
        if mort_prob < 0.4:
            return +1.0  # Correctly low mortality
        elif mort_prob < 0.7:
            return 0.0   # Uncertain
        else:
            return -1.0  # Incorrectly high mortality
```

**Example:**

| Sample | GT | Mort Prob | Reward | Reason |
|--------|----|-----------|----|--------|
| 1 | 1 (Mortality) | 0.85 | **+1.0** ✓ | High mort for mort case |
| 2 | 0 (Survival) | 0.20 | **+1.0** ✓ | Low mort for surv case |
| 3 | 1 (Mortality) | 0.55 | **0.0** ~ | Uncertain (0.4-0.7 range) |
| 4 | 0 (Survival) | 0.75 | **-1.0** ✗ | High mort for surv case |
| 5 | 1 (Mortality) | 0.30 | **-1.0** ✗ | Low mort for mort case |

**Pros:**
- ✅ Strong penalty for wrong direction (faster learning)
- ✅ Neutral zone prevents over-confidence on borderline cases
- ✅ Clear separation between good/uncertain/bad predictions

**Cons:**
- ❌ Negative rewards may slow GRPO convergence initially
- ❌ Harder thresholds (0.4/0.7 may need tuning)

---

#### **Recommendation:**

**Start with Option A (Positive-Only)** for initial training:
- Simpler optimization landscape
- Proven to work in Search-R1's GRPO framework
- Smooth gradients for calibration learning

**Switch to Option B (±1/0/-1)** if you observe:
- Model predicting wrong direction frequently (e.g., mort=0.6 on survival cases)
- Need stronger differentiation between good/bad predictions
- Reward variance too low (all predictions clustered around 0.5)

Both options work with GRPO's advantage normalization - the choice depends on how aggressive you want the learning signal to be.

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

#### **What the Model Learns Across Both Experiments:**

**Experiment 1 (Binary Label):**
- ✅ Predict correct outcome (0 or 1)
- ✅ Use search to improve prediction accuracy
- ❌ No calibration (doesn't learn probability quality)

**Experiment 2 (Probability-Based):**
- ✅ Predict correct outcome direction
- ✅ Calibrate probabilities to reflect true risk
- ✅ Express uncertainty (via 0.4-0.7 range in Option B)
- ✅ Learn when to search for better probability estimates

**Training signal flow:**
```
Episode → Final Answer → Extract Probabilities → Compute Reward → Advantage Estimation → Policy Gradient
                                                        ↓
                                    Higher reward for accurate, calibrated
```
Episode → Final Answer → Reward (0-1) → Advantage Estimation → Policy Gradient
                                              ↓
                            Higher reward for efficient, accurate predictions
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
│    → Extract mortality probability from final answers           │
│    → Compare with ground_truth: reward ∈ [0, 1]                 │
│    → No penalties for searching or invalid actions              │
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

## Implementing the Two Experiments

### Experiment 1: Binary Label Match (Current Setup)

**Status:** ✅ Ready to use (Search-R1 default)

**Data:** Already generated at `searchr1/data/kare_mortality_single_agent/`

**Prompt format:** Model outputs binary prediction in `<answer>` tag
```
<answer>1</answer>  # For mortality prediction
<answer>0</answer>  # For survival prediction
```

**Training command:** Use existing `train_searchr1_single_agent.sh` as-is

---

### Experiment 2: Probability-Based Calibration (Requires Custom Reward)

**Status:** 🔧 Requires implementation

#### **Step 1: Update Prompt to Request Both Probabilities**

Modify [prepare_searchr1_balanced_data.py](data_generation/prepare_searchr1_balanced_data.py):

```python
def create_single_agent_prompt(sample: Dict) -> str:
    prompt = f"""You are a medical AI assistant specialized in mortality prediction.
Your task is to assess patient mortality risk and provide calibrated probability estimates.

You must conduct reasoning inside <think> and </think> tags every time you analyze information.
If you need additional medical evidence, search for relevant clinical information by writing <search>query</search>.
The search engine will return medical literature between <information> and </information> tags.
You can search multiple times to gather comprehensive evidence.

When confident in your assessment, provide your final answer with BOTH probabilities:
<answer>
MORTALITY PROBABILITY: 0.XX
SURVIVAL PROBABILITY: 0.YY
</answer>

Note: The two probabilities must sum to 1.0 (SURVIVAL = 1.0 - MORTALITY).

## Target Patient
{target_context}

## Similar Patient Cases
...
"""
```

#### **Step 2: Create Custom Reward Function**

Create `searchr1/reward_functions/kare_mortality_probability.py`:

```python
import re

# Regex to extract probabilities from <answer> tag
MORT_PROB_RE = re.compile(r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
SURV_PROB_RE = re.compile(r'SURVIVAL PROBABILITY:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)

def extract_probabilities(solution_str):
    """Extract mortality and survival probabilities from answer."""
    mort_match = MORT_PROB_RE.search(solution_str)
    surv_match = SURV_PROB_RE.search(solution_str)
    
    mort_prob = float(mort_match.group(1)) if mort_match else None
    surv_prob = float(surv_match.group(1)) if surv_match else None
    
    return mort_prob, surv_prob

def compute_score_prob_positive(solution_str, ground_truth, **kwargs):
    """
    Option A: Positive-only reward (recommended).
    
    Reward = mortality_prob if GT=1, else 1-mortality_prob
    Range: [0.0, 1.0]
    """
    mort_prob, surv_prob = extract_probabilities(solution_str)
    
    # Validation
    if mort_prob is None:
        return 0.0  # No mortality probability found
    
    if not (0.0 <= mort_prob <= 1.0):
        return 0.0  # Invalid range
    
    # Optional: Check if probabilities sum to 1.0 (with tolerance)
    if surv_prob is not None:
        if abs((mort_prob + surv_prob) - 1.0) > 0.05:
            return 0.0  # Probabilities don't sum to 1.0
    
    # Extract ground truth label
    gt_label = int(ground_truth['target'][0])  # 0 or 1
    
    # Compute reward
    if gt_label == 1:  # Mortality case
        reward = mort_prob
    else:  # Survival case
        reward = 1.0 - mort_prob
    
    return reward

def compute_score_prob_symmetric(solution_str, ground_truth, **kwargs):
    """
    Option B: Symmetric ±1/0/-1 reward.
    
    Thresholds: mort < 0.4 (low), [0.4, 0.7) (uncertain), ≥ 0.7 (high)
    Returns: +1 (correct), 0 (uncertain), -1 (wrong)
    """
    mort_prob, surv_prob = extract_probabilities(solution_str)
    
    # Validation
    if mort_prob is None or not (0.0 <= mort_prob <= 1.0):
        return -1.0  # Invalid format gets penalty
    
    if surv_prob is not None:
        if abs((mort_prob + surv_prob) - 1.0) > 0.05:
            return -1.0  # Probabilities don't sum to 1.0
    
    gt_label = int(ground_truth['target'][0])
    
    if gt_label == 1:  # Mortality case
        if mort_prob >= 0.7:
            return +1.0  # Correctly high
        elif mort_prob >= 0.4:
            return 0.0   # Uncertain
        else:
            return -1.0  # Incorrectly low
    else:  # Survival case
        if mort_prob < 0.4:
            return +1.0  # Correctly low
        elif mort_prob < 0.7:
            return 0.0   # Uncertain
        else:
            return -1.0  # Incorrectly high

# Default export (choose which to use)
compute_score = compute_score_prob_positive  # or compute_score_prob_symmetric
```

#### **Step 3: Register Custom Reward in Search-R1**

Add to Search-R1's reward registry (`Search-R1/verl/utils/reward_score/__init__.py`):

```python
from searchr1.reward_functions.kare_mortality_probability import compute_score as kare_prob_reward

REWARD_REGISTRY = {
    # ... existing rewards ...
    'kare_mortality_prob': kare_prob_reward,
}
```

Or use custom reward path in training script:

```bash
# In train_searchr1_single_agent.sh, add:
+custom_reward_function.path="/path/to/kare_mortality_probability.py" \
+custom_reward_function.name="compute_score"
```

#### **Step 4: Regenerate Data with New Prompt**

```bash
cd /data/wang/junh/githubs/Debate/KARE

# Update prompt in prepare_searchr1_balanced_data.py (Step 1)
# Then regenerate data
python searchr1/data_generation/prepare_searchr1_balanced_data.py \
    --balanced_json searchr1/data_generation/train_balanced_100pos_100neg.json \
    --split train \
    --output_dir searchr1/data/kare_mortality_prob

python searchr1/data_generation/prepare_searchr1_balanced_data.py \
    --balanced_json searchr1/data_generation/val_balanced_25pos_25neg.json \
    --split val \
    --output_dir searchr1/data/kare_mortality_prob
```

#### **Step 5: Update Training Script**

Modify `train_searchr1_single_agent.sh`:

```bash
# Change data path
export DATA_DIR='/path/to/searchr1/data/kare_mortality_prob'
export EXPERIMENT_NAME='searchr1-kare-mortality-prob'

# Add custom reward (if not registered in __init__.py)
# ... in python3 -m verl.trainer.main_ppo call:
    +custom_reward_function.path="searchr1/reward_functions/kare_mortality_probability.py" \
    +custom_reward_function.name="compute_score"
```

#### **Step 6: Train**

```bash
bash searchr1/train_searchr1_single_agent.sh
```

---

### Comparing the Two Experiments

| Aspect | Experiment 1 (Binary) | Experiment 2 (Probability) |
|--------|-----------------------|----------------------------|
| **Output Format** | `<answer>0</answer>` or `<answer>1</answer>` | `MORTALITY PROBABILITY: 0.XX`<br>`SURVIVAL PROBABILITY: 0.YY` |
| **Reward Range** | {0.0, 1.0} (binary) | **Option A:** [0.0, 1.0] (continuous)<br>**Option B:** {-1.0, 0.0, +1.0} |
| **Training Signal** | Sparse (only correct/wrong) | Dense (quality of probability) |
| **Calibration** | None | ✓ Learns probability quality |
| **Uncertainty** | Cannot express | ✓ Can express via probabilities |
| **Implementation** | ✓ Works out-of-box | Requires custom reward function |

---
