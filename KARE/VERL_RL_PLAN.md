# VERL RL Training Plan for KARE Integrator Format Enforcement

## Executive Summary

This document provides a **revised and evaluated plan** for using VERL (Volcano Engine Reinforcement Learning) to enforce output format compliance in the KARE mortality prediction debate system's integrator agent. The goal is to use RL to train the integrator model to reliably output probability values in the expected format.

---

## 1. Problem Statement

### Current Issue
The integrator agent in `mortality_debate_rag.py` frequently fails to output mortality and survival probabilities in the expected format:
- **Expected format**: `MORTALITY PROBABILITY: 0.XX` and `SURVIVAL PROBABILITY: 0.XX`
- **Current behavior**: Model often generates verbose explanations without structured probability outputs
- **Impact**: 
  - Complex regex patterns needed to extract probabilities (15+ patterns in `_extract_prediction_and_probabilities`)
  - Frequent parsing failures leading to `None` predictions
  - Retry logic required (up to 2 retries per integrator call)

### Target Solution
Use VERL's GRPO (Group Relative Policy Optimization) to train the integrator model with **binary format rewards**:
- **Reward = 1**: Output contains valid probability in correct format and valid range [0.0, 1.0]
- **Reward = 0**: Output missing probability or invalid format

### Key Clarifications
- **Scope**: RL training focuses **ONLY on format compliance**, not reasoning quality or retrieval accuracy
- **Independence**: Mortality and survival probabilities are **independent predictions** (not required to sum to 1.0)
- **Dataset**: 996 patients available in test set (sufficient for RL training data generation)

---

## 2. VERL Framework Understanding

### 2.1 What is VERL?
VERL is ByteDance's open-source RL training framework for LLMs with key features:
- **Flexible RL algorithms**: PPO, GRPO, RLOO, ReMax, etc.
- **Integration with existing infra**: FSDP, Megatron-LM, vLLM, SGLang
- **Efficient device mapping**: Supports various GPU placement strategies
- **Custom reward functions**: Easy integration of task-specific rewards

### 2.2 Why GRPO for This Task?

**GRPO (Group Relative Policy Optimization)** is the recommended approach because:

1. **No critic model required**: Eliminates computational overhead
2. **Group sampling for exploration**: Generates multiple completions per prompt, helping discover correct format
3. **Relative rewards**: Compares outputs within a group, natural for binary format compliance
4. **Proven for format tasks**: Similar to GSM8K format enforcement in VERL examples

**Key GRPO Mechanism**:
```
For each prompt:
  1. Sample N completions (e.g., n=4)
  2. Score each completion (0 or 1 for format compliance)
  3. Use group average as baseline
  4. Update policy to favor above-average completions
```

### 2.3 VERL Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VERL Training Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Data Loading (Parquet)                                   │
│     └── prompt, data_source, ground_truth, extra_info       │
│                                                               │
│  2. Rollout Generation (vLLM/SGLang)                        │
│     └── Generate N completions per prompt                    │
│                                                               │
│  3. Reward Computation (Custom Function)                     │
│     └── compute_score(data_source, solution, gt, extra)     │
│                                                               │
│  4. Advantage Estimation (GRPO)                              │
│     └── Group-based normalization                            │
│                                                               │
│  5. Policy Update (FSDP/Megatron)                           │
│     └── Update actor model with PPO-style clipping          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Evaluation of Original Plan

### ✅ Strengths of the Original Plan

1. **Correct algorithm choice**: GRPO is appropriate for format enforcement
2. **Binary reward structure**: Simple 1/0 reward matches the task well
3. **Custom reward function**: Properly structured with regex validation
4. **Two separate tasks**: Mortality and survival probability tasks correctly separated
5. **Sampling strategy**: Recognizes need for exploration with temperature/sampling

### ⚠️ Issues and Concerns

#### 3.1 Data Preparation Gaps

**Issue**: Plan lacks details on how to construct training prompts from current debate system

**Current System**:
- Integrator receives: patient context + similar patients + debate history + medical knowledge
- Two-step process: separate mortality and survival assessments
- Context can be 10,000+ tokens long

**Concerns**:
- How to fit long contexts (10k+ tokens) into VERL's training pipeline?
- Should we use full debate history or summarized version for training efficiency?
- Context truncation strategy needed (keep most relevant information)

#### 3.2 Reward Function Considerations

**Proposed reward function structure**:

```python
# Proposed reward (format compliance only)
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "kare_integrator_mortality_format":
        m = MORT_RE.search(solution_str or "")
        if not m:
            return 0.0
        p = float(m.group(1))
        return 1.0 if 0.0 <= p <= 1.0 else 0.0
```

**Design considerations**:
- ✅ **Pure format validation**: Only checks probability format and range [0.0, 1.0]
- ✅ **Independence**: No constraint that mortality + survival = 1.0 (they are independent)

#### 3.3 Integration Strategy Considerations

**Issue**: How to integrate RL-trained model back into two-step integrator workflow?

**Current workflow**:
```python
# Step 1: Mortality assessment (with optional tool calling)
mortality_response = integrator_model(mortality_prompt + retrieve_tool)

# Step 2: Survival assessment (with optional tool calling)
survival_response = integrator_model(survival_prompt + retrieve_tool)
```

**Decisions needed**:
- **Training data**: Should prompts include tool call examples or train without tools?
  - **Recommendation**: Train with tool calls included in context (matches inference)
- **Model architecture**: Full fine-tuning vs LoRA adapter?
  - **Recommendation**: LoRA for safety (less catastrophic forgetting risk)
- **Tool preservation**: Model already knows tool calling from pretraining
  - **No special handling needed**: Format enforcement won't break tool capability

#### 3.4 Training Data Sourcing

**Available resources**: 996 patients in test set

**Options**:
1. **Use test set directly** (996 patients): 
   - ✅ Sufficient size for RL training
   - ✅ Real integrator contexts from actual debates
   - ⚠️ Need to run debate system to generate contexts (2-3 GPU days)
   - **Recommended approach**

2. **Synthetic prompt generation**: Generate prompts without running full debate
   - ✅ Fast generation (no debate needed)
   - ❌ May not match real distribution
   - **Fallback option for initial testing**

3. **Use subset (500 samples) for efficiency**:
   - ✅ Faster data generation (1-1.5 GPU days)
   - ✅ Still sufficient for GRPO training
   - ✅ Can expand later if needed
   - **Best balance of cost/benefit**

**Recommendation**: Start with 500 test samples, expand to full 996 if format compliance <95%

#### 3.5 Evaluation Metrics

**Evaluation scope**: Focus on format compliance and task performance

**Required metrics**:
1. **Format compliance** (Primary): % of outputs with valid probability format
   - Target: ≥95% on validation set
   - Critical success metric

2. **Prediction accuracy** (Primary): Does enforcing format hurt correctness?
   - Compare RL model accuracy vs baseline
   - Must not degrade

3. **Inference speed** (Secondary): Generation time per sample
   - Measure: tokens/second
   - Target: ≤10% slowdown

4. **Response length** (Secondary): Average token count per response
   - Check if RL causes verbosity or over-truncation

**Out of scope** (not objectives of this RL training):
- ❌ Reasoning quality/explanation depth
- ❌ Retrieval accuracy/relevance  
- ❌ Probability calibration (nice-to-have, not required)

---

## 4. Revised Implementation Plan

### Phase 1: Data Preparation (Week 1-2)

#### Step 1.1: Create Training Dataset

**Primary Approach: Run Debate on Test Set (Recommended)**
```bash
# Generate training data from 500 test set samples
python generate_rl_training_data.py \
    --source test_set \
    --num_samples 500 \
    --output_dir data/rl_training/raw_contexts/ \
    --save_integrator_prompts True \
    --gpus "6,7"

# Expected output:
# - 500 mortality prompts (one per patient)
# - 500 survival prompts (one per patient)  
# - Total: 1000 training examples
# - Estimated time: 1-1.5 GPU days
```

**Fallback: Synthetic Prompts (For quick testing only)**
```bash
# Create synthetic integrator prompts without running debate
python create_synthetic_prompts.py \
    --num_samples 200 \
    --template_type both \
    --output_dir data/rl_training_synthetic/
    
# Use for initial debugging, not final training
```

**Dataset Structure** (Parquet format):
```python
{
    'prompt': str,  # Full integrator prompt (with context, debate history, tool description)
    'data_source': str,  # 'kare_integrator_mortality_format' or 'kare_integrator_survival_format'
    'ground_truth': str,  # Ground truth outcome (0 or 1), for optional accuracy reward
    'extra_info': dict,  # {
        'patient_id': str,
        'context_length': int,
        'has_positive_similars': bool,
        'has_negative_similars': bool
    }
}
```

**Key decisions**:
- [x] **Context length**: Use full context (8-12k tokens) - models can handle it
  - Qwen3-4B supports up to 32k tokens, no need for aggressive truncation
  - Only truncate if exceeding model limits
  
- [x] **Tool calling**: Include tool description in prompts
  - Train with tool calling context (matches inference scenario)
  - Model preserves tool capability from pretraining
  
- [x] **Train/Val split**: 80/20 split with stratification by outcome
  - 400 train / 100 val samples
  - Stratify to maintain mortality rate balance

#### Step 1.2: Preprocess Data Script

```python
# File: examples/data_preprocess/kare_integrator_format.py

def preprocess_kare_integrator_data(
    input_dir: str,
    output_dir: str,
    max_prompt_length: int = 12000,  # Allow long contexts (Qwen3 supports 32k)
    train_split: float = 0.8
):
    """
    Convert KARE debate contexts to VERL parquet format.
    
    Args:
        input_dir: Directory with raw integrator contexts (from debate runs)
        output_dir: Output directory for parquet files
        max_prompt_length: Maximum prompt length in tokens (12k is safe for Qwen3-4B)
        train_split: Fraction for training (0.8 = 80% train, 20% val)
    """
    # Load KARE contexts (mortality and survival prompts for each patient)
    contexts = load_kare_contexts(input_dir)
    
    processed_data = []
    for ctx in contexts:
        # Keep full context (only truncate if exceeding max_prompt_length)
        prompt = ctx['integrator_prompt']  # Already formatted by debate system
        
        # Truncate only if necessary
        if len(prompt.split()) > max_prompt_length:
            prompt = truncate_prompt_intelligently(prompt, max_prompt_length)
        
        # Determine data source based on task type
        data_source = (
            "kare_integrator_mortality_format"
            if ctx['task'] == 'mortality'
            else "kare_integrator_survival_format"
        )
        
        processed_data.append({
            'data_source': data_source,
            'prompt': prompt,
            'ability': 'medical_format',
            'reward_model': {'style': 'rule'},
            'extra_info': {
                'patient_id': ctx['patient_id'],
                'ground_truth': str(ctx['label']),
                'context_length': len(prompt.split())
            }
        })
    
    # Shuffle and split into train/val with stratification
    dataset = Dataset.from_list(processed_data)
    dataset = dataset.shuffle(seed=42)
    
    # Split by data_source to maintain balance
    train_size = int(len(dataset) * train_split)
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    # Save to parquet
    train_dataset.to_parquet(os.path.join(output_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(output_dir, 'val.parquet'))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
```

### Phase 2: Custom Reward Function (Week 2)

#### Step 2.1: Enhanced Reward Function

```python
# File: kare_format_reward.py

import re
from typing import Dict, Any, Optional

# Regex patterns matching your current extraction logic
MORT_RE = re.compile(
    r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)
SURV_RE = re.compile(
    r'SURVIVAL PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)

# Response length penalties (optional - can disable if not concerned about efficiency)
MAX_REASONABLE_LENGTH = 1024  # tokens - allow longer medical reasoning
MIN_REASONABLE_LENGTH = 20    # tokens - just needs probability statement

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Binary format reward for KARE integrator outputs.
    
    ONLY checks format compliance:
      - Probability value found in correct format
      - Value in valid range [0.0, 1.0]
    
    Does NOT check:
      - Reasoning quality
      - Retrieval accuracy
      - Probability calibration
    
    Reward = 1 if format correct, 0 otherwise
    """
    
    # Basic validation
    if not solution_str:
        return 0.0
    
    # Optional: Length penalties for efficiency (can disable by commenting out)
    # response_length = len(solution_str.split())
    # if response_length > MAX_REASONABLE_LENGTH:
    #     print(f"[REWARD] Response too long: {response_length} tokens")
    #     return 0.0
    # if response_length < MIN_REASONABLE_LENGTH:
    #     print(f"[REWARD] Response too short: {response_length} tokens")
    #     return 0.0
    
    # Task-specific validation
    if data_source == "kare_integrator_mortality_format":
        match = MORT_RE.search(solution_str)
        if not match:
            return 0.0
        
        try:
            prob = float(match.group(1))
        except ValueError:
            return 0.0
        
        # Check valid range
        if not (0.0 <= prob <= 1.0):
            return 0.0
        
        # Success!
        return 1.0
    
    elif data_source == "kare_integrator_survival_format":
        match = SURV_RE.search(solution_str)
        if not match:
            return 0.0
        
        try:
            prob = float(match.group(1))
        except ValueError:
            return 0.0
        
        # Check valid range
        if not (0.0 <= prob <= 1.0):
            return 0.0
        
        # Success!
        return 1.0
    
    # Unknown data source
    print(f"[REWARD] Unknown data_source: {data_source}")
    return 0.0


# Alternative: Multi-component reward (more sophisticated)
def compute_score_with_accuracy(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Enhanced reward with format + accuracy components.
    
    Returns dict with:
      - score: total reward
      - format_score: binary format compliance
      - accuracy: prediction accuracy (if applicable)
    """
    
    # Format reward (binary)
    format_score = compute_score(data_source, solution_str, ground_truth, extra_info)
    
    if format_score == 0.0:
        # No format compliance, return 0 total
        return {'score': 0.0, 'format_score': 0.0, 'accuracy': 0.0}
    
    # If format is correct, optionally add accuracy reward
    # (only if ground_truth is available and we want to train for correctness)
    if ground_truth and extra_info:
        # Extract probability from solution
        if data_source == "kare_integrator_mortality_format":
            match = MORT_RE.search(solution_str)
            prob = float(match.group(1))
            
            # Simple accuracy: correct direction (high prob for mortality=1, low for mortality=0)
            gt_label = int(ground_truth)
            if (prob > 0.5 and gt_label == 1) or (prob < 0.5 and gt_label == 0):
                accuracy = 1.0
            else:
                accuracy = 0.0
        else:
            # Similar logic for survival
            match = SURV_RE.search(solution_str)
            prob = float(match.group(1))
            gt_label = int(ground_truth)
            
            # Survival probability should be high when mortality=0
            if (prob > 0.5 and gt_label == 0) or (prob < 0.5 and gt_label == 1):
                accuracy = 1.0
            else:
                accuracy = 0.0
        
        # Weighted combination
        total_score = 0.7 * format_score + 0.3 * accuracy
        return {
            'score': total_score,
            'format_score': format_score,
            'accuracy': accuracy
        }
    
    # No ground truth, use only format score
    return {'score': format_score, 'format_score': format_score, 'accuracy': 0.0}
```

**Key design decisions**:
- [x] **Pure format only**: Only validate format compliance, not reasoning or accuracy
  - This is the core objective - enforce format output
  - Accuracy/reasoning preserved through base model knowledge
  
- [ ] **Length penalties**: Optional - include only if concerned about inference cost
  - Disabled by default (reasoning may need space)
  - Enable if model generates excessively long responses during training
  
- [x] **Strict regex**: Use exact patterns from `_extract_prediction_and_probabilities`
  - Match current extraction logic for consistency
  - Patterns: `MORTALITY PROBABILITY: X.XX` and `SURVIVAL PROBABILITY: X.XX`

### Phase 3: GRPO Training Configuration (Week 3)

#### Step 3.1: Training Script

```bash
# File: run_kare_integrator_grpo.sh

#!/bin/bash

# Model configuration
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"  # Same as current integrator
BASE_MODEL=$MODEL_PATH

# Data paths
TRAIN_DATA="/data/wang/junh/githubs/Debate/KARE/data/rl_training/train.parquet"
VAL_DATA="/data/wang/junh/githubs/Debate/KARE/data/rl_training/val.parquet"

# Output
OUTPUT_DIR="/data/wang/junh/githubs/Debate/KARE/rl_checkpoints"

# GPU configuration
export CUDA_VISIBLE_DEVICES="6,7"  # Match your current setup
NPROC_PER_NODE=2

# Training with GRPO
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout.ref.rollout.n=4 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    \
    custom_reward_function.path=/data/wang/junh/githubs/Debate/KARE/kare_format_reward.py \
    custom_reward_function.name=compute_score \
    \
    trainer.logger=['wandb'] \
    trainer.project_name='kare_integrator_format_rl' \
    trainer.experiment_name='grpo_mortality_survival_v1' \
    trainer.total_epochs=3 \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.default_hdfs_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NPROC_PER_NODE
```

**Key hyperparameters**:
- **rollout.n=4**: Sample 4 completions per prompt for GRPO
- **temperature=0.7**: Enable exploration to discover correct format
- **lr=1e-6**: Very small learning rate to preserve pretrained knowledge
- **kl_loss_coef=0.01**: KL regularization to prevent catastrophic forgetting
- **ppo_epochs=2**: Multiple passes over sampled data

#### Step 3.2: Alternative: LoRA Training (Resource-efficient)

If full fine-tuning is too expensive:

```bash
# Add LoRA configuration
actor_rollout_ref.actor.use_lora=True \
actor_rollout_ref.actor.lora_rank=16 \
actor_rollout_ref.actor.lora_alpha=32 \
actor_rollout_ref.actor.lora_dropout=0.05 \
actor_rollout_ref.actor.lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'
```

**Trade-offs**:
- ✅ Faster training, less memory
- ✅ Easier to merge/swap adapters
- ⚠️ May have lower capacity to learn format
- ⚠️ Need to test if LoRA is sufficient for this task

### Phase 4: Integration Back to Debate System (Week 4)

#### Step 4.1: Load RL-Trained Model

Modify `mortality_debate_rag.py`:

```python
class MortalityDebateSystem:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 integrator_model_name: str = None,
                 integrator_rl_checkpoint: str = None,  # NEW
                 ...):
        
        # Initialize main agents (1-3) as before
        self.llm = VLLMWrapper(model_name=model_name)
        
        # Initialize integrator with RL checkpoint if provided
        if integrator_rl_checkpoint:
            print(f"Loading RL-trained integrator from: {integrator_rl_checkpoint}")
            # Load RL checkpoint (may be full model or LoRA adapter)
            self.integrator_llm = VLLMWrapper(
                model_name=integrator_model_name or model_name,
                model_path=integrator_rl_checkpoint,  # Override with RL checkpoint
                enable_thinking=True
            )
        else:
            # Use base model
            self.integrator_llm = VLLMWrapper(
                model_name=integrator_model_name or model_name
            )
```

#### Step 4.2: Update Run Script

```python
# run_kare_debate_mortality.py

parser.add_argument(
    '--integrator_rl_checkpoint',
    type=str,
    default=None,
    help='Path to RL-trained integrator checkpoint'
)

# In run_kare_debate_evaluation()
if debate_mode.lower() == "rag":
    debate_system = MortalityDebateRAG(
        model_name=model_name,
        integrator_model_name=integrator_model_name,
        integrator_rl_checkpoint=args.integrator_rl_checkpoint,  # NEW
        ...
    )
```

#### Step 4.3: Validation Testing

```bash
# Test RL-trained integrator on validation set
python run_kare_debate_mortality.py \
    --start_idx 0 \
    --num_samples 50 \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --integrator_rl_checkpoint "./rl_checkpoints/best_checkpoint" \
    --output "./results/rl_integrator_validation.json"
```

**Metrics to track**:
1. **Format compliance rate** (Primary goal): Should reach ≥95%
2. **Prediction accuracy** (Critical): Must not degrade vs baseline
3. **Inference speed**: Should be comparable (≤10% slower)
4. **Average response length**: Monitor for efficiency (not a constraint)

---

## 5. Concrete Next Steps

### Immediate Actions (This Week)

1. **Create data generation script** (`generate_rl_training_data.py`)
   - Run debate system on 500 test set samples (from 996 available)
   - Save integrator prompts (mortality + survival) to disk
   - Expected: 1000 training examples (500 patients × 2 tasks)
   - Estimated time: 1-1.5 GPU days on your hardware

2. **Implement preprocessing script** (`examples/data_preprocess/kare_integrator_format.py`)
   - Convert raw prompts to parquet format
   - Create 80/20 train/val split with stratification
   - Validate data structure and token lengths

3. **Implement reward function** (`kare_format_reward.py`)
   - Pure format compliance checking (binary 0/1)
   - Match existing regex patterns from extraction code
   - Test on sample outputs from current system

### Week 2-3: Training

4. **Run initial GRPO training**
   - Start with small dataset (100 samples) for debugging
   - Monitor format compliance on validation set
   - Tune hyperparameters if needed

5. **Scale up training**
   - Full dataset (500+ samples)
   - Train for 3-5 epochs
   - Save best checkpoint based on format compliance

### Week 4: Integration and Evaluation

6. **Integrate RL model into debate system**
   - Update initialization code
   - Test on validation set
   - Compare with baseline

7. **Comprehensive evaluation**
   - Format compliance rate
   - Prediction accuracy
   - Error analysis on remaining failures
   - Inference speed benchmarking

---

## 6. Risk Assessment and Mitigation

### Risk 1: RL Training Instability
**Likelihood**: Medium  
**Impact**: High  
**Mitigation**:
- Use very small learning rate (1e-6)
- Strong KL regularization (0.01-0.05)
- Save frequent checkpoints
- Monitor validation metrics every 100 steps

### Risk 2: Catastrophic Forgetting
**Likelihood**: Medium  
**Impact**: High  
**Mitigation**:
- Use LoRA instead of full fine-tuning (primary defense)
- Strong KL penalty (0.01-0.05) to stay close to reference policy
- Keep RL training short (2-3 epochs max)
- Monitor prediction accuracy during training (must not degrade)

### Risk 3: Insufficient Training Data
**Likelihood**: Medium  
**Impact**: Medium  
**Mitigation**:
- Start with 500 samples, expand if needed
- Use data augmentation (paraphrase prompts)
- Consider synthetic prompt generation
- Use group sampling (n=4) for more diversity

### Risk 4: Format Enforcement Degrading Task Performance
**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**:
- Monitor prediction accuracy on validation set during training
- Use strong KL penalty to preserve base model knowledge
- Test on real mortality prediction task after training
- Keep baseline model for comparison and fallback

---

## 7. Success Criteria

### Primary Goals (Required for Success)
- ✅ **Format compliance**: ≥95% of outputs have valid probability format
  - Core objective of RL training
  - Measured on validation set
  
- ✅ **No accuracy degradation**: Prediction accuracy ≥ baseline model
  - Must preserve task performance
  - Measured on mortality prediction task
  
- ✅ **Inference speed**: ≤10% slowdown vs baseline
  - Ensure practical usability

### Secondary Goals (Nice-to-have)
- 🎯 **Reduced retries**: Eliminate need for retry logic in production
- 🎯 **Response efficiency**: Shorter responses without losing format compliance

### Out of Scope (Not objectives)
- ❌ Improving reasoning quality (preserved through base model)
- ❌ Improving calibration (not the focus)
- ❌ Improving retrieval quality (not related to format)

---

## 8. Estimated Timeline and Resources

### Timeline
- **Week 1**: Data generation (run debate on 500 test samples)
  - Compute time: 1-1.5 GPU days
  - Human time: 4-6 hours (setup + monitoring)
  
- **Week 2**: Preprocessing + reward function + initial training
  - Preprocessing: 2-3 hours
  - Reward function: 2-3 hours
  - Training: 0.5-1 GPU day
  
- **Week 3**: Full GRPO training and validation
  - Training compute: 1-2 GPU days
  - Hyperparameter tuning: 0.5 GPU day
  - Human time: 6-8 hours (monitoring + analysis)
  
- **Week 4**: Integration and evaluation
  - Integration: 3-4 hours
  - Evaluation: 4-6 hours
  
- **Total**: 4 weeks calendar time, ~3-4 GPU days compute time

### Compute Resources
- **Training**: 2x A100 80GB (GPUs 6,7) - your current setup
- **Data generation**: Same GPUs (overnight runs)
- **Estimated cost**: $0 (using existing infrastructure)

---

## 9. Alternative Approaches (If VERL Fails)

### Backup Plan A: Constrained Decoding
Instead of RL, use vLLM's constrained decoding:
```python
# Force model to output in specific format using grammar
from vllm import SamplingParams

sampling_params = SamplingParams(
    guided_regex=r"MORTALITY PROBABILITY: [0-9]\.[0-9]{2}"
)
```
**Pros**: No training needed, immediate results  
**Cons**: May generate nonsensical probabilities, less flexible

### Backup Plan B: Prompt Engineering + Few-Shot
Improve prompts with strong few-shot examples:
```python
# Add 3-5 perfect format examples to prompt
example_outputs = [
    "MORTALITY PROBABILITY: 0.75",
    "SURVIVAL PROBABILITY: 0.82",
    ...
]
```
**Pros**: Quick to implement, no training  
**Cons**: Uses more tokens, may not fully solve issue

### Backup Plan C: Post-Processing + Retry with Different Prompts
Keep current approach but add more robust post-processing:
```python
# If format fails, retry with progressively simpler prompts
prompts = [original_prompt, simplified_prompt, minimal_prompt]
for prompt in prompts:
    if parse_success(response):
        break
```

---

## 10. Conclusion and Recommendation

### Overall Assessment of Plan
The original VERL plan is **fundamentally sound** with clarifications:
- ✅ GRPO is correct choice for format enforcement
- ✅ Binary reward structure is appropriate (pure format, no reasoning validation)
- ✅ 996 test samples provide sufficient training data
- ✅ Mortality and survival are independent predictions (no sum=1 constraint)
- ✅ Scope is clear: format compliance only, not reasoning quality

### Recommended Path Forward

**Phase 1 (Essential)**: Data Generation
1. Run debate system on 500 test samples (from 996 available)
2. Extract integrator prompts (1000 examples: 500 mortality + 500 survival)
3. Implement parquet preprocessing with 80/20 train/val split

**Phase 2 (Core)**: RL Training
1. Implement pure format reward function (binary 0/1)
2. Run GRPO training with LoRA (safer than full fine-tuning)
3. Monitor format compliance and accuracy on validation set

**Phase 3 (Validation)**: Integration & Testing
1. Load RL checkpoint into debate system
2. Test on held-out samples (100-200 patients)
3. Compare format compliance and accuracy vs baseline
4. Verify no degradation in task performance

**Phase 4 (Optional)**: Expansion
1. If format compliance <95%, generate data from remaining 496 test samples
2. Continue training with expanded dataset
3. Fine-tune hyperparameters if needed

### Go/No-Go Decision Points

**After Phase 1 (Data Generation)**: 
- ✅ Proceed if data generation completes successfully (500 samples in 1-1.5 GPU days)
- ✅ Validate parquet format loads correctly in VERL
- ❌ Abort if data generation fails or exceeds 3 GPU days

**After Phase 2 (Training)**:
- ✅ Proceed if format compliance ≥90% on validation set
- ⚠️ If format compliance 70-90%, try Phase 4 (expand dataset)
- ❌ Switch to Backup Plan A (constrained decoding) if format compliance <70%

**After Phase 3 (Integration)**:
- ✅ Deploy if format compliance ≥95% AND accuracy ≥ baseline
- ⚠️ Iterate if format good but accuracy drops (adjust KL penalty, use LoRA)
- ❌ Revert to baseline if accuracy drops >5% and unfixable

---

## Appendix A: File Structure

```
githubs/Debate/KARE/
├── mortality_debate_rag.py          # Main debate system (MODIFY)
├── run_kare_debate_mortality.py     # Evaluation script (MODIFY)
├── kare_data_adapter.py             # Data loading (UNCHANGED)
│
├── rl_training/                     # NEW DIRECTORY
│   ├── generate_rl_training_data.py    # Generate training data
│   ├── kare_format_reward.py           # Custom reward function
│   └── run_kare_integrator_grpo.sh     # Training script
│
├── data/rl_training/                # GENERATED DATA
│   ├── train.parquet
│   ├── val.parquet
│   └── raw_contexts/                   # Intermediate contexts
│
└── rl_checkpoints/                  # RL MODEL OUTPUTS
    ├── checkpoint_500/
    ├── checkpoint_1000/
    └── best_checkpoint/

githubs/verl/
├── examples/data_preprocess/
│   └── kare_integrator_format.py    # NEW: Preprocessing script
```

## Appendix B: Key VERL Commands

```bash
# Install VERL
cd /data/wang/junh/githubs/verl
pip install -e .

# Preprocess data
python examples/data_preprocess/kare_integrator_format.py \
    --input_dir /path/to/raw/contexts \
    --output_dir /path/to/parquet

# Run GRPO training
bash run_kare_integrator_grpo.sh

# Monitor training
wandb login
# Check dashboard at https://wandb.ai/your-project/kare_integrator_format_rl
```

## Appendix C: Debugging Checklist

- [ ] Parquet files loadable by VERL
- [ ] Custom reward function returns 0.0 or 1.0
- [ ] Reward function handles edge cases (empty strings, malformed output)
- [ ] Training loop starts without errors
- [ ] Validation format compliance improves over epochs
- [ ] GPU memory usage reasonable (<80% of 80GB)
- [ ] Checkpoint saving working correctly
- [ ] Integration script loads checkpoint successfully
- [ ] Validation accuracy matches or exceeds baseline

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-11  
**Author**: AI Assistant  
**Status**: Ready for Implementation
