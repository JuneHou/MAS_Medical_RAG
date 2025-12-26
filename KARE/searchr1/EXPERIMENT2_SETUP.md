# Search-R1 Experiment 2: Probability-Based Calibration

## Overview
This document describes the setup for Experiment 2, which trains Search-R1 with probability-based rewards instead of binary Exact Match (EM) rewards.

## Key Changes

### 1. Prompt Alignment with Downstream Task
The training prompt has been updated to match the `balanced_clinical_integrator` format used during inference:

**Key Features:**
- Conservative mortality assessment (requires strong evidence for high mortality)
- Requests both MORTALITY and SURVIVAL probabilities
- Emphasizes that probabilities must sum to 1.0
- Instructs model to analyze both risk factors AND protective factors
- Uses Search-R1 tags: `<think>`, `<search>`, `<answer>`

**Location:** [data_generation/prepare_searchr1_balanced_data.py](data_generation/prepare_searchr1_balanced_data.py) in `create_single_agent_prompt()` function

### 2. Custom Reward Function
Created `kare_mortality_probability.py` with Option A (positive-only) reward:

**Formula:**
```
If ground_truth == 1 (patient died):
    reward = mortality_probability

If ground_truth == 0 (patient survived):
    reward = 1.0 - mortality_probability
```

**Range:** [0.0, 1.0] (smoother optimization than ±1/0/-1)

**Validation:**
- Extracts both MORTALITY and SURVIVAL probabilities using regex
- Validates sum equals 1.0 (±0.01 tolerance)
- Returns 0.0 for invalid/unparseable outputs
- Logs debug info to `/tmp/searchr1_reward_debug.txt`

**Location:** [reward_functions/kare_mortality_probability.py](reward_functions/kare_mortality_probability.py)

### 3. Data Generation Updates
Modified `prepare_searchr1_balanced_data.py` to support dual-probability format:

**New Flag:** `--probability` (enables Experiment 2 mode)

**Changes:**
- `create_single_agent_prompt()`: Added `use_probability_format` parameter
  - When True: Uses downstream task prompt format
  - When False: Uses binary prediction format (Experiment 1)
  
- `format_sample_for_searchr1()`: Sets `ability` field based on experiment type
  - Experiment 1: `ability = "medical-mortality-prediction"` (uses default EM reward)
  - Experiment 2: `ability = "kare_mortality_probability"` (uses custom reward)
  
- Fixed syntax errors (double comma, incomplete arguments)

**Location:** [data_generation/prepare_searchr1_balanced_data.py](data_generation/prepare_searchr1_balanced_data.py)

## Running Experiment 2

### Step 1: Generate Probability-Based Data
```bash
cd /data/wang/junh/githubs/Debate/KARE/searchr1

# Training data
python data_generation/prepare_searchr1_balanced_data.py \
    --balanced_json balanced_data/train_balanced_100pos_100neg.json \
    --split train \
    --output_dir data/kare_mortality_probability \
    --probability

# Validation data
python data_generation/prepare_searchr1_balanced_data.py \
    --balanced_json balanced_data/val_balanced_50pos_50neg.json \
    --split val \
    --output_dir data/kare_mortality_probability \
    --probability
```

### Step 2: Register Custom Reward in Search-R1
The custom reward function needs to be registered in Search-R1's config. Update the reward mapping to include:

```python
# In Search-R1 config or reward registry:
reward_functions = {
    "medical-mortality-prediction": qa_em.compute_score_em,  # Experiment 1 (default)
    "kare_mortality_probability": kare_mortality_probability.compute_score  # Experiment 2
}
```

### Step 3: Start MedRAG Retrieval Server
```bash
python searchr1/medrag_retrieval_server.py --port 8000
```

### Step 4: Train Search-R1
Update `train_searchr1_single_agent.sh` to point to the new data directory:
```bash
DATA_DIR="data/kare_mortality_probability"
```

Then run training:
```bash
bash searchr1/train_searchr1_single_agent.sh
```

## Expected Behavior

### Training Phase
- Model receives prompts requesting MORTALITY + SURVIVAL probabilities
- Model outputs format:
  ```
  <answer>
  MORTALITY PROBABILITY: 0.35
  SURVIVAL PROBABILITY: 0.65
  </answer>
  ```
- Custom reward function:
  - Validates probabilities sum to 1.0
  - Computes calibration-based reward
  - Penalizes invalid outputs (reward = 0.0)

### Inference Phase
- Use the same `balanced_clinical_integrator` prompt
- Model should be better calibrated due to probability-based training
- Can evaluate with proper scoring rules (Brier score, log loss)

## Comparison with Experiment 1

| Aspect | Experiment 1 (Binary) | Experiment 2 (Probability) |
|--------|----------------------|---------------------------|
| **Prompt** | Requests binary label (0/1) | Requests MORTALITY + SURVIVAL probabilities |
| **Output Format** | `<answer>0</answer>` or `<answer>1</answer>` | `MORTALITY PROBABILITY: X.XX\nSURVIVAL PROBABILITY: Y.XX` |
| **Reward Function** | Exact Match (EM): 1.0 if correct, 0.0 if wrong | Calibration: `mort_prob` if GT=1, `1-mort_prob` if GT=0 |
| **Reward Range** | {0.0, 1.0} (binary) | [0.0, 1.0] (continuous) |
| **Ability Tag** | `medical-mortality-prediction` | `kare_mortality_probability` |
| **Optimization** | Discrete label matching | Probability calibration |
| **Evaluation Metrics** | Accuracy, F1, AUROC | Brier score, Log loss, ECE |

## Validation Checklist
- [ ] Both data files generated (train.parquet, val.parquet)
- [ ] Parquet files contain `ability = "kare_mortality_probability"`
- [ ] MedRAG server running and responsive
- [ ] Custom reward function registered in Search-R1 config
- [ ] Training script points to correct data directory
- [ ] Reward debug logs appear in `/tmp/searchr1_reward_debug.txt`

## Troubleshooting

### Invalid Output Format
**Symptom:** Reward = 0.0 for all samples
**Solution:** Check `/tmp/searchr1_reward_debug.txt` for extraction errors

### Probabilities Don't Sum to 1.0
**Symptom:** Many samples get reward = 0.0 due to validation failure
**Solution:** 
1. Increase tolerance in reward function (currently 0.01)
2. Review model outputs to see if it's learning the constraint
3. Add explicit loss term for sum constraint

### Training Instability
**Symptom:** Reward variance is high
**Solution:** Consider Option B (±1/0/-1) for more stable gradients

### Low Rewards Even for Correct Predictions
**Symptom:** Average reward < 0.5 for GT=1 cases
**Solution:** This is expected for probability calibration - model needs to learn confidence, not just accuracy

## Next Steps
After successful training:
1. Evaluate calibration (ECE, reliability diagrams)
2. Compare with Experiment 1 (binary) baseline
3. Test on held-out test set
4. Analyze probability distributions by GT class
5. Consider ensemble with binary model (Experiment 1)
