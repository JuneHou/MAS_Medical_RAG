# KARE Mortality Format Enforcement Training with VERL GRPO

This directory contains training setup for using VERL's GRPO (Group Relative Policy Optimization) to train Qwen2.5-7B-Instruct to reliably output mortality probabilities in the correct format.

## 🎯 Training Objective

**Format-only training** (not accuracy training) to ensure the integrator model outputs probabilities in the correct format:

**Mortality Format:**
```
MORTALITY PROBABILITY: X.XX
```

**Survival Format:**
```
SURVIVAL PROBABILITY: X.XX
```

where `X.XX` is a valid float between 0.00 and 1.00.

### Why Format Training?

The current integrator often generates verbose explanations without structured outputs, causing parsing failures. This RL training uses **binary rewards**:
- ✅ **Reward = 1.0**: Valid format with probability in [0.0, 1.0]
- ❌ **Reward = 0.0**: Missing format or invalid value

### Training Strategy

1. **Phase 1 (Completed)**: Train on mortality format with hard samples (step 57, ~75-81% reward)
2. **Phase 2 (Current)**: Continue from mortality checkpoint to learn survival format
3. **Result**: Dual-format model supporting both mortality and survival probability outputs

## 📊 Dataset

- **Source**: Generated from 996 MIMIC-III mortality test samples
- **Formats**: Both mortality and survival probability formats supported
- **Modes**: Standard (all samples) and Hard (extraction failure samples only)
- **Locations**:
  - `data_generation/mortality_grpo_data/` - Standard mortality dataset (~996 samples)
  - `data_generation/mortality_grpo_data_hard/` - Hard mortality samples (193 samples: 154 train / 39 test)
  - `data_generation/survival_grpo_data/` - Standard survival dataset (~996 samples)
  - `data_generation/survival_grpo_data_hard/` - Hard survival samples (193 samples: 154 train / 39 test)
- **Split**: ~80% train / ~20% test
- **Format**: Parquet files with chat-formatted prompts

### Hard Mode

**Hard mode** filters for samples where the base model failed format extraction (identified by "EXTRACTED MORTALITY PROBABILITY: None" in logs). These challenging samples help the model learn robust format compliance:

- **Hard samples identified**: Both mortality and survival datasets use the **same 193 samples**
  - Only difference: system prompt specifies "MORTALITY PROBABILITY" vs "SURVIVAL PROBABILITY"
  - Same patient data, same agent analyses, same retrieved documents

Generate hard samples with:
```bash
python generate_training_data.py --class_type mortality --hard  # 193 samples
python generate_training_data.py --class_type survival --hard   # Same 193 samples, different format
```

### Prompt Structure
Each sample contains:
1. System instructions (mortality or survival assessment based on `--class_type`)
2. Patient EHR context (conditions, medications, procedures)
3. Three agent analyses (target patient, mortality risk, protective factors)
4. Retrieved medical evidence (MedRAG documents)
5. Task instruction to output probability in specified format

**Prompt Statistics**:
- Mean: 13,134 chars (~3,283 tokens)
- 95th percentile: 18,091 chars (~4,523 tokens)
- Max: 60,932 chars (~15,233 tokens)

## 🚀 Quick Start

### Step 1: Generate Training Data

```bash
cd /data/wang/junh/githubs/Debate/KARE/verl/data_generation

# Generate standard mortality dataset
python generate_training_data.py --class_type mortality

# Generate hard mortality samples (recommended for fine-tuning)
python generate_training_data.py --class_type mortality --hard

# Generate survival datasets
python generate_training_data.py --class_type survival
python generate_training_data.py --class_type survival --hard
```

### Step 2: Launch Training

**Phase 1: Mortality Format Training**
```bash
cd /data/wang/junh/githubs/Debate/KARE/verl

# Make sure you're in the medrag conda environment
source /usr/local/anaconda3/bin/activate /data/wang/junh/envs/medrag

# Edit run_kare_mortality_grpo.sh to use mortality dataset
# Set: train_path=.../mortality_grpo_data_hard/train.parquet
#      custom_reward_function.path=.../kare_mortality_format.py

# Launch training on GPUs 3,4,5,6
./run_kare_mortality_grpo.sh
```

**Phase 2: Survival Format Training (from mortality checkpoint)**
```bash
# Edit run_kare_mortality_grpo.sh:
#   1. Set train_path to survival_grpo_data_hard/train.parquet
#   2. Set actor_rollout_ref.model.path to your mortality checkpoint
#   3. Set custom_reward_function.path to kare_survival_format.py
#   4. Update trainer.experiment_name to include 'survival'

./run_kare_mortality_grpo.sh
```

Training will run for **3 epochs** with checkpoints saved after each epoch.

## ⚙️ Configuration

### Model & Hardware
- **Base Model**: Qwen/Qwen2.5-7B-Instruct (Phase 1) → Fine-tuned checkpoint (Phase 2)
- **GPUs**: 4x A40 44GB (GPUs 3,4,5,6)
- **Tensor Parallel**: 2 (model split across 2 GPUs per instance)
- **Data Parallel**: 2 (2 model instances for parallel training)
- **GPU Utilization**: 0.4 (40% = ~17.7GB per GPU)

### Training Hyperparameters
```yaml
Algorithm: GRPO (Group Relative Policy Optimization)
Learning Rate: 1e-6
Batch Size: 8 (total)
  - Mini-batch: 4
  - Micro-batch per GPU: 1
Group Size (n): 4 (samples per prompt)
Epochs: 3
KL Loss: Enabled (coef=0.001)
FSDP: Enabled with parameter and optimizer offloading
```

### Context Lengths
```yaml
Max Prompt Length: 16384 tokens (covers 99%+ of samples)
Max Response Length: 4096 tokens (allows reasoning before format)
```

### Current Training Status
- **Mortality format**: ✅ Completed at step 57 (~75-81% reward on hard samples)
- **Survival format**: 🔄 Ready to train from mortality checkpoint

## 📁 Directory Structure

```
verl/
├── data_generation/
│   ├── mortality_grpo_data/           # Standard mortality dataset
│   │   ├── train.parquet (~796 samples)
│   │   └── test.parquet (~200 samples)
│   ├── mortality_grpo_data_hard/      # Hard mortality samples
│   │   ├── train.parquet (154 samples)
│   │   └── test.parquet (39 samples)
│   ├── survival_grpo_data_hard/       # Hard survival samples (same as mortality)
│   │   ├── train.parquet (154 samples)
│   │   └── test.parquet (39 samples)
│   ├── generate_training_data.py      # Data generation with --class_type and --hard
│   ├── find_hard_samples.py           # Identifies failed extraction samples
│   ├── parse_debate_logs.py
│   └── parse_retrieval_logs.py
├── reward_score/
│   ├── kare_mortality_format.py       # Mortality format reward (MORTALITY PROBABILITY: X.XX)
│   └── kare_survival_format.py        # Survival format reward (SURVIVAL PROBABILITY: X.XX)
├── models/
│   └── format_enforcer_7b_step57/     # Converted HuggingFace model (mortality)
├── checkpoints/                       # Training checkpoints (FSDP format)
│   └── global_step_57/                # Mortality training checkpoint
├── logs/                              # Training logs (auto-created)
├── convert_fsdp_to_hf.py             # Checkpoint conversion script
├── convert_checkpoint.sh              # Conversion convenience wrapper
├── run_kare_mortality_grpo.sh        # Main training script
├── setup_wandb.sh                     # WandB setup helper
└── README.md                          # This file
```

## 📈 Monitoring Training

### WandB Dashboard (Online Mode)
Visit: https://wandb.ai/<your-username>/verl-kare-mortality-format

**Key Metrics to Watch**:
- `reward/mean`: Should increase toward 1.0 (format compliance)
- `reward/std`: Should decrease (more consistent outputs)
- `kl_divergence`: Should stay low (<0.1) to preserve base model capabilities
- `loss/total`: Should decrease steadily

### Checkpoints
Saved after each epoch to: `/data/wang/junh/githubs/Debate/KARE/verl/checkpoints/`

Format: `grpo-qwen2.5-7b-it-format_epoch_N/`

## 🔧 Troubleshooting

### Out of Memory (OOM)
If you hit OOM errors:
1. Reduce `ppo_micro_batch_size_per_gpu` from 4 to 2
2. Reduce `max_response_length` from 4096 to 2048
3. Enable optimizer offload: `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`

### Slow Training
If training is too slow:
1. Increase `gpu_memory_utilization` from 0.65 to 0.75
2. Reduce `max_prompt_length` to 12288 (will truncate ~5% of samples)

### Low Reward Not Improving
If reward stays low (<0.3) after 3 epochs:
1. Check reward function is working: `python -c "from reward_score.kare_mortality_format import compute_score; print(compute_score('MORTALITY PROBABILITY: 0.75'))"`
2. Increase learning rate to 5e-6
3. Increase group size `n` from 2 to 4

### WandB Issues
- **"wandb: ERROR API key not configured"**: Run `wandb login` or use offline mode
- **"wandb: Network error"**: Use offline mode: `export WANDB_MODE=offline`
- **Don't want WandB**: Set `WANDB_DISABLED=true` or change trainer config to `trainer.logger='["console"]'`

### Data Loading Error: `'str' object has no attribute 'get'` or `KeyError: 'reward_model'`
If you see these errors during training:
- **Cause 1**: `extra_info` field was JSON string instead of dict
- **Cause 2**: Missing `reward_model` column (VERL requires this structure)
- **Quick Fix** (convert existing files without regeneration):
  ```bash
  cd /data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction
  python -c "
import pandas as pd
import json
for fname in ['train.parquet', 'val.parquet']:
    try:
        df = pd.read_parquet(fname)
        # Fix 1: Convert extra_info from JSON string to dict
        if 'extra_info' in df.columns:
            df['extra_info'] = df['extra_info'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        # Fix 2: Add reward_model column if missing
        if 'reward_model' not in df.columns and 'ground_truth' in df.columns:
            df['reward_model'] = df.apply(lambda row: {'ground_truth': row['ground_truth']}, axis=1)
        df.to_parquet(fname, index=False)
        print(f'✓ Fixed {fname}')
    except FileNotFoundError:
        pass
"
  ```
- **Note**: `reward_model` is just a data structure containing `ground_truth`. Your actual reward function (`kare_prediction_reward.py`) runs during training - no separate reward model needed!

## 📝 Key Implementation Details

### Reward Function
Located in `reward_score/kare_mortality_format.py`:
```python
def compute_score(solution_str, ground_truth=None, format_score=0.0, score=1.0):
    # Extract probability using regex
    prob = extract_mortality_probability(solution_str)
    if prob is None:
        return 0.0  # No format found
    if 0.0 <= prob <= 1.0:
        return 1.0  # Valid format
    return 0.0  # Invalid value
```

### GRPO Algorithm
- **Group sampling**: Generates `n=2` responses per prompt
- **Relative rewards**: Compares responses within each group
- **No critic**: Uses group average as baseline (simpler than PPO)
- **KL penalty**: Prevents model drift from base capabilities

### Data Generation
Training data is generated from debate logs using:
1. `parse_debate_logs.py`: Extracts 3 agent analyses with hierarchical fallback
2. `parse_retrieval_logs.py`: Loads MedRAG retrieved documents
3. `generate_training_data.py`: Assembles complete integrator prompts
   - `--class_type mortality|survival`: Choose format (default: mortality)
   - `--hard`: Filter for samples where base model failed extraction (303 samples)
4. `find_hard_samples.py`: Scans logs for "EXTRACTED MORTALITY PROBABILITY: None"

**Example usage:**
```bash
# Standard mortality dataset (all samples)
python generate_training_data.py --class_type mortality

# Hard mortality samples only (challenging cases)
python generate_training_data.py --class_type mortality --hard

# Survival format datasets
python generate_training_data.py --class_type survival
python generate_training_data.py --class_type survival --hard
```

## 📚 References

- **VERL Framework**: https://github.com/volcengine/verl
- **GRPO Paper**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
- **KARE**: Knowledge-Augmented Reasoning for EHR mortality prediction

## 🔄 Converting Checkpoint to HuggingFace Format

After training completes, convert the FSDP checkpoint to HuggingFace format for use with VLLM:

```bash
# Convert the latest checkpoint (step 57)
./convert_checkpoint.sh

# Or convert a specific checkpoint
./convert_checkpoint.sh 50  # For global_step_50
```

This will create a HuggingFace-compatible model at:
```
/data/wang/junh/githubs/Debate/KARE/verl/models/format_enforcer_7b_step57/
```

**What the conversion does**:
- Merges FSDP sharded weights (`model_world_size_4_rank_*.pt`) into single model
- Converts to HuggingFace format compatible with transformers/vllm
- Preserves tokenizer and model config

### 💾 Saving Disk Space After Conversion

Once you've successfully converted and tested the model, you can delete the original FSDP checkpoint files to save **~86GB per checkpoint**:

```bash
# ONLY delete after verifying the converted model works!
cd /data/wang/junh/githubs/Debate/KARE/verl/checkpoints/global_step_57/actor

# Safe to delete (86GB total):
rm model_world_size_4_rank_*.pt      # 28GB - model weights (sharded)
rm optim_world_size_4_rank_*.pt      # 58GB - optimizer states
rm extra_state_world_size_4_rank_*.pt  # 64KB - training state

# Keep these (16MB total):
# - fsdp_config.json (needed for reference)
# - huggingface/ directory (tokenizer files)
```

**Disk usage breakdown**:
- Original FSDP checkpoint: ~86GB (sharded model + optimizer states)
- Converted HuggingFace model: ~15GB (merged safetensors)
- **Space saved**: ~71GB per checkpoint

**⚠️ Important**: Only delete after:
1. Conversion completes successfully
2. You've tested the converted model with `test_converted_model.py`
3. You've verified it works in your debate system

## 🤝 Integration After Training

Once checkpoint is converted, use it as the integrator model in your debate system:

```bash
# Run debate with format-enforced integrator
cd /data/wang/junh/githubs/Debate/KARE

# For mortality predictions
python run_kare_debate_mortality.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --integrator_model /data/wang/junh/githubs/Debate/KARE/verl/models/format_enforcer_7b_step57 \
    --gpus 3,4,5,6 \
    --integrator_gpu 4 \
    --mode rag \
    --num_samples 100

# For survival predictions (if using dual-format model)
# System prompt should request "SURVIVAL PROBABILITY: X.XX" format
```

### Python API Usage

```python
from mortality_debate_rag import MortalityDebateSystem

# Initialize with format-enforced integrator
debate_system = MortalityDebateSystem(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Agents 1-3
    integrator_model_name="/data/wang/junh/githubs/Debate/KARE/verl/models/format_enforcer_7b_step57",  # Agent 4
    gpu_ids="3,4,5,6",
    integrator_gpu="4"
)

# Run prediction - integrator will now always output correct format!
result = debate_system.debate_mortality_prediction(
    patient_context=patient_data,
    positive_similars=positive_examples,
    negative_similars=negative_examples
)
```

This should significantly reduce parsing failures and improve output consistency!

---

# KARE Mortality Prediction Accuracy Training with VERL GRPO

**NEW**: This section covers **prediction accuracy training** (distinct from the format enforcement training above).

## 🎯 Training Objective

**Prediction accuracy training** to improve the integrator's mortality and survival probability predictions using reinforcement learning with actual patient outcomes.

### Goal
Train the integrator to produce **more accurate probability estimates** that better reflect true patient mortality risk, rewarding predictions that align with ground truth outcomes.

### Why Prediction Training?

The format-enforced model reliably outputs structured probabilities, but those probabilities may not be well-calibrated. This RL training uses **outcome-based rewards**:

**Reward Function (Symmetric ±1/0/-1)**:
- **Ground Truth = 0 (Survival)**:
  - Mortality < 0.4 → **+1** (correctly low mortality prediction)
  - Mortality [0.4, 0.7) → **0** (uncertain)
  - Mortality ≥ 0.7 → **-1** (incorrectly high mortality prediction)
  - Survival ≥ 0.6 → **+1** (correctly high survival prediction)
  - Survival [0.3, 0.6) → **0** (uncertain)
  - Survival < 0.3 → **-1** (incorrectly low survival prediction)

- **Ground Truth = 1 (Mortality)**:
  - Mortality ≥ 0.7 → **+1** (correctly high mortality prediction)
  - Mortality [0.4, 0.7) → **0** (uncertain)
  - Mortality < 0.4 → **-1** (incorrectly low mortality prediction)
  - Survival < 0.3 → **+1** (correctly low survival prediction)
  - Survival [0.3, 0.6) → **0** (uncertain)
  - Survival ≥ 0.6 → **-1** (incorrectly high survival prediction)

## 📊 Dataset

- **Source**: KARE MIMIC-III mortality dataset with balanced sampling
- **Location**: `/data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction/`
- **Splits**: 
  - Train: ~200 samples (100 positive mortality + 100 negative survival)
  - Val: ~177 samples (77 positive + 100 negative - limited by available positive cases)
- **Format**: Parquet files with integrator prompts including:
  - Target patient EHR context
  - 3-round debate history (target analysis, mortality risk assessment, protective factors)
  - Retrieved medical evidence (mortality and survival specific)
  - Separate prompts for mortality and survival probability predictions

### Key Differences from Format Training

| Aspect | Format Training | Prediction Training |
|--------|----------------|---------------------|
| **Goal** | Output valid format | Output accurate probabilities |
| **Reward** | Binary (valid/invalid) | Continuous (-1/0/+1 based on accuracy) |
| **Data** | All test samples (~996) | Balanced train samples (~200) |
| **Ground Truth** | Not used (format only) | Critical (determines reward) |
| **Context** | Debate logs only | Debate logs + retrieved medical evidence |

## 🚀 Step-by-Step Setup

### Step 1: Sample Balanced Training Data

The prediction task requires balanced positive/negative samples to avoid bias. Use the sampling script to filter patients with complete similar patient data:

```bash
cd /data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction

# Sample balanced training data (100 positive + 100 negative)
python sample_balanced_data.py --split train --n_positive 100 --n_negative 100

# Sample balanced validation data (use all available positive samples)
python sample_balanced_data.py --split val --n_positive 100 --n_negative 100
```

**Output**: 
- `train_balanced_100pos_100neg.json` - Training patient IDs
- `val_balanced_77pos_100neg.json` - Validation patient IDs (only 77 positive available)

**Note**: The script automatically filters for patients with complete similar patient contexts (both positive and negative similar patients must exist).

### Step 2: Generate GRPO Training Data

Run the full 3-round debate pipeline to generate integrator prompts with retrieved medical evidence:

```bash
cd /data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction

python generate_prediction_training_data.py \
    --split train \
    --balanced_file train_balanced_100pos_100neg.json \
    --gpus 3,4 \
    --model Qwen/Qwen2.5-7B-Instruct

python generate_prediction_training_data.py \
    --split val \
    --balanced_file val_balanced_77pos_100neg.json \
    --gpus 3,4 \
    --model Qwen/Qwen2.5-7B-Instruct
```

**What this does**:
1. Loads balanced patient samples
2. Runs full 3-round debate for each patient, EXACTLY matching downstream deployment:
   - Round 1: Target patient analysis (with retrieval using full patient EHR as query)
   - Round 2: Mortality risk + protective factor analysis (parallel batch, with retrieval using full patient EHR as query)
   - Round 3: **Integrator LLM tool calling** (exactly as in deployment):
     - Calls integrator LLM with mortality system prompt → LLM generates short query via tool call
     - Executes retrieval with LLM-generated query → saves logs
     - Calls integrator LLM with survival system prompt → LLM generates short query via tool call
     - Executes retrieval with LLM-generated query → saves logs
3. Constructs two training prompts per patient that EXACTLY match downstream deployment:
   - **Mortality prompt**: System prompt + `## Target Patient EHR Context ##` + `## Previous Debate Analysis ##` + `You called: retrieve("LLM-generated-query")` + Retrieved Evidence + instruction
   - **Survival prompt**: Same structure with survival-specific system prompt and LLM-generated query
4. Saves to Parquet with ground truth labels and metadata

**Output**:
- `train.parquet` - ~400 training examples (200 patients × 2 assessments)
- `val.parquet` - ~354 validation examples (177 patients × 2 assessments)

**Expected time**: ~2-3 minutes per patient (200 patients ≈ 6-10 hours for training data)

### Step 3: Verify Generated Data

Check the generated Parquet files to ensure data quality:

```bash
cd /data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction

# Quick inspection
python -c "
import pandas as pd
df = pd.read_parquet('train.parquet')
print(f'Training examples: {len(df)}')
print(f'Ground truth distribution: {df[\"ground_truth\"].value_counts()}')
print(f'Assessment types: {df[\"extra_info\"].apply(lambda x: eval(x)[\"assessment_type\"]).value_counts()}')
print(f'Mean prompt length: {df[\"prompt\"].str.len().mean():.0f} chars')
"
```

**Expected output**:
```
Training examples: 400
Ground truth distribution: 
0    200  # Survival cases
1    200  # Mortality cases
Assessment types:
mortality    200
survival     200
Mean prompt length: ~15000 chars (~3750 tokens)
```

### Step 4: Launch Prediction GRPO Training

**Prerequisites**:
- Format-enforced model checkpoint from Phase 1 (optional but recommended)
- Reward function: `reward_score/kare_prediction_reward.py`
- Training script: `run_kare_prediction_grpo.sh` (separate from format training)

```bash
cd /data/wang/junh/githubs/Debate/KARE/verl

# Activate environment
source /usr/local/anaconda3/bin/activate /data/wang/junh/envs/medrag

# Create prediction training script by copying and modifying format training script
cp run_kare_mortality_grpo.sh run_kare_prediction_grpo.sh

# Edit run_kare_prediction_grpo.sh:
# 1. Set train_path=data_generation/prediction/train.parquet
# 2. Set val_path=data_generation/prediction/val.parquet (optional)
# 3. Set custom_reward_function.path=reward_score/kare_prediction_reward.py
# 4. Set actor_rollout_ref.model.path to format-enforced checkpoint, use base model: Qwen/Qwen2.5-7B-Instruct
# 5. Update trainer.experiment_name='verl-kare-prediction-accuracy'
# 6. Set trainer.n_epochs=3
# 7. Set optimizer.lr=1e-6 (conservative for accuracy tuning)
# 8. Verify trainer.project_name='verl-kare-prediction' (for WandB organization)

# Launch training
./run_kare_prediction_grpo.sh
```

**Training Configuration**:
```yaml
Algorithm: GRPO
Base Model: Format-enforced Qwen2.5-7B-Instruct (from Step 57) or base model
Learning Rate: 1e-6 (conservative for prediction accuracy)
Batch Size: 8
Group Size (n): 4 (samples per prompt for relative comparison)
Epochs: 3
KL Coefficient: 0.01 (prevent drift from medical knowledge)
Max Prompt Length: 16384 tokens
Max Response Length: 4096 tokens
```

**Monitoring**:
- WandB dashboard: https://wandb.ai/<username>/verl-kare-prediction-accuracy
- Key metrics:
  - `reward/mean`: Should increase from ~0.0 toward +0.5-0.7
  - `reward/by_ground_truth/0` and `reward/by_ground_truth/1`: Both should improve
  - `kl_divergence`: Should stay low (<0.1)
  - `loss/total`: Should decrease

### Step 5: Evaluate Trained Model

After training completes, convert checkpoint and evaluate on test set:

```bash
cd /data/wang/junh/githubs/Debate/KARE/verl

# Convert FSDP checkpoint to HuggingFace format
./convert_checkpoint.sh  # Uses latest checkpoint by default

# Evaluate on KARE test set (996 samples)
cd /data/wang/junh/githubs/Debate/KARE

python run_kare_debate_mortality.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --integrator_model /data/wang/junh/githubs/Debate/KARE/verl/models/prediction_accuracy_7b_step<N> \
    --gpus 3,4,5,6 \
    --integrator_gpu 4 \
    --mode rag \
    --num_samples 996 \
    --output_dir results/prediction_grpo_evaluation
```

**Expected Improvements**:
- Better calibrated probabilities (ECE metric)
- Improved AUROC for mortality prediction
- Higher accuracy at standard thresholds (0.5, 0.7)
- More confident predictions (higher separation between classes)

## 📁 Prediction Training Directory Structure

```
verl/
├── data_generation/
│   └── prediction/                          # NEW: Prediction accuracy training data
│       ├── sample_balanced_data.py          # Step 1: Sample balanced patients
│       ├── generate_prediction_training_data.py  # Step 2: Generate GRPO data
│       ├── kare_data_adapter.py             # Data loading utilities
│       ├── mortality_debate_rag_grpo.py     # Debate system (renamed for consistency)
│       ├── run_kare_debate_mortality_grpo.py # Evaluation script
│       ├── train_balanced_100pos_100neg.json # Balanced training patient IDs
│       ├── val_balanced_77pos_100neg.json   # Balanced validation patient IDs
│       ├── train.parquet                    # Training data (~400 examples)
│       ├── val.parquet                      # Validation data (~354 examples)
│       └── debate_logs/                     # Debate + retrieval logs
├── reward_score/
│   ├── kare_mortality_format.py             # Format enforcement reward
│   ├── kare_survival_format.py              # Survival format reward
│   └── kare_prediction_reward.py            # NEW: Prediction accuracy reward
├── models/
│   ├── format_enforcer_7b_step57/           # Format-enforced model (base for prediction)
│   └── prediction_accuracy_7b_step<N>/      # NEW: Prediction-tuned model
├── checkpoints/                             # Training checkpoints
├── run_kare_mortality_grpo.sh              # Format enforcement training script
├── run_kare_prediction_grpo.sh             # NEW: Prediction accuracy training script
└── README.md                                # This file
```

## 🔍 Key Implementation Details

### Prompt Construction

Each training example contains the EXACT prompt format used in downstream deployment:

1. **System Prompt** (mortality or survival specific):
   ```
   You are a medical AI Clinical Assistant analyzing MORTALITY risk...
   Available tools: retrieve(query)
   Instructions: Call retrieve() with custom query, review evidence, provide probability
   ```

2. **Primary Context**:
   ```
   ## Target Patient EHR Context ##
   Patient ID: XXXXX
   Visit N:
   Conditions: ...
   Procedures: ...
   Medications: ...
   ```

3. **Debate History**:
   ```
   ## Previous Debate Analysis ##
   target_patient_analyst: [analysis with prediction]
   mortality_risk_assessor: [mortality risk factors analysis]
   protective_factor_analyst: [protective factors analysis]
   ```

4. **Simulated Tool Call**:
   ```
   You called: retrieve("respiratory failure pneumonia mortality outcomes")
   ```
   Note: The query is generated by ACTUAL integrator LLM tool calling (not manual extraction), exactly matching downstream deployment behavior.

5. **Retrieved Medical Evidence** (assessment-specific):
   ```
   Retrieved Evidence:
   [Document 1] Title
   Content...
   
   [Document 2] Title
   Content...
   ```

6. **Task Instruction**:
   ```
   Now provide your complete [mortality/survival] probability assessment based on the retrieved evidence:
   ```

**Key Point**: This EXACTLY matches the downstream deployment prompt in `mortality_debate_rag.py` Step 1c/2c (`mortality_reasoning_prompt`/`survival_reasoning_prompt`), ensuring the model trained with GRPO will work seamlessly in production.

### Reward Function Logic

Located in `reward_score/kare_prediction_reward.py`:

```python
def compute_score(solution_str, ground_truth, **kwargs):
    # Extract both probabilities
    mort_prob = extract_mortality_probability(solution_str)
    surv_prob = extract_survival_probability(solution_str)
    
    # Determine which probability to use based on prompt
    assessment_type = kwargs.get('assessment_type', 'mortality')
    prob = mort_prob if assessment_type == 'mortality' else surv_prob
    
    if prob is None:
        return 0.0  # No valid probability found
    
    # Symmetric reward based on ground truth
    if ground_truth == 0:  # Survival
        if assessment_type == 'mortality':
            if prob < 0.4: return 1.0    # Correctly low
            if prob < 0.7: return 0.0    # Uncertain
            return -1.0                   # Incorrectly high
        else:  # survival
            if prob >= 0.6: return 1.0   # Correctly high
            if prob >= 0.3: return 0.0   # Uncertain
            return -1.0                   # Incorrectly low
    
    else:  # Mortality (ground_truth == 1)
        if assessment_type == 'mortality':
            if prob >= 0.7: return 1.0   # Correctly high
            if prob >= 0.4: return 0.0   # Uncertain
            return -1.0                   # Incorrectly low
        else:  # survival
            if prob < 0.3: return 1.0    # Correctly low
            if prob < 0.6: return 0.0    # Uncertain
            return -1.0                   # Incorrectly high
### Training Strategy

**Why Two Prompts Per Patient?**
- Each patient generates both mortality and survival prompts
- Model learns to assess risk from both perspectives
- Improves calibration: mort_prob + surv_prob should ≈ 1.0
- GRPO compares multiple responses per prompt, learning which probabilities are more accurate

**Query Generation for Training Data** (EXACTLY matches downstream deployment):
- **Rounds 1-2 (target_patient_analyst, mortality_risk_assessor, protective_factor_analyst)**: Use FULL patient EHR context as retrieval query
- **Round 3 (balanced_clinical_integrator)**: Uses actual LLM tool calling to generate queries
  - Step 3a: Call integrator LLM with mortality system prompt → LLM generates short query (e.g., "respiratory failure mortality risk")
  - Step 3b: Execute retrieval with LLM-generated query
  - Step 3c: Call integrator LLM with survival system prompt → LLM generates short query (e.g., "diabetes treatment survival outcomes")
  - Step 3d: Execute retrieval with LLM-generated query
- Training data preparation replicates Steps 1-3 of deployment exactly, with real LLM tool calls and retrieved evidence

**Balanced Sampling**:
- Equal positive (mortality=1) and negative (mortality=0) samples
- Prevents model from learning dataset bias
- Ensures reward function sees both outcome types equally

**Conservative Learning Rate**:
- 1e-6 (vs 1e-5 for format training)
- Preserves medical knowledge from base model
- Focuses on calibration rather than dramatic behavior change
- Preserves medical knowledge from base model
- Focuses on calibration rather than dramatic behavior change

## 🔧 Troubleshooting Prediction Training

### Low or Negative Average Reward
If `reward/mean` stays below 0.0:
- Check reward function: `python -c "from reward_score.kare_prediction_reward import compute_score; print(compute_score('MORTALITY PROBABILITY: 0.85', ground_truth=1))"`
- Verify ground truth labels in data: Should be balanced 50/50
- Increase learning rate to 5e-6
- Increase group size from 4 to 8 for better comparisons

### High Variance in Rewards
If `reward/std` is very high (>1.0):
- Model predictions may be too random
- Decrease temperature in generation
- Increase KL coefficient from 0.01 to 0.05
- Consider starting from format-enforced checkpoint

### Model Outputs Invalid Format
If format extraction fails during prediction training:
- **Must** start from format-enforced checkpoint
- Increase format reward weight (mix format + prediction rewards)
- Use lower learning rate (5e-7)

### Calibration Not Improving
If ECE (Expected Calibration Error) doesn't improve:
- Increase training epochs to 5-10
- Add calibration-specific reward term
- Use temperature scaling post-training

## 📈 Expected Results

### Baseline (Format-Enforced Model)
- **Format Success**: 95-100% (from format training)
- **Accuracy**: ~60-70% (no accuracy tuning)
- **AUROC**: ~0.70-0.75
- **ECE**: ~0.15-0.25 (moderate calibration)

### After Prediction GRPO
- **Format Success**: 95-100% (maintained)
- **Accuracy**: ~70-80% (improved)
- **AUROC**: ~0.75-0.85 (improved discrimination)
- **ECE**: ~0.05-0.15 (better calibration)
- **Average Reward**: +0.4 to +0.6 on validation set

## 🤝 Integration After Prediction Training

Use the prediction-tuned model as the integrator in your debate system:

```python
from mortality_debate_rag import MortalityDebateSystem

# Initialize with prediction-tuned integrator
debate_system = MortalityDebateSystem(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Agents 1-3
    integrator_model_name="/data/wang/junh/githubs/Debate/KARE/verl/models/prediction_accuracy_7b_step150",  # Tuned agent 4
    gpu_ids="3,4,5,6",
    integrator_gpu="4",
    rag_enabled=True
)

# Run prediction with improved accuracy
result = debate_system.debate_mortality_prediction(
    patient_context=patient_data,
    positive_similars=mortality_cases,
    negative_similars=survival_cases
)

# Extract improved probabilities
mortality_prob = result['final_mortality_probability']  # Better calibrated!
survival_prob = result['final_survival_probability']    # Better calibrated!
```

## 🆚 Comparison: Format vs Prediction Training

| Training Phase | Objective | Data | Reward | Outcome |
|---------------|-----------|------|--------|---------|
| **Phase 1: Format** | Output valid structure | All samples (~996) | Binary (valid/invalid) | Reliable format extraction |
| **Phase 2: Prediction** | Output accurate probabilities | Balanced samples (~200) | Continuous (-1/0/+1) | Calibrated risk estimates |
| **Combined** | Structure + Accuracy | Sequential training | Phased rewards | Production-ready model |

**Recommended Approach**: 
1. Start with base Qwen2.5-7B-Instruct
2. Train format enforcement (Phase 1) → 95%+ format success
3. Train prediction accuracy (Phase 2 from Phase 1 checkpoint) → Calibrated probabilities
4. Deploy combined model → Reliable structure + accurate predictions

---

# Updated Reward Function: Aggressive Brier Score

## 🎯 New Reward Logic (December 2025)

The prediction reward function has been updated from discrete threshold-based rewards to a **continuous Brier score with aggressive penalty** to address the issue of sparse training signals.

### Problem with Original Reward Function

The original threshold-based approach suffered from:
- **Sparse signals**: Most predictions fell into "uncertain" zones (reward = 0.0)
- **Weak penalties**: Bad predictions (e.g., 70% survival for mortality patients) received barely positive rewards (+0.02)
- **Limited gradient information**: Discrete thresholds provided poor optimization signals

### New Brier-Based Reward Function

**Formula**: `r = max(-1, min(1, 1 - 4(p - y)²))`

Where:
- `p` = predicted probability (mortality or survival based on assessment_type)
- `y` = target value:
  - Mortality assessment: `y = ground_truth` (0=survival, 1=mortality)
  - Survival assessment: `y = 1 - ground_truth` (1=survival, 0=mortality)

### Key Improvements

| Aspect | Original Approach | New Brier Approach |
|--------|-------------------|---------------------|
| **Signal Type** | Discrete (-1/0/+1) | Continuous (clamped to [-1,+1]) |
| **Bad Predictions** | +0.02 (barely positive!) | -0.96 (strong penalty) |
| **Random Predictions** | +0.5 (misleading) | 0.0 (neutral) |
| **Gradient Quality** | Sparse (many zeros) | Rich (non-zero everywhere) |
| **Training Stability** | Poor (sparse rewards) | Good (continuous signals) |

### Reward Characteristics

```python
# Perfect predictions (p = y)
mortality_patient_100pct_mortality = +1.0  # ✅
survival_patient_0pct_mortality = +1.0     # ✅

# Good predictions (error ≈ 0.1) 
mortality_patient_90pct_mortality = +0.64  # ✅ Still rewarded
survival_patient_10pct_mortality = +0.64   # ✅ Still rewarded

# Random predictions (error = 0.25)
any_patient_50pct_probability = 0.0        # Neutral (encourages exploration)

# Bad predictions (error ≈ 0.5)
mortality_patient_70pct_survival = -0.96   # ❌ Strong penalty (was +0.02!)
survival_patient_70pct_mortality = -0.96   # ❌ Strong penalty (was +0.02!)

# Worst predictions (error = 1.0)
mortality_patient_0pct_mortality = -1.0    # ❌ Maximum penalty
survival_patient_100pct_mortality = -1.0   # ❌ Maximum penalty
```

### Benefits

1. **Continuous Signal**: Non-zero gradients everywhere enable stable RL optimization
2. **Proper Penalties**: Bad predictions now receive strong negative rewards (-0.96 vs +0.02)
3. **Neutral Random**: 50% predictions get exactly 0.0 reward (neither encouraging nor discouraging)
4. **Aggressive but Fair**: Strong penalty for large errors, good rewards for accurate predictions
5. **Training Stability**: Clamped to [-1, +1] prevents extreme values during training

### Implementation

The updated reward function is implemented in `reward_score/kare_prediction_reward.py`:

```python
# Convert ground truth and probability to same scale
if assessment_type == 'mortality':
    y = float(ground_truth)  # 0 for survival, 1 for mortality
    p = extracted_mortality_prob
else:  # survival assessment  
    y = 1.0 - float(ground_truth)  # 1 for survival, 0 for mortality
    p = extracted_survival_prob

# Aggressive Brier score with clamping
brier_error = (p - y) ** 2
raw_reward = 1.0 - 4.0 * brier_error
reward = max(-1.0, min(1.0, raw_reward))  # Clamp to [-1, +1]
```

This provides much better training signals for GRPO optimization while maintaining the continuous nature required for stable reinforcement learning.
