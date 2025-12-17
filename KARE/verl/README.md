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

### Step 1: Set Up WandB (Optional but Recommended)

WandB provides experiment tracking, metrics visualization, and checkpoint management.

```bash
# Run the setup script
./setup_wandb.sh
```

**Setup Options**:
1. **Online mode** (with WandB account): Best for tracking and sharing experiments
   ```bash
   wandb login
   # Paste your API key from https://wandb.ai/authorize
   ```

2. **Offline mode** (no account needed): Logs saved locally
   ```bash
   export WANDB_MODE=offline
   ```

3. **Disable WandB**: Use console logging only
   ```bash
   export WANDB_DISABLED=true
   ```

### Step 2: Generate Training Data

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

### Step 3: Launch Training

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

### Console Output
Training progress is logged to console in real-time:
```
Epoch 1/10: reward=0.45, kl=0.02, loss=2.34
Epoch 2/10: reward=0.62, kl=0.03, loss=1.89
...
```

### Checkpoints
Saved after each epoch to: `/data/wang/junh/githubs/Debate/KARE/verl/checkpoints/`

Format: `grpo-qwen2.5-7b-it-format_epoch_N/`

## 🧪 Testing Format Compliance

After training, test the model's format compliance on the test set:

```python
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load trained model
model_path = "/data/wang/junh/githubs/Debate/KARE/verl/checkpoints/grpo-qwen2.5-7b-it-format_epoch_10"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load test data
test_df = pd.read_parquet("data_generation/mortality_grpo_data/test.parquet")

# Test format compliance
def check_format(output):
    pattern = r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        try:
            prob = float(match.group(1))
            return 0.0 <= prob <= 1.0
        except:
            return False
    return False

# Run inference on test samples
compliance_count = 0
for idx, row in test_df.head(50).iterrows():  # Test on 50 samples
    prompt = row['prompt']
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if check_format(response):
        compliance_count += 1

print(f"Format compliance: {compliance_count}/50 = {compliance_count/50*100:.1f}%")
```

**Target**: >95% format compliance on test set

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

## 🔄 Dual-Format Training Workflow

### Overview

The training follows a two-phase approach to create a model that handles both mortality and survival probability formats:

**Phase 1: Mortality Format (Completed)**
```bash
# 1. Generate hard mortality samples
python generate_training_data.py --class_type mortality --hard

# 2. Train from base model
# Config: actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
#         custom_reward_function.path=kare_mortality_format.py
./run_kare_mortality_grpo.sh

# 3. Convert checkpoint
./convert_checkpoint.sh 57  # Creates models/format_enforcer_7b_step57/
```

**Phase 2: Survival Format (Current)**
```bash
# 1. Generate hard survival samples
python generate_training_data.py --class_type survival --hard

# 2. Continue training from mortality checkpoint
# Config: actor_rollout_ref.model.path=/path/to/format_enforcer_7b_step57
#         custom_reward_function.path=kare_survival_format.py
./run_kare_mortality_grpo.sh

# 3. Convert final checkpoint
./convert_checkpoint.sh [final_step]  # Dual-format model ready!
```

### Why This Approach?

1. **Warm start**: Model already understands format constraints from Phase 1
2. **Efficient**: Survival training converges faster (~3 epochs vs 10+)
3. **Robust**: Hard samples ensure both formats work on challenging cases
4. **Dual capability**: Final model handles both MORTALITY and SURVIVAL formats

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

**Key points**:
- ✅ **Regular agents** use base `Qwen/Qwen2.5-7B-Instruct` 
- ✅ **Integrator** uses your fine-tuned format-enforcer model
- ✅ **Dual-format**: After Phase 2, model supports both mortality and survival formats
- ✅ **VLLM compatible**: Works directly with VLLMWrapper (no modifications needed)
- ✅ **Same interface**: Drop-in replacement for base model
- ✅ **GPU memory**: Set `gpu_memory_utilization=0.4` in run_medrag_vllm.py for tensor parallelism

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
