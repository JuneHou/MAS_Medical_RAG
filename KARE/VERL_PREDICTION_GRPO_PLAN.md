# VERL GRPO Plan for KARE Mortality Prediction Task

## Executive Summary

This document provides a comprehensive plan for using VERL GRPO (Group Relative Policy Optimization) to fine-tune the integrator model for **improved mortality prediction accuracy** in the KARE debate system. Unlike the previous format enforcement plan (VERL_RL_PLAN.md), this plan focuses on the **downstream prediction task** - teaching the model to make better mortality predictions by rewarding correct probability estimates.

---

## 1. Problem Statement & Objectives

### Current Situation
The integrator agent in `mortality_debate_rag.py` produces two independent probability predictions:
- **Mortality probability**: `P(death in next visit)`
- **Survival probability**: `P(survival in next visit)`

These probabilities are used to make binary predictions (0=survival, 1=mortality) based on thresholds and comparisons.

### Training Objective
Use VERL GRPO to improve prediction accuracy by:
1. **Rewarding correct probability ranges** based on ground truth labels
2. **Maintaining the debate reasoning workflow** during training (similar patients + retrieval + multi-agent reasoning)
3. **Training on the train split** (not test split) with proper train/val separation

### Key Differences from Format Enforcement Plan
| Aspect | Format Enforcement (VERL_RL_PLAN.md) | Prediction Improvement (This Plan) |
|--------|--------------------------------------|-------------------------------------|
| **Objective** | Enforce `MORTALITY PROBABILITY: X.XX` format | Improve prediction accuracy |
| **Reward** | Binary (1=valid format, 0=invalid) | Probability-based (reward for correct ranges) |
| **Ground Truth** | Not used (format-only validation) | Essential (defines reward ranges) |
| **Data Generation** | Can skip debate (synthetic prompts) | Must run full debate pipeline |
| **Complexity** | Simple regex validation | Complex probability evaluation logic |

---

## 2. Data Preparation Analysis

### 2.1 Training Data Availability

**Available datasets in `/data/wang/junh/datasets/KARE/ehr_data`:**
```
✓ mimic3_mortality_samples_train.json    - Training split
✓ mimic3_mortality_samples_val.json      - Validation split  
✓ mimic3_mortality_samples_test.json     - Test split (996 samples)
✓ pateint_mimic3_mortality.json          - Full patient data
```

**Dataset sizes** (need to check):
- Train split: Unknown (typically ~70-80% of total)
- Val split: Unknown (typically ~10-15% of total)
- Test split: 996 samples (confirmed)

**Action Item**: Check train/val split sizes
```bash
python -c "import json; data=json.load(open('/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_train.json')); print(f'Train: {len(data)} samples')"
python -c "import json; data=json.load(open('/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_val.json')); print(f'Val: {len(data)} samples')"
```

### 2.2 Similar Patient Retrieval - Can We Use `improved_faiss_retrieval.py` Directly?

**Current `improved_faiss_retrieval.py` configuration:**
```python
DATASET = 'mimic3'
TASK = 'mortality'
BASE_PATH = "./data"

# Input paths
PATIENT_CONTEXT_PATH = f"{BASE_PATH}/patient_context/base_context_qwen/patient_contexts_{DATASET}_{TASK}.json"
PATIENT_EMBEDDINGS_PATH = f"{BASE_PATH}/patient_context/base_context_qwen/patient_embeddings_{DATASET}_{TASK}.pkl"
PATIENT_DATA_PATH = f"{BASE_PATH}/ehr_data/pateint_{DATASET}_{TASK}.json"  # Note: typo "pateint"

# Output path
OUTPUT_PATH = f"{BASE_PATH}/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_{DATASET}_{TASK}_improved.json"
```

**Analysis:**

✅ **YES, we can use it directly with modifications:**

1. **Path compatibility**: The script uses `pateint_mimic3_mortality.json` (note typo), which contains ALL patients (train + val + test)
2. **Pre-computed embeddings**: If `patient_embeddings_mimic3_mortality.pkl` contains embeddings for all patients, we can use it
3. **Context files**: If `patient_contexts_mimic3_mortality.json` contains contexts for all patients, we're good

⚠️ **Required modifications:**

```python
# Option 1: Generate similar patients for ALL patients (train + val + test)
# - Run once on full dataset
# - Use same output for all splits
# - Advantage: Consistent, no data leakage concerns (retrieval from full dataset is OK)

# Option 2: Generate separate similar patient files for train/val/test
# - Restrict candidate pool to same split
# - More restrictive, but ensures no "future information" leakage
# - Disadvantage: Smaller candidate pools
```

**Recommendation**: **Option 1** - Generate similar patients from the full dataset
- Similar patient retrieval is a **data augmentation step**, not a test-time prediction
- Retrieving similar patients from the full dataset (including test) during training is acceptable
- KARE paper uses this approach (retrieve from full dataset)

**Status: ✅ VERIFIED**

The file `patient_to_top_1_patient_contexts_mimic3_mortality_improved.json` **already exists** with:
- **9,717 patients** with similar patient contexts
- Train split has **7,730 patients**
- Val split has **~1,000 patients** (estimated)
- Test split has **996 patients**

✅ **All train/val/test patients are covered** (9,717 > 7,730 + 1,000 + 996)

**No action needed** - Use existing similar patient file directly!

### 2.3 Do We Need a Separate Data Generation Script?

**Answer: YES, but it needs a wrapper script**

**Why we need data generation:**
1. **Full debate pipeline required (Agents 1-3 only)**: We need to run rounds 1-3 to generate:
   - Agent 1 (Target Patient Analyst) reasoning
   - Agent 2 (Mortality Risk Assessor) reasoning  
   - Agent 3 (Protective Factor Analyst) reasoning
   - Retrieved documents (from MedRAG) for each agent
   - Complete debate history for integrator context

2. **Capture integrator prompts WITHOUT running integrator**: 
   - ✅ Run agents 1-3 to build context
   - ✅ Construct integrator prompts (mortality + survival)
   - ❌ **Do NOT run integrator to get predictions** (that's what GRPO will train)
   - Save prompts with ground truth for GRPO training

3. **Training data format**: VERL expects Parquet format with specific schema:
   ```python
   {
       'prompt': str,              # Full integrator prompt with ALL context
       'data_source': str,         # Task identifier (e.g., "kare_mortality_prediction")
       'ground_truth': int,        # 0 or 1 (survival or mortality)
       'extra_info': dict          # Patient ID, visit ID, etc.
   }
   ```

**Key clarification**: 
- We **run the debate pipeline** (agents 1-3) to generate context
- We **capture the integrator prompts** at the point where integrator would be called
- We **do NOT run integrator inference** - that's what GRPO learns to do better
- This is why we need a wrapper - to stop before integrator execution

**Can we use `mortality_debate_rag.py` directly?**

✅ **YES - use `debate_mortality_prediction()` method directly**

The `mortality_debate_rag.py` already has the complete pipeline in `debate_mortality_prediction()`:
- Runs 3-round debate (agents 1-3) with proper sequencing
- Handles batch processing for Round 2 (parallel agent execution)
- Generates all agent responses with retrieval
- Constructs integrator prompts internally
- This is the SAME code used in downstream deployment

**The wrapper script will**:
- Call `debate_mortality_prediction()` to run the full debate pipeline
- Extract debate_history from the result (first 3 rounds = agents 1-3)
- Use `_prepare_integrator_history()` to format debate history (same as integrator sees it)
- Construct the final integrator prompt using the agent prompts from the system
- **Capture prompts WITHOUT re-running integrator** - prompts are constructed from debate_history
- Convert to Parquet format for GRPO training

This approach ensures **100% consistency** between training data and deployment.

---

## 3. Data Generation Strategy

### 3.1 Proposed Data Generation Script: `generate_prediction_training_data.py`

**Architecture**:
```python
#!/usr/bin/env python3
"""
Generate GRPO training data for mortality prediction task.
Runs full debate pipeline on train/val splits and captures integrator prompts.
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import existing components
from kare_data_adapter import KAREDataAdapter
from mortality_debate_rag import MortalityDebateSystem

class PredictionDataGenerator:
    def __init__(self, 
                 data_split='train',  # 'train' or 'val'
                 model_name="Qwen/Qwen3-4B-Instruct-2507",
                 gpu_ids="6,7"):
        
        self.data_split = data_split
        self.data_adapter = KAREDataAdapter(base_path="./data")
        
        # Initialize debate system
        self.debate_system = MortalityDebateSystem(
            model_name=model_name,
            gpu_ids=gpu_ids,
            rag_enabled=True
        )
        
        # Load appropriate split
        self.load_data_split()
    
    def load_data_split(self):
        """Load train or val split from KARE dataset."""
        split_file = f"./data/ehr_data/mimic3_mortality_samples_{self.data_split}.json"
        with open(split_file, 'r') as f:
            self.split_data = json.load(f)
        print(f"Loaded {len(self.split_data)} samples from {self.data_split} split")
    
    def generate_training_examples(self, num_samples=None, output_dir="./data/grpo_training"):
        """
        Generate training examples by running debate on train/val split.
        
        Strategy:
        1. For each patient in split:
           - Run full 3-round debate (agents 1-3)
           - Capture TWO integrator prompts:
             * Mortality probability assessment prompt
             * Survival probability assessment prompt
           - Save prompts WITHOUT running integrator (we'll train integrator)
           - Store with ground truth label
        
        Output: Parquet file with schema:
        - prompt: Full integrator prompt (patient + debate + retrieval context)
        - data_source: "kare_mortality_prediction" or "kare_survival_prediction"  
        - ground_truth: 0 (survival) or 1 (mortality)
        - extra_info: {patient_id, visit_id, split, assessment_type}
        """
        
        if num_samples is None:
            num_samples = len(self.split_data)
        
        training_examples = []
        
        for i in tqdm(range(num_samples), desc=f"Generating {self.data_split} data"):
            patient_data = self.split_data[i]
            
            # Format patient data for debate system
            sample = self._format_patient_sample(patient_data)
            
            # Run debate (rounds 1-3 only, no integrator)
            debate_result = self._run_debate_rounds(sample)
            
            # Extract integrator prompts (mortality and survival)
            mortality_prompt = self._construct_integrator_prompt(
                sample, debate_result, assessment_type='mortality'
            )
            survival_prompt = self._construct_integrator_prompt(
                sample, debate_result, assessment_type='survival'
            )
            
            # Add mortality assessment example
            training_examples.append({
                'prompt': mortality_prompt,
                'data_source': 'kare_mortality_prediction',
                'ground_truth': sample['ground_truth'],
                'extra_info': {
                    'patient_id': sample['patient_id'],
                    'visit_id': sample['visit_id'],
                    'split': self.data_split,
                    'assessment_type': 'mortality'
                }
            })
            
            # Add survival assessment example
            training_examples.append({
                'prompt': survival_prompt,
                'data_source': 'kare_survival_prediction',
                'ground_truth': sample['ground_truth'],
                'extra_info': {
                    'patient_id': sample['patient_id'],
                    'visit_id': sample['visit_id'],
                    'split': self.data_split,
                    'assessment_type': 'survival'
                }
            })
        
        # Save to Parquet
        self._save_parquet(training_examples, output_dir)
    
    def _format_patient_sample(self, patient_data):
        """Convert raw patient data to debate system format."""
        # Use KAREDataAdapter logic to format
        base_patient_id = str(patient_data['patient_id'])
        visit_id = str(patient_data['visit_id'])
        
        # Get visit index from rolling context
        num_visits = max(
            len(patient_data.get('conditions', [[]])),
            len(patient_data.get('procedures', [[]])),
            len(patient_data.get('drugs', [[]]))
        )
        visit_index = num_visits - 1
        kare_patient_id = f"{base_patient_id}_{visit_index}"
        
        # Format patient context
        patient_context = self.data_adapter.format_patient_context(patient_data)
        
        # Get similar patients (from pre-generated file)
        similar_patients_data = self.data_adapter.similar_patients.get(kare_patient_id, {})
        positive_similars = self._format_similar_patients(similar_patients_data.get('positive', []))
        negative_similars = self._format_similar_patients(similar_patients_data.get('negative', []))
        
        return {
            'patient_id': kare_patient_id,
            'visit_id': visit_id,
            'patient_context': patient_context,
            'positive_similars': positive_similars,
            'negative_similars': negative_similars,
            'ground_truth': patient_data['label']
        }
    
    def _run_debate_rounds(self, sample):
        """
        Run 3-round debate (agents 1-3) WITHOUT integrator.
        
        Returns:
            debate_history: List of agent responses
            retrieved_docs: Documents retrieved during debate
        """
        # Call debate system's internal methods to run agents 1-3
        # WITHOUT running integrator (agent 4)
        
        debate_history = []
        
        # Round 1: Target Patient Analyst
        agent1_response = self.debate_system._agent_turn(
            role='target_patient_analyst',
            patient_context=sample['patient_context'],
            similar_patients={
                'positive': sample['positive_similars'],
                'negative': sample['negative_similars']
            },
            debate_history=[],
            patient_id=sample['patient_id']
        )
        debate_history.append(agent1_response)
        
        # Round 2: Mortality Risk Assessor  
        agent2_response = self.debate_system._agent_turn(
            role='mortality_risk_assessor',
            patient_context=sample['patient_context'],
            similar_patients={
                'positive': sample['positive_similars'],
                'negative': sample['negative_similars']
            },
            debate_history=debate_history,
            patient_id=sample['patient_id']
        )
        debate_history.append(agent2_response)
        
        # Round 3: Protective Factor Analyst
        agent3_response = self.debate_system._agent_turn(
            role='protective_factor_analyst',
            patient_context=sample['patient_context'],
            similar_patients={
                'positive': sample['positive_similars'],
                'negative': sample['negative_similars']
            },
            debate_history=debate_history,
            patient_id=sample['patient_id']
        )
        debate_history.append(agent3_response)
        
        return {
            'debate_history': debate_history,
            'patient_context': sample['patient_context'],
            'similar_patients': {
                'positive': sample['positive_similars'],
                'negative': sample['negative_similars']
            }
        }
    
    def _construct_integrator_prompt(self, sample, debate_result, assessment_type='mortality'):
        """
        Construct the exact integrator prompt that will be used during GRPO training.
        
        This should match the prompt structure in mortality_debate_rag.py's
        _execute_integrator_attempt() method.
        """
        
        # Get system prompt for integrator
        if assessment_type == 'mortality':
            system_prompt = self.debate_system.agent_prompts['balanced_clinical_integrator_mortality']
        else:
            system_prompt = self.debate_system.agent_prompts['balanced_clinical_integrator_survival']
        
        # Format debate history
        history_text = self.debate_system._prepare_integrator_history(debate_result['debate_history'])
        
        # Construct full prompt
        prompt_parts = [
            "## Task Description:",
            self.data_adapter.get_task_description(),
            "",
            "## Target Patient EHR:",
            sample['patient_context'],
            "",
            "## Similar Patients (Positive Outcome):",
            debate_result['similar_patients']['positive'],
            "",
            "## Similar Patients (Negative Outcome):",
            debate_result['similar_patients']['negative'],
            "",
            history_text,
            "",
            "## Your Assessment:",
            system_prompt
        ]
        
        full_prompt = "\n".join(prompt_parts)
        
        return full_prompt
    
    def _format_similar_patients(self, similar_list):
        """Format similar patients block."""
        if similar_list and similar_list[0] != "None":
            return "\n\n".join(similar_list)
        else:
            return "No similar patients available."
    
    def _save_parquet(self, examples, output_dir):
        """Save training examples to Parquet format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(examples)
        
        # Save with proper schema
        output_file = output_path / f"{self.data_split}.parquet"
        df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        print(f"Saved {len(examples)} examples to {output_file}")
        print(f"  - {len([e for e in examples if e['data_source'] == 'kare_mortality_prediction'])} mortality prompts")
        print(f"  - {len([e for e in examples if e['data_source'] == 'kare_survival_prediction'])} survival prompts")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--gpus', type=str, default='6,7')
    parser.add_argument('--output_dir', type=str, default='./data/grpo_training')
    args = parser.parse_args()
    
    generator = PredictionDataGenerator(
        data_split=args.split,
        model_name=args.model,
        gpu_ids=args.gpus
    )
    
    generator.generate_training_examples(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Generate training data (train split)
python generate_prediction_training_data.py --split train --gpus 6,7

# Generate validation data (val split)
python generate_prediction_training_data.py --split val --gpus 6,7

# Test on subset first
python generate_prediction_training_data.py --split train --num_samples 100 --gpus 6,7
```

**Estimated time**:
- Full debate pipeline: ~2-3 minutes per patient (3 agent rounds + retrieval)
- Train split (~3000 patients): ~100-150 GPU hours (~4-6 days on 2 GPUs)
- **Recommendation**: Start with 500 samples, evaluate, then scale

---

## 4. Reward Function Design

### 4.1 Reward Logic - Probability Range Based on Ground Truth

**Challenge**: How to reward probabilistic outputs when we only have binary labels (0 or 1)?

**Proposed solution**: Define **acceptable probability ranges** based on ground truth

#### 4.1.1 Scenario-Based Reward Design (REVISED)

**Key insight**: Since we generate TWO prompts per patient (mortality + survival), we should reward BOTH assessments jointly to ensure coherent reasoning.

**For Ground Truth = 0 (Survival):**

| Assessment Type | Predicted Probability | Interpretation | Reward |
|----------------|----------------------|----------------|---------|
| **Mortality** | `< 0.4` | Correctly low mortality risk | +1.0 |
| **Mortality** | `≥ 0.4 and < 0.7` | Uncertain/ambiguous | 0.0 |
| **Mortality** | `≥ 0.7` | Wrongly predicts mortality | -1.0 |
| **Survival** | `≥ 0.6` | Correctly high survival prob | +1.0 |
| **Survival** | `≥ 0.3 and < 0.6` | Uncertain/ambiguous | 0.0 |
| **Survival** | `< 0.3` | Wrongly predicts mortality | -1.0 |

**For Ground Truth = 1 (Mortality):**

| Assessment Type | Predicted Probability | Interpretation | Reward |
|----------------|----------------------|----------------|---------|
| **Mortality** | `≥ 0.7` | Correctly high mortality risk | +1.0 |
| **Mortality** | `≥ 0.4 and < 0.7` | Uncertain/ambiguous | 0.0 |
| **Mortality** | `< 0.4` | Wrongly predicts survival | -1.0 |
| **Survival** | `< 0.3` | Correctly low survival prob | +1.0 |
| **Survival** | `≥ 0.3 and < 0.6` | Uncertain/ambiguous | 0.0 |
| **Survival** | `≥ 0.6` | Wrongly predicts survival | -1.0 |

**Key design decisions:**
- ✅ **Symmetric rewards**: +1 for correct, 0 for uncertain, -1 for wrong
- ✅ **Negative rewards**: Penalize confidently wrong predictions (encourages calibration)
- ✅ **Independent assessments**: Each prompt (mortality/survival) gets separate reward
- ✅ **Ambiguity zone**: 0.3-0.6 for survival, 0.4-0.7 for mortality (no penalty/reward)
- ✅ **No constraint on sum**: Mortality + survival probabilities are independent

**Rationale for negative rewards:**
- Discourage overconfident wrong predictions (e.g., 0.9 mortality for survivor)
- Encourage model to be uncertain when evidence is mixed
- GRPO will learn to avoid high-penalty regions

#### 4.1.2 Alternative: Calibration-Based Reward

**More sophisticated approach** (optional, for later iteration):

```python
def compute_calibration_reward(mort_prob, surv_prob, ground_truth):
    """
    Reward based on probability calibration.
    
    Idea: Model should express confidence that correlates with accuracy
    - High confidence + correct = high reward
    - Low confidence + correct = moderate reward  
    - High confidence + wrong = zero/negative reward
    - Low confidence + wrong = small penalty
    """
    
    # Determine prediction from probabilities
    if mort_prob > surv_prob:
        prediction = 1
        confidence = mort_prob
    else:
        prediction = 0
        confidence = surv_prob
    
    # Reward structure
    if prediction == ground_truth:
        # Correct prediction: reward scales with confidence
        reward = confidence
    else:
        # Wrong prediction: penalty scales with confidence
        reward = 0.0  # Or negative: -confidence
    
    return reward
```

**Comparison**:
| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Range-based** | Simple, interpretable, aligns with thresholds | Arbitrary ranges, may not reflect uncertainty well | **Start with this** |
| **Calibration-based** | Encourages well-calibrated probabilities, continuous reward | More complex, requires careful tuning | Use for refinement |

### 4.2 Implementation: `kare_prediction_reward.py`

```python
#!/usr/bin/env python3
"""
Reward function for KARE mortality prediction GRPO training.
"""

import re
from typing import Dict, Any, Optional

# Regex patterns for extracting probabilities
MORT_PROB_RE = re.compile(
    r'MORTALITY PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)
SURV_PROB_RE = re.compile(
    r'SURVIVAL PROBABILITY:\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)

def extract_probabilities(solution_str: str) -> Dict[str, Optional[float]]:
    """
    Extract mortality and survival probabilities from model output.
    
    Returns:
        dict with 'mortality_prob' and 'survival_prob' (None if not found)
    """
    result = {
        'mortality_prob': None,
        'survival_prob': None
    }
    
    # Extract mortality probability
    mort_match = MORT_PROB_RE.search(solution_str)
    if mort_match:
        try:
            mort_val = float(mort_match.group(1))
            if 0.0 <= mort_val <= 1.0:
                result['mortality_prob'] = mort_val
        except ValueError:
            pass
    
    # Extract survival probability
    surv_match = SURV_PROB_RE.search(solution_str)
    if surv_match:
        try:
            surv_val = float(surv_match.group(1))
            if 0.0 <= surv_val <= 1.0:
                result['survival_prob'] = surv_val
        except ValueError:
            pass
    
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: int,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute reward for mortality/survival prediction.
    
    Args:
        data_source: 'kare_mortality_prediction' or 'kare_survival_prediction'
        solution_str: Model's generated response
        ground_truth: 0 (survival) or 1 (mortality)
        extra_info: Additional metadata (patient_id, etc.)
    
    Returns:
        reward: Float in [0.0, 1.0]
    """
    
    # Extract probabilities
    probs = extract_probabilities(solution_str)
    
    # Get assessment type from data_source
    if data_source == 'kare_mortality_prediction':
        assessment_type = 'mortality'
        primary_prob = probs['mortality_prob']
    elif data_source == 'kare_survival_prediction':
        assessment_type = 'survival'
        primary_prob = probs['survival_prob']
def compute_probability_reward(
    predicted_prob: float,
    ground_truth: int,
    assessment_type: str
) -> float:
    """
    Compute reward based on predicted probability and ground truth.
    
    REVISED Reward design (symmetric with negative penalties):
    - For survival (GT=0): Reward low mortality / high survival
    - For mortality (GT=1): Reward high mortality / low survival
    - Use +1 for correct, 0 for uncertain, -1 for wrong
    """
    
    if assessment_type == 'mortality':
        # Mortality probability assessment
        if ground_truth == 0:
            # Patient survives - should predict LOW mortality
            if predicted_prob < 0.4:
                return 1.0  # Correct: low mortality
            elif predicted_prob < 0.7:
                return 0.0  # Uncertain/ambiguous
            else:  # >= 0.7
                return -1.0  # Wrong: confidently predicts mortality
        
        else:  # ground_truth == 1
            # Patient dies - should predict HIGH mortality
            if predicted_prob >= 0.7:
                return 1.0  # Correct: high mortality
            elif predicted_prob >= 0.4:
                return 0.0  # Uncertain/ambiguous
            else:  # < 0.4
                return -1.0  # Wrong: confidently predicts survival
    
    elif assessment_type == 'survival':
        # Survival probability assessment
        if ground_truth == 0:
            # Patient survives - should predict HIGH survival
            if predicted_prob >= 0.6:
                return 1.0  # Correct: high survival
            elif predicted_prob >= 0.3:
                return 0.0  # Uncertain/ambiguous
            else:  # < 0.3
                return -1.0  # Wrong: confidently predicts mortality
        
        else:  # ground_truth == 1
            # Patient dies - should predict LOW survival
            if predicted_prob < 0.3:
                return 1.0  # Correct: low survival
            elif predicted_prob < 0.6:
                return 0.0  # Uncertain/ambiguous
            else:  # >= 0.6
                return -1.0  # Wrong: confidently predicts survival
    
    return 0.0  # Fallback  # Moderate correct prediction
            elif predicted_prob > 0.5:
                return 0.5  # Weak correct prediction
            else:
                return 0.0  # Incorrect (predicts mortality)
        
        else:  # ground_truth == 1
            # Patient dies - should predict LOW survival
            if predicted_prob < 0.3:
                return 1.0  # Strong correct prediction
            elif predicted_prob < 0.4:
                return 0.8  # Moderate correct prediction
            elif predicted_prob < 0.5:
                return 0.5  # Weak correct prediction
            else:
                return 0.0  # Incorrect (predicts survival)
    
    return 0.0  # Fallback


def compute_score_with_calibration(
    data_source: str,
    solution_str: str,
    ground_truth: int,
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Alternative reward with calibration metrics.
    
    Returns:
        dict with 'score', 'format_score', 'prediction_score', 'calibration_score'
    """
    
    # Extract probabilities
    probs = extract_probabilities(solution_str)
    
    # Format score (did model output valid probabilities?)
    format_score = 0.0
    if data_source == 'kare_mortality_prediction':
        if probs['mortality_prob'] is not None:
            format_score = 1.0
        primary_prob = probs['mortality_prob']
    elif data_source == 'kare_survival_prediction':
        if probs['survival_prob'] is not None:
            format_score = 1.0
        primary_prob = probs['survival_prob']
    else:
        return {'score': 0.0, 'format_score': 0.0, 'prediction_score': 0.0, 'calibration_score': 0.0}
    
    if primary_prob is None:
        return {'score': 0.0, 'format_score': 0.0, 'prediction_score': 0.0, 'calibration_score': 0.0}
    
    # Prediction score (is prediction correct?)
    assessment_type = 'mortality' if data_source == 'kare_mortality_prediction' else 'survival'
    prediction_score = compute_probability_reward(primary_prob, ground_truth, assessment_type)
    
    # Calibration score (confidence matches accuracy)
    # For now, use same as prediction_score (can be refined)
    calibration_score = prediction_score
    
    # Overall score (weighted combination)
    score = 0.2 * format_score + 0.8 * prediction_score
    
    return {
        'score': score,
        'format_score': format_score,
        'prediction_score': prediction_score,
        'calibration_score': calibration_score
    }


# Test the reward function
if __name__ == "__main__":
    # Test case 1: Survival patient (GT=0), correct mortality prediction
    test_response_1 = """
    Clinical reasoning...
    
    MORTALITY PROBABILITY: 0.25
    """
# Test the reward function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing REVISED Reward Function (with negative penalties)")
    print("=" * 60)
    
    # Test case 1: Survival patient (GT=0), correct mortality prediction
    test_response_1 = """
    Clinical reasoning...
    
    MORTALITY PROBABILITY: 0.25
    """
    reward_1 = compute_score('kare_mortality_prediction', test_response_1, ground_truth=0)
    print(f"Test 1 (GT=0, mort_prob=0.25): Reward = {reward_1}")  # Should be +1.0
    
    # Test case 2: Mortality patient (GT=1), correct survival prediction
    test_response_2 = """
    Clinical reasoning...
    
    SURVIVAL PROBABILITY: 0.15
    """
    reward_2 = compute_score('kare_survival_prediction', test_response_2, ground_truth=1)
    print(f"Test 2 (GT=1, surv_prob=0.15): Reward = {reward_2}")  # Should be +1.0
    
    # Test case 3: Survival patient (GT=0), WRONG prediction (high mortality)
    test_response_3 = """
    Clinical reasoning...
    
    MORTALITY PROBABILITY: 0.75
    """
    reward_3 = compute_score('kare_mortality_prediction', test_response_3, ground_truth=0)
    print(f"Test 3 (GT=0, mort_prob=0.75): Reward = {reward_3}")  # Should be -1.0 (NEGATIVE)
    
    # Test case 4: Survival patient (GT=0), uncertain prediction
    test_response_4 = """
    Clinical reasoning...
    
    MORTALITY PROBABILITY: 0.55
    """
    reward_4 = compute_score('kare_mortality_prediction', test_response_4, ground_truth=0)
    print(f"Test 4 (GT=0, mort_prob=0.55 - uncertain): Reward = {reward_4}")  # Should be 0.0
    
    # Test case 5: Survival patient (GT=0), correct survival prediction
    test_response_5 = """
    Clinical reasoning...
    
    SURVIVAL PROBABILITY: 0.75
    """
    reward_5 = compute_score('kare_survival_prediction', test_response_5, ground_truth=0)
    print(f"Test 5 (GT=0, surv_prob=0.75): Reward = {reward_5}")  # Should be +1.0
    
    # Test case 6: Survival patient (GT=0), WRONG survival prediction (low survival)
    test_response_6 = """
    Clinical reasoning...
    
    SURVIVAL PROBABILITY: 0.15
    """
    reward_6 = compute_score('kare_survival_prediction', test_response_6, ground_truth=0)
    print(f"Test 6 (GT=0, surv_prob=0.15 - wrong): Reward = {reward_6}")  # Should be -1.0 (NEGATIVE)
    
    # Test case 7: Missing probability (format error)
    test_response_7 = """
    Clinical reasoning without probability output
    """
    reward_7 = compute_score('kare_mortality_prediction', test_response_7, ground_truth=0)
    print(f"Test 7 (Missing prob): Reward = {reward_7}")  # Should be 0.0
    
    print("\n" + "=" * 60)
    print("Expected behavior:")
    print("  +1.0 : Correct confident prediction")
    print("   0.0 : Uncertain/ambiguous prediction OR format error")
    print("  -1.0 : Wrong confident prediction (PENALTY)")
    print("=" * 60)
```bash
#!/bin/bash

# Model configuration
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"  # Same as current integrator
OUTPUT_DIR="/data/wang/junh/githubs/Debate/KARE/grpo_checkpoints/prediction_task"

# Data paths
TRAIN_DATA="/data/wang/junh/githubs/Debate/KARE/data/grpo_training/train.parquet"
VAL_DATA="/data/wang/junh/githubs/Debate/KARE/data/grpo_training/val.parquet"

# GPU configuration
export CUDA_VISIBLE_DEVICES="6,7"
NPROC_PER_NODE=2

# Reward function
REWARD_FUNCTION="kare_prediction_reward.compute_score"

# Training with GRPO
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.01 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    algorithm.adv_estimator=grpo \
    trainer.logger=['console','tracking'] \
    trainer.project_name='kare_prediction_grpo' \
    trainer.experiment_name='qwen3_4b_mortality_prediction' \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.default_hdfs_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    +reward_model.reward_fn_path=kare_prediction_reward \
    +reward_model.compute_score_fn=compute_score
```

**Key hyperparameters explanation:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr=1e-6` | Very small | Preserve pretrained knowledge, careful fine-tuning |
| `rollout.n=4` | 4 completions per prompt | GRPO requirement: sample multiple outputs |
| `temperature=0.8` | Higher than inference | Encourage exploration during training |
| `kl_coef=0.01` | Small KL regularization | Allow learning while staying close to base model |
| `max_prompt_length=8192` | Large context | Accommodate full debate history |
| `max_response_length=2048` | Moderate | Allow detailed reasoning + probability |
| `total_epochs=3` | Few epochs | Prevent overfitting on small dataset |

### 5.2 Alternative: LoRA Training (Resource-efficient)

If full fine-tuning is too expensive:

```bash
# Add LoRA configuration
actor_rollout_ref.actor.use_lora=True \
actor_rollout_ref.actor.lora_rank=16 \
actor_rollout_ref.actor.lora_alpha=32 \
actor_rollout_ref.actor.lora_dropout=0.05 \
actor_rollout_ref.actor.lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'
```

---

## 6. Integration & Evaluation

### 6.1 Load GRPO-Trained Model

Modify `mortality_debate_rag.py`:

```python
class MortalityDebateSystem:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 integrator_rl_checkpoint: str = None,  # NEW PARAMETER
                 ...):
        
        # Initialize main model for agents 1-3 (unchanged)
        self.llm = VLLMWrapper(model_name=model_name)
        
        # Initialize integrator model
        if integrator_rl_checkpoint:
            # Load GRPO-trained model for integrator
            print(f"Loading GRPO-trained integrator from: {integrator_rl_checkpoint}")
            self.integrator_llm = VLLMWrapper(model_name=integrator_rl_checkpoint)
        elif self.integrator_model_name != model_name:
            # Use different base model for integrator
            self.integrator_llm = VLLMWrapper(model_name=self.integrator_model_name)
        else:
            # Use same model for all agents
            self.integrator_llm = self.llm
```

### 6.2 Evaluation Script

```python
#!/usr/bin/env python3
"""
Evaluate GRPO-trained integrator on validation/test set.
"""

import json
from pathlib import Path
from run_kare_debate_mortality import run_kare_debate_evaluation

def evaluate_grpo_model(
    checkpoint_path: str,
    eval_split: str = 'val',
    num_samples: int = None,
    output_dir: str = None
):
    """
    Evaluate GRPO-trained model on validation or test set.
    
    Args:
        checkpoint_path: Path to GRPO checkpoint
        eval_split: 'val' or 'test'
        num_samples: Number of samples to evaluate
        output_dir: Output directory for results
    """
    
    if output_dir is None:
        checkpoint_name = Path(checkpoint_path).name
        output_dir = f"./results/grpo_eval/{checkpoint_name}_{eval_split}"
    
    # Run evaluation with GRPO checkpoint
    results = run_kare_debate_evaluation(
        start_idx=0,
        num_samples=num_samples,
        output_path=f"{output_dir}/results.json",
        model_name="Qwen/Qwen3-4B-Instruct-2507",  # Base model for agents 1-3
        integrator_rl_checkpoint=checkpoint_path,   # GRPO model for integrator
        gpu_ids="6,7",
        debate_mode="rag"
    )
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to GRPO checkpoint')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()
    
    evaluate_grpo_model(
        checkpoint_path=args.checkpoint,
        eval_split=args.split,
        num_samples=args.num_samples
    )
```

### 6.3 Evaluation Metrics

**Primary metrics:**
1. **Accuracy**: Binary classification accuracy (0 vs 1)
2. **F1 Score**: Balanced measure for imbalanced data
3. **Precision/Recall**: For mortality class specifically
4. **AUROC**: Area under ROC curve (using probability outputs)
5. **Calibration**: Expected calibration error (ECE)

**Comparison baselines:**
- Baseline: Integrator without GRPO training
- Oracle: Human expert predictions (if available)
- Simple: Majority class predictor

**Success criteria:**
- Accuracy improvement: ≥2% over baseline
- AUROC improvement: ≥0.03 over baseline
- F1 improvement: ≥3% over baseline

---

## 7. Implementation Timeline & Risk Assessment
### 8.1 Confirmed Decisions

✅ **Use train/val splits** (not test split) for training data generation  
✅ **Similar patients file READY** (`patient_to_top_1_patient_contexts_mimic3_mortality_improved.json` with 9,717 patients covers all splits)  
✅ **Run debate rounds 1-3 ONLY** (capture integrator prompts, do NOT run integrator inference)  
✅ **Two training examples per patient** (mortality + survival assessments evaluated separately)  
✅ **Symmetric reward function** (+1 correct, 0 uncertain, -1 wrong) - **REVISED**  
✅ **Negative penalties** to discourage overconfident wrong predictions - **NEW**  
✅ **Independent probabilities** (mortality and survival not required to sum to 1.0)  
✅ **Start with 500 samples** (validate approach before scaling to 7,730 train samples) logic thoroughly |
| Week 3 | Run initial GRPO training (small) | 2-3 days | Debug training pipeline |
| Week 3-4 | Scale up training (full dataset) | 4-6 days (GPU) | Small training works |
| Week 4 | Evaluation & analysis | 2-3 days | Model checkpoint ready |
### Immediate Actions

1. **✅ DONE: Data splits verified**:
   - Train: 7,730 samples
   - Val: ~1,000 samples (need to verify exact count)
   - Test: 996 samples
   ```bash
   python -c "import json; print(f'Val: {len(json.load(open(\"/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_val.json\")))} samples')"
   ```

2. **✅ DONE: Similar patient file verified**:
   - File exists: `data/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json`
   - Contains: 9,717 patients (covers all train/val/test)
   - **No action needed** - ready to use!

3. **Implement `generate_prediction_training_data.py`** (see Section 3.1)
   - Key: Run agents 1-3, capture integrator prompts, do NOT run integrator

4. **Implement `kare_prediction_reward.py`** with REVISED reward logic (see Section 4.2)
   - Use +1 / 0 / -1 rewards (not 1.0 / 0.8 / 0.5 / 0.0)
   - Include negative penalties for wrong confident predictions

5. **Test data generation on 10 samples**:
   ```bash
   python generate_prediction_training_data.py --split train --num_samples 10 --gpus 6,7
   ```

6. **Test reward function**:
   ```bash
   python kare_prediction_reward.py  # Run built-in tests
   ```
### 8.2 Open Questions

❓ **Train/val split sizes** - Need to check actual sizes in data files  
❓ **Full fine-tuning vs LoRA** - Depends on computational budget  
❓ **Reward threshold tuning** - May need adjustment based on initial results  
❓ **Number of GRPO epochs** - Balance between convergence and overfitting

---

## 9. Next Steps (This Week)

### Immediate Actions

1. **Check data splits**:
   ```bash
   python -c "import json; print(len(json.load(open('./data/ehr_data/mimic3_mortality_samples_train.json'))))"
   python -c "import json; print(len(json.load(open('./data/ehr_data/mimic3_mortality_samples_val.json'))))"
   ```

2. **Verify similar patient files exist**:
   ```bash
   ls -lh ./data/patient_context/base_context_qwen/patient_contexts_mimic3_mortality.json
   ls -lh ./data/patient_context/base_context_qwen/patient_embeddings_mimic3_mortality.pkl
   ```

3. **Run FAISS retrieval if needed**:
   ```bash
   cd /data/wang/junh/githubs/Debate/KARE
   python improved_faiss_retrieval.py
   ```

4. **Implement `generate_prediction_training_data.py`** (see Section 3.1)

5. **Implement `kare_prediction_reward.py`** (see Section 4.2)

6. **Test data generation on 10 samples**:
   ```bash
   python generate_prediction_training_data.py --split train --num_samples 10 --gpus 6,7
   ```

---

## 10. Conclusion

This plan provides a comprehensive approach to using VERL GRPO for improving mortality prediction accuracy in the KARE debate system. Key differences from the format enforcement plan:

- **Focus on prediction accuracy** (not just format)
- **Full debate pipeline required** (cannot use synthetic prompts)
- **Probability-based rewards** (not binary format validation)
- **Train on train split** (not test split)
- **More complex data generation** (requires running full 3-round debate)

The approach is feasible but requires significant computational resources for data generation (~100-150 GPU hours for full training set). Starting with 500 samples is recommended to validate the approach before scaling up.
