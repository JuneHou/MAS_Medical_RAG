# GRPO Training Data Generation Plan for Mortality Integrator

## Overview
Generate training prompts for VERL GRPO training to enforce format compliance in the mortality integrator's outputs.

**Target Format**: `MORTALITY PROBABILITY: X.XX (0.00 to 1.00)`

---

## Data Sources

### 1. Patient EHR Context (Primary Context)
**Location**: `/data/wang/junh/githubs/Debate/KARE/data/ehr_data/mimic3_mortality_samples_test.json`

**Structure**:
```json
{
    "patient_id": "10117",
    "visit_id": "0",
    "conditions": [["pneumonia", "septicemia", ...]],
    "procedures": [["hemodialysis", ...]],
    "drugs": [["vitamin b12", ...]],
    "label": 0  // 0=survival, 1=mortality
}
```

**Parsing Strategy**:
- Use `KAREDataAdapter` from `kare_data_adapter.py`
- Call `format_patient_context(patient_data)` to get formatted EHR context
- This provides temporal rolling visit format as used in debate system

### 2. Debate History Context
**Locations** (Priority order):
1. **Primary**: `/data/wang/junh/githubs/Debate/KARE/results/arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8/`
2. **Fallback**: `/data/wang/junh/githubs/Debate/KARE/results/fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8/`

**File Pattern**: `debate_responses_{patient_id}_{visit_id}.log`

**Content Structure** (from log analysis):
```
TARGET_PATIENT_ANALYST response:
- Clinical summary
- Risk factors
- Protective factors
- Initial prediction (boxed{0} or boxed{1})

POSITIVE_SIMILAR_COMPARATOR response:
- Common mortality patterns
- Risk factor analysis from similar fatal cases

NEGATIVE_SIMILAR_COMPARATOR response:
- Common survival patterns
- Protective factor analysis from similar survival cases
```

**Parsing Strategy**:
- Extract sections between agent role headers
- Parse each agent's response (target analyst, risk assessor, protective analyst)
- Combine into `Previous Debate Analysis` section
- Handle case where integrator prediction is None → use fallback logs

### 3. Retrieved Medical Evidence
**Locations**: Same directories as debate logs
**File Pattern**: `retrieve_mortality_assessment_{patient_id}_{visit_id}.json`

**Structure**:
```json
{
    "patient_id": "mortality_assessment_34_0",
    "query": "acute myocardial infarction mortality risk factors",
    "retrieved_documents": [
        {
            "id": 1,
            "title": "...",
            "content": "...",
            "score": 66.15,
            "source": "medcorp"
        }
    ]
}
```

**Parsing Strategy**:
- Load JSON file
- Extract `query` (the retrieval query used)
- Extract top 8-16 documents
- Format as: `[{id}] {title}\n{content}\n\n`

---

## Prompt Template Structure

### Final Prompt Format
```
You are a medical AI Clinical Assistant analyzing MORTALITY risk for the NEXT hospital visit.

Available tools:
- retrieve(query): Retrieve medical evidence for your mortality risk assessment

Instructions:
1) Based on the patient's specific conditions, call retrieve() with a custom query about their mortality risk factors (e.g., "sepsis mortality risk elderly patients" or "respiratory failure cardiac arrest prognosis")
2) Review all available information and the retrieved evidence
3) Focus ONLY on factors that increase death probability 
4) Be conservative: mortality is rare, so strong evidence is needed for high mortality probability

Provide comprehensive clinical reasoning and end with:
MORTALITY PROBABILITY: X.XX (0.00 to 1.0)

## Target Patient EHR Context ##
{patient_ehr_context}

## Previous Debate Analysis ##
{debate_history_context}

You called: retrieve("{retrieval_query}")

Retrieved Evidence:
{retrieved_documents_context}

Now provide your complete mortality probability assessment based on the retrieved evidence:
```

### Component Details

#### 1. Patient EHR Context (`{patient_ehr_context}`)
```
Patient ID: 10117

Visit 0:
Conditions:
  - pneumonia (new)
  - septicemia (new)
  - respiratory failure (new)
  ...

Procedures:
  - hemodialysis (new)
  ...

Medications:
  - vitamin b12 and folic acid (new)
  - beta-lactam antibacterials (new)
  ...

Ground Truth Label: 0 (survival)
```

#### 2. Debate History Context (`{debate_history_context}`)
```
TARGET_PATIENT_ANALYST:
[Extract from log: clinical summary, risk factors, protective factors]

MORTALITY_RISK_ASSESSOR:
[Extract from log: risk factor analysis from similar fatal cases]

PROTECTIVE_FACTOR_ANALYST:
[Extract from log: protective factor analysis from similar survival cases]
```

#### 3. Retrieval Query (`{retrieval_query}`)
```
acute myocardial infarction mortality risk factors
```

#### 4. Retrieved Documents Context (`{retrieved_documents_context}`)
```
[1] Acute Myocardial Infarction -- Prognosis
Medical management following an AMI is crucial for improving long-term outcomes...

[2] Acute Myocardial Infarction -- Etiology -- Nonmodifiable Risk Factors
Sex Age Family history Male pattern baldness...

[3] Myocardial Infarction -- Prognosis
Despite many advances in treatment, acute MI still carries a mortality rate of 5-30%...
```

---

## Data Generation Pipeline

### Step 1: Sample Selection
```python
# Load test data
adapter = KAREDataAdapter(base_path="/data/wang/junh/githubs/Debate/KARE/data")
test_samples = adapter.test_data

# Target: 500 samples for initial training
# Strategy: Select samples with successful integrator runs
selected_samples = []
for sample in test_samples[:996]:
    patient_id = sample['patient_id']
    visit_id = sample['visit_id']
    
    # Check if debate log exists and integrator provided probability
    if has_valid_integrator_response(patient_id, visit_id):
        selected_samples.append(sample)
    
    if len(selected_samples) >= 500:
        break
```

### Step 2: Context Extraction
```python
def extract_patient_context(sample):
    """Extract formatted EHR context."""
    adapter = KAREDataAdapter(...)
    return adapter.format_patient_context(sample)

def extract_debate_history(patient_id, visit_id, log_dirs):
    """Extract debate history from log files."""
    log_file = find_debate_log(patient_id, visit_id, log_dirs)
    
    # Parse log file
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract agent responses
    target_analyst = extract_section(log_content, "TARGET_PATIENT_ANALYST")
    risk_assessor = extract_section(log_content, "POSITIVE_SIMILAR_COMPARATOR")
    protective_analyst = extract_section(log_content, "NEGATIVE_SIMILAR_COMPARATOR")
    
    # Format for prompt
    return f"""TARGET_PATIENT_ANALYST:
{target_analyst}

MORTALITY_RISK_ASSESSOR:
{risk_assessor}

PROTECTIVE_FACTOR_ANALYST:
{protective_analyst}"""

def extract_retrieved_docs(patient_id, visit_id, log_dirs):
    """Extract retrieved medical evidence."""
    retrieval_file = find_retrieval_log(patient_id, visit_id, log_dirs)
    
    with open(retrieval_file, 'r') as f:
        data = json.load(f)
    
    query = data['query']
    docs = data['retrieved_documents'][:16]  # Top 16 docs
    
    # Format documents
    docs_text = ""
    for doc in docs:
        docs_text += f"[{doc['id']}] {doc['title']}\n{doc['content']}\n\n"
    
    return query, docs_text
```

### Step 3: Prompt Assembly
```python
def create_training_prompt(sample, debate_history, retrieval_query, retrieved_docs):
    """Assemble final training prompt."""
    
    system_prompt = """You are a medical AI Clinical Assistant analyzing MORTALITY risk for the NEXT hospital visit.

Available tools:
- retrieve(query): Retrieve medical evidence for your mortality risk assessment

Instructions:
1) Based on the patient's specific conditions, call retrieve() with a custom query about their mortality risk factors
2) Review all available information and the retrieved evidence
3) Focus ONLY on factors that increase death probability 
4) Be conservative: mortality is rare, so strong evidence is needed for high mortality probability

Provide comprehensive clinical reasoning and end with:
MORTALITY PROBABILITY: X.XX (0.00 to 1.0)"""

    patient_context = extract_patient_context(sample)
    
    full_prompt = f"""{system_prompt}

## Target Patient EHR Context ##
{patient_context}

## Previous Debate Analysis ##
{debate_history}

You called: retrieve("{retrieval_query}")

Retrieved Evidence:
{retrieved_docs}

Now provide your complete mortality probability assessment based on the retrieved evidence:"""

    return full_prompt
```

### Step 4: Parquet Generation
```python
import pandas as pd
from datasets import Dataset

training_data = []

for sample in selected_samples:
    patient_id = sample['patient_id']
    visit_id = sample['visit_id']
    ground_truth = sample['label']  # 0 or 1
    
    # Extract all components
    patient_context = extract_patient_context(sample)
    debate_history = extract_debate_history(patient_id, visit_id, log_dirs)
    retrieval_query, retrieved_docs = extract_retrieved_docs(patient_id, visit_id, log_dirs)
    
    # Assemble prompt
    prompt = create_training_prompt(sample, debate_history, retrieval_query, retrieved_docs)
    
    # Create training example
    training_data.append({
        'prompt': prompt,
        'data_source': 'kare_integrator_mortality_format',
        'ground_truth': str(ground_truth),
        'ability': 'medical_format',
        'reward_model': {'style': 'rule'},
        'extra_info': {
            'patient_id': f"{patient_id}_{visit_id}",
            'context_length': len(prompt.split()),
            'ground_truth_label': ground_truth
        }
    })

# Create dataset
dataset = Dataset.from_list(training_data)

# Split train/val (80/20)
dataset = dataset.shuffle(seed=42)
train_size = int(len(dataset) * 0.8)
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))

# Save to parquet
train_dataset.to_parquet('data/rl_training/train.parquet')
val_dataset.to_parquet('data/rl_training/val.parquet')
```

---

## File Organization

### Input Files
```
/data/wang/junh/githubs/Debate/KARE/
├── data/
│   └── ehr_data/
│       └── mimic3_mortality_samples_test.json
└── results/
    ├── arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8/
    │   ├── debate_responses_{pid}_{vid}.log
    │   └── debate_logs/
    │       └── retrieve_mortality_assessment_{pid}_{vid}.json
    └── fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8/
        ├── debate_responses_{pid}_{vid}.log
        └── debate_logs/
            └── retrieve_mortality_assessment_{pid}_{vid}.json
```

### Output Files
```
/data/wang/junh/githubs/Debate/KARE/
└── data/
    └── rl_training/
        ├── train.parquet (400 samples, 80%)
        ├── val.parquet (100 samples, 20%)
        └── metadata.json (generation statistics)
```

---

## Quality Checks

### 1. Prompt Length Validation
- **Target**: 8,000 - 12,000 tokens per prompt
- **Max**: 12,000 tokens (Qwen3-4B supports 32k)
- **Action if exceeded**: Truncate debate history intelligently

### 2. Data Completeness
- [ ] Patient context exists and non-empty
- [ ] Debate history extracted for all 3 agents
- [ ] Retrieval query and documents present
- [ ] Ground truth label available

### 3. Format Validation
- [ ] Prompt ends with proper instruction
- [ ] All sections properly formatted
- [ ] No encoding errors (UTF-8)

### 4. Ground Truth Distribution
- [ ] Check mortality rate balance (likely ~10-20% mortality)
- [ ] Stratify train/val split to maintain distribution

---

## Implementation Steps

### Phase 1: Parser Development (2-3 hours)
1. Create `parse_debate_logs.py`:
   - Function to find log files (with fallback)
   - Extract agent responses from logs
   - Handle log parsing edge cases

2. Create `parse_retrieval_logs.py`:
   - Load retrieval JSON files
   - Extract query and documents
   - Format for prompt inclusion

3. Create `format_patient_context.py`:
   - Use `KAREDataAdapter`
   - Format EHR data consistently

### Phase 2: Prompt Assembly (1-2 hours)
4. Create `assemble_prompts.py`:
   - Combine all components
   - Apply template
   - Validate lengths

### Phase 3: Dataset Generation (1 hour)
5. Create `generate_training_data.py`:
   - Main pipeline script
   - Process 500 samples
   - Generate train/val split
   - Save to parquet

### Phase 4: Validation (30 mins)
6. Create `validate_dataset.py`:
   - Check parquet loads correctly
   - Verify VERL compatibility
   - Generate statistics

---

## Expected Output Statistics

### Dataset Size
- **Train**: 400 prompts
- **Val**: 100 prompts
- **Total**: 500 prompts

### Context Lengths
- **Min**: ~6,000 tokens
- **Avg**: ~10,000 tokens
- **Max**: ~12,000 tokens

### Ground Truth Distribution
- **Survival (0)**: ~85-90%
- **Mortality (1)**: ~10-15%
- **Stratified split**: Maintain distribution in train/val

---

## Next Steps After Data Generation

1. **Test load in VERL**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset('parquet', data_files='train.parquet')
   print(dataset[0])
   ```

2. **Implement reward function** (`kare_format_reward.py`)

3. **Run GRPO training** with generated dataset

4. **Evaluate format compliance** on validation set

---

## Notes and Considerations

### Context Length Management
- Debate history can be very long (5,000+ tokens)
- If prompt exceeds 12,000 tokens:
  - Truncate debate history while preserving key information
  - Keep: agent role + key findings + prediction
  - Remove: detailed reasoning, verbose explanations

### Missing Data Handling
- If debate log missing → skip sample or use fallback directory
- If retrieval log missing → skip sample (retrieval is required)
- If integrator had None prediction → use fallback logs

### Data Source Priority
1. Primary: `arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8`
2. Fallback: `fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8`
3. If both missing: Skip sample

### Future Expansion
- After initial 500 samples, can expand to full 996 test set
- Can generate survival integrator dataset similarly (separate task)
- Can add data augmentation (paraphrase prompts, vary retrieval docs)

---

**Status**: Ready for Implementation
**Estimated Time**: 4-6 hours total
**Dependencies**: `datasets`, `pandas`, `json`, `KAREDataAdapter`
