# KARE Mortality Prediction: Workflow Summary

This document summarizes the workflows, system prompts, and input variables for all four model configurations (2 modes × 2 approaches).

---

## 1. Single-Agent RAG Mode

**File**: `mortality_single_agent_rag.py`

### Workflow

1. **Initialization**
   - Initialize MedRAG retrieval system (MedCPT retriever, MedCorp2 corpus)
   - Initialize VLLM with tensor parallelism for generation
   - Setup retrieval tool for medical evidence

2. **Step 1: Initial Query with Retrieval Request**
   - Agent receives patient context (with/without similar patients based on `in_context` mode)
   - Agent is instructed to call `retrieve(query)` for medical evidence
   - Parse agent response for tool call

3. **Step 2: Retrieval Execution**
   - Extract retrieval query from agent's response
   - Execute MedRAG retrieval (k=8 documents)
   - Format retrieved documents as supplementary information

4. **Step 3: Final Prediction with Evidence**
   - Provide patient context + retrieved evidence to agent
   - Agent generates reasoning and prediction (format: 1/0)
   - Extract prediction from response

### System Prompt

**Task Description** (stored in `self.task_description`):
```
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based 
solely on conditions, procedures, and medications. 
Labels: 1 = mortality, 0 = survival

Key Considerations:
1. Conditions:
   - Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)
   - Presence of multiple comorbidities
   - Acute vs. chronic nature of conditions

2. Procedures:
   - Invasiveness and complexity of recent procedures 
   - Emergency vs. elective procedures
   - Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)

3. Medications:
   - Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)
   - Multiple medication use indicating complex health issues
   - Presence of medications typically used in end-of-life care

Note: Focus on combinations of conditions, procedures, and medications that indicate 
critical illness or a high risk of mortality. Only the patients with extremely very high 
risk of mortality (definitely die) should be predicted as 1.
```

**Retrieval Instruction** (stored in `self.retrieval_instruction`):
```
Before making your prediction, you should retrieve relevant medical evidence using 
retrieve(query) to support your reasoning.
```

### Input Prompts

#### Zero-shot Mode (`in_context="zero-shot"`)

**Step 1 - Initial Retrieval Request**:
```
Given the following task and patient context, first retrieve relevant medical evidence, 
then make a prediction.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

{self.retrieval_instruction}
```

**Step 2 - Final Prediction** (after retrieval):
```
Given the following task description, patient context, and relevant supplementary 
information, please make a prediction with reasoning based on the patient's context 
and relevant medical knowledge.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

# Supplementary Information #
{docs_context}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:
```

#### Few-shot Mode (`in_context="few-shot"`)

**Step 1 - Initial Retrieval Request**:
```
Given the following task, patient EHR context, and similar patients, first retrieve 
relevant medical evidence, then make a prediction.

========================================
# Task #
{self.task_description}

========================================
# Patient EHR Context #
{patient_context}

========================================
# Similar Patients #
Similar Patients Who Died:
{positive_similars}

Similar Patients Who Survived:
{negative_similars}

========================================

{self.retrieval_instruction}
```

**Step 2 - Final Prediction** (after retrieval):
```
Given the following task description, patient EHR context, similar patients, and 
relevant supplementary information, please make a prediction with reasoning based 
on the patient's context and relevant medical knowledge.

========================================
# Task #
{self.task_description}

========================================
# Patient EHR Context #
{patient_context}

========================================
# Similar Patients #
Similar Patients Who Died:
{positive_similars}

Similar Patients Who Survived:
{negative_similars}

========================================
# Supplementary Information #
{docs_context}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:
```

### Input Variables
- `patient_context`: Target patient clinical information (string)
- `positive_similars`: Similar patients who died (string)
- `negative_similars`: Similar patients who survived (string)
- `docs_context`: Retrieved medical evidence formatted as documents (string)
- `in_context`: Mode selection - "zero-shot" or "few-shot"

---

## 2. Single-Agent CoT Mode

**File**: `mortality_single_agent_cot.py`

### Workflow

1. **Initialization**
   - Initialize VLLM directly (no MedRAG - pure Chain-of-Thought)
   - Single GPU setup with higher memory utilization (0.85)

2. **Single-Step Prediction**
   - Agent receives patient context (with/without similar patients based on `in_context` mode)
   - Agent generates reasoning and prediction in one step
   - Extract prediction from response (format: 1/0)

### System Prompt

**Task Description**: Same as Single-Agent RAG mode (`self.task_description`)

**No additional system prompt** - Direct user prompt only

### Input Prompts

#### Zero-shot Mode (`in_context="zero-shot"`)

```
Given the following task description and patient context, please make a prediction 
with reasoning based on the patient's context.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #  
[Your prediction here (1/0)]

Output:
```

#### Few-shot Mode (`in_context="few-shot"`)

```
Given the following task description, patient EHR context, similar patients, please 
make a prediction with reasoning based on the patient's context.

========================================
# Task #
{self.task_description}

========================================
# Patient EHR Context #
{patient_context}

========================================
# Similar Patients #
Similar Patients Who Died:
{positive_similars}

Similar Patients Who Survived:
{negative_similars}

========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #  
[Your prediction here (1/0)]


Output:
```

### Input Variables
- `patient_context`: Target patient clinical information (string)
- `positive_similars`: Similar patients who died (string)
- `negative_similars`: Similar patients who survived (string)
- `in_context`: Mode selection - "zero-shot" or "few-shot"

---

## 3. Multi-Agent RAG Mode

**File**: `mortality_debate_rag.py`

### Workflow

1. **Initialization**
   - Initialize MedRAG retrieval system
   - Initialize main VLLM for agents 1-3 (debate agents)
   - Initialize separate integrator VLLM (optional different model)
   - Setup 3 specialized agents with distinct roles

2. **Preprocessing** (Label-Blind Contrastive Analysis)
   - Create contrastive views comparing target patient with similar patients
   - Generate shared/unique feature analysis for each analyst
   - Removes outcome labels to prevent bias

3. **Round 1: Similar Patient Comparisons (Parallel Batch)**
   - **Agent 1** (Mortality Risk Assessor): Analyze target vs. positive similar patient
   - **Agent 2** (Protective Factor Analyst): Analyze target vs. negative similar patient
   - Both agents run in parallel using VLLM batch generation
   - Each agent performs RAG retrieval independently
   - Focus on clinical pattern analysis (label-blind)

4. **Round 2: Integration and Consensus**
   - **Agent 3** (Balanced Clinical Integrator): Reviews both analyses
   - Performs additional RAG retrieval for medical knowledge
   - Identifies key factors for mortality prediction
   - Generates final probability assessment (mortality & survival probabilities)

### System Prompts

#### Agent 1 & 2: Clinical Pattern Analysts (stored in `self.agent_prompts`)

**Mortality Risk Assessor** and **Protective Factor Analyst** (same prompt):
```
You are a medical AI that analyzes clinical patterns between patients.

Task:
Given (1) Target patient and (2) One Similar patient, produce a contrastive comparison 
that is grounded in the provided codes.

**CLINICAL PATTERN ANALYSIS:**

1. **Shared Clinical Features:**
   - What conditions, procedures, and medications appear in BOTH patients?
   - What is the clinical significance of these commonalities?

2. **Similar-Specific Features:**
   - What is unique to the similar patient?
   - What does this tell us about different clinical paths?

**TEMPORAL PROGRESSION:**
Analyze how shared and unique patterns evolve across visits.

**IMPORTANT:** Do NOT speculate about outcomes or mortality. Focus solely on clinical 
pattern analysis.
```

#### Agent 3: Balanced Clinical Integrator

```
You are a medical AI Clinical Assistant analyzing mortality and survival probabilities 
for the NEXT hospital visit.

IMPORTANT: Mortality is rare - only predict mortality probability > 0.5 if evidence 
STRONGLY supports it. When uncertain, predict survival probability > 0.5. The Target 
patient is the source of truth. Do not treat Similar-only items as present in the Target.

Available tools:
- <search>query</search>: Retrieve medical evidence. Retrieved information will appear 
  in <information>...</information> tags.

Workflow:
1) Compare the Target patient to two similar cases using the two analysis, and write 
   3-4 key factors contribute to the target patient's next visit.
2) When you need additional knowledge, call <search>your custom query</search> based 
   on the patient's specific conditions (e.g., <search>sepsis mortality prognosis 
   elderly patients</search>)
3) After seeing the <information>retrieved evidence</information>, analyze BOTH risky 
   factors AND survival factors. Be conservative: mortality is rare, so strong evidence 
   is needed for high mortality probability
4) After reviewing all evidence, provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00
```

### Input Prompts

#### Round 1: Agent 1 (Mortality Risk Assessor)

```
{system_prompt}

{primary_context}

{retrieval_context}

Provide your clinical analysis and mortality risk assessment:
```

Where:
- `primary_context`: Preprocessed contrastive analysis (target + positive similar)
- `retrieval_context`: Retrieved medical evidence (if RAG enabled)

#### Round 1: Agent 2 (Protective Factor Analyst)

```
{system_prompt}

{primary_context}

{retrieval_context}

Provide your clinical analysis and mortality risk assessment:
```

Where:
- `primary_context`: Preprocessed contrastive analysis (target + negative similar)
- `retrieval_context`: Retrieved medical evidence (if RAG enabled)

#### Round 2: Agent 3 (Integrator)

```
{system_prompt}

## Target Patient EHR Context ##
{patient_context}

## Retrieved Medical Evidence ##
{retrieval_context}

## Previous Debate Rounds:

### mortality_risk_assessor:
{agent1_message}

### protective_factor_analyst:
{agent2_message}

Provide your clinical analysis and mortality risk assessment:
```

### Input Variables
- `patient_context`: Target patient clinical information (string)
- `positive_similars`: Similar patients who died - raw format (string)
- `negative_similars`: Similar patients who survived - raw format (string)
- `preprocessed_inputs`: Dict with 'analyst1_input' and 'analyst2_input' (contrastive views)
- `debate_history`: List of agent responses from previous rounds
- `retrieval_context`: Retrieved medical evidence formatted as documents (string)

---

## 4. Multi-Agent CoT Mode

**File**: `mortality_debate_cot.py`

### Workflow

1. **Initialization**
   - Initialize single VLLM instance (no MedRAG)
   - All agents share same model instance
   - Setup 3 specialized agents with distinct roles
   - Optional: Use Fireworks API instead of local VLLM

2. **Round 1: Similar Patient Comparisons (Sequential, Label-Blind)**
   - **Agent 1** (Mortality Risk Assessor): Analyze target vs. one similar patient
   - **Agent 2** (Protective Factor Analyst): Analyze target vs. another similar patient
   - Both agents operate label-blind (no mortality labels shown)
   - Focus on clinical pattern analysis only

3. **Round 2: Integration and Consensus**
   - **Agent 3** (Balanced Clinical Integrator): Reviews both analyses
   - Identifies key factors based on previous agent insights
   - Analyzes both risky and protective factors
   - Generates final probability assessment (mortality & survival probabilities)

### System Prompts

#### Agent 1 & 2: Clinical Pattern Analysts

**Same as Multi-Agent RAG mode** - identical prompts for Mortality Risk Assessor and Protective Factor Analyst

#### Agent 3: Balanced Clinical Integrator

```
You are a medical AI Clinical Assistant analyzing mortality and survival probabilities 
for the NEXT hospital visit.

IMPORTANT: Mortality is rare - only predict mortality probability > 0.5 if evidence 
STRONGLY supports it. When uncertain, predict survival probability > 0.5.

Workflow:
1) Review the two clinical pattern analyses from previous agents
2) Identify 3-4 key factors that contribute to the target patient's next visit outcome
3) Analyze BOTH risky factors AND survival factors
4) Be conservative: mortality is rare, so strong evidence is needed for high mortality 
   probability
5) Provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00
```

### Input Prompts

#### Round 1: Agent 1 (Mortality Risk Assessor)

```
{system_prompt}

## Target Patient ##
{patient_context}

## Similar Patient ##
{positive_similars}

Provide your clinical analysis and mortality risk assessment:
```

#### Round 1: Agent 2 (Protective Factor Analyst)

```
{system_prompt}

## Target Patient ##
{patient_context}

## Similar Patient ##
{negative_similars}

Provide your clinical analysis and mortality risk assessment:
```

#### Round 2: Agent 3 (Integrator)

```
{system_prompt}

## Target Patient EHR Context ##
{patient_context}

## Previous Debate Rounds:

### Round 1 - Agent 1 (mortality_risk_assessor):
{agent1_message}

### Round 1 - Agent 2 (protective_factor_analyst):
{agent2_message}

Provide your clinical analysis and mortality risk assessment:
```

### Input Variables
- `patient_context`: Target patient clinical information (string)
- `positive_similars`: Similar patients who died - shown label-blind to Agent 1 (string)
- `negative_similars`: Similar patients who survived - shown label-blind to Agent 2 (string)
- `debate_history`: List of agent responses from Round 1 (List[Dict])
- `history_text`: Formatted string of previous agent analyses for integrator

---

## Key Differences Summary

| Feature | Single-Agent RAG | Single-Agent CoT | Multi-Agent RAG | Multi-Agent CoT |
|---------|------------------|------------------|-----------------|-----------------|
| **Agents** | 1 | 1 | 3 (2 analysts + 1 integrator) | 3 (2 analysts + 1 integrator) |
| **RAG Retrieval** | Yes (MedRAG) | No | Yes (MedRAG) | No |
| **Debate Rounds** | 2 steps (query → retrieve → answer) | 1 step | 2 rounds | 2 rounds |
| **Parallel Processing** | No | No | Yes (Round 1 batch) | No |
| **Label-Blind Analysis** | No | No | Yes (preprocessed) | Yes |
| **External Knowledge** | Medical literature | Internal reasoning | Medical literature | Agent collaboration |
| **GPU Setup** | Tensor parallelism | Single GPU | Tensor parallelism + separate integrator | Tensor parallelism or Fireworks API |
| **Temperature** | 0.7 | 0.7 | 0.3 (analysts), varies (integrator) | 0.3 (analysts), 0.5 (integrator) |
| **Max Tokens** | 2048 | 2048 | 32768 (analysts), varies (integrator) | 2048 (analysts), 4096 (integrator) |

---

## Output Format (All Models)

All models produce predictions in KARE format:
- **Prediction**: 0 (survival) or 1 (mortality)
- **Reasoning**: Chain-of-thought explanation
- **Probabilities** (multi-agent only): Mortality and survival probabilities (0.00-1.00)

### Expected Response Structure

**Single-Agent**:
```
# Reasoning #
[Analysis of patient condition, risk factors, etc.]

# Prediction #
[0 or 1]
```

**Multi-Agent (Integrator)**:
```
[Analysis of debate and key factors]

MORTALITY PROBABILITY: 0.XX
SURVIVAL PROBABILITY: 0.XX
```

---

## Common Variables Across All Models

- `patient_context` (str): Target patient's EHR with conditions, procedures, medications
- `positive_similars` (str): Similar patients who died
- `negative_similars` (str): Similar patients who survived  
- `in_context` (str): "zero-shot" or "few-shot" mode
- `patient_id` (str): Patient identifier for logging
- `output_dir` (str): Directory for debate logs
- `ground_truth` (int): Actual outcome (0 or 1) for fallback

---

*Generated: 2026-01-03*
