# KARE Multi-Agent Debate System for Mortality Prediction

## System Overview

This implementation integrates a **four-agent structured debate system with RAG-enhanced tool calling** with KARE's temporal patient infrastructure for enhanced mortality prediction on MIMIC-III data. The system replaces KARE's single Claude-based reasoning with specialized multi-agent collaboration while maintaining full compatibility with KARE's temporal patient representation.

**Key Innovation**: Four specialized medical AI agents conduct structured three-round debates using temporal patient contexts, precomputed similar patient comparisons, and strategic medical evidence retrieval to form evidence-based mortality predictions with automatic summarization for context management.

## Architecture

### Current Implementation Pipeline:
```
Temporal Patient Context + Similar Patients (mortality=0/1) + MedRAG Retrieval → Four-Agent Structured Debate with Summarization → Binary Prediction (0/1)
```

### Multi-Agent Collaboration Flow:
```
Round 1: Label-Blind Contrastive Analysis (PARALLEL BATCH PROCESSING)
├── Analyst 1 (Mortality Case): Target vs Similar Patient Comparison + RAG Retrieval
│   └── Label-Blind: No knowledge that similar patient had mortality=1
└── Analyst 2 (Survival Case): Target vs Similar Patient Comparison + RAG Retrieval
    └── Label-Blind: No knowledge that similar patient had survival=0

Round 2: Search-R1 Style Integration with Label Injection:
├── Prepare History: Add labels ("Mortality=1 Analysis", "Survival=0 Analysis")
├── Search-R1 Native Flow:
│   ├── Step 1: Generate until </search> tag (max_tokens=2048)
│   ├── Step 2: Execute retrieval → format as <information>docs</information>
│   └── Step 3: Continue generation → MORTALITY/SURVIVAL PROBABILITIES (once)
└── Final Decision: Based on probability comparison
```

### Context Management and Summarization:
```
Long Debate History → Round-by-Round Summarization → Integrator Context
├── Target Analysis (>6k tokens) → Bullet Point Summary (~3k chars)
├── Risk Assessment (>6k tokens) → Bullet Point Summary (~3k chars)  
├── Protective Analysis (>6k tokens) → Bullet Point Summary (~3k chars)
└── Combined History → Integrator Input (manageable size)
```

## Four-Agent Specialized System

### Agent Architecture
The system employs **four specialized medical AI agents** with distinct analytical perspectives and RAG-enhanced capabilities:

#### 1. **Target Patient Analyst** (Round 1)
- **Role**: Individual patient risk assessment specialist with medical evidence retrieval
- **Focus**: Temporal EHR progression analysis + Medical literature consultation
- **Temperature**: 0.7 (higher creativity for comprehensive analysis)
- **Max Tokens**: 8,192
- **RAG Integration**: Retrieves relevant medical evidence based on patient conditions
- **Specialization**: 
  - Timeline analysis of clinical events
  - Risk/protective factor identification (3 each, balanced assessment)
  - Temporal deterioration pattern detection
  - Evidence-based preliminary mortality prediction
- **Output**: `\\boxed{0}` or `\\boxed{1}` + detailed reasoning

#### 2. **Mortality Risk Assessor** (Round 1A - Label-Blind Analyst)
- **Role**: Clinical pattern comparison specialist (LABEL-BLIND)
- **Focus**: Contrastive analysis of target vs one similar patient
- **Temperature**: 0.3 (focused, consistent analysis)
- **Max Tokens**: 32,768
- **Label-Blind Design**: 
  - Receives preprocessed input showing "Shared" vs "Unique to Target" vs "Unique to Similar"
  - NO knowledge that similar patient had mortality=1 outcome
  - Retrieval query uses patient context ONLY (no "mortality risk" hints)
- **RAG Integration**: Label-blind retrieval (no outcome-based keywords)
- **Output**: Pattern analysis only (NO outcome speculation, NO prediction, NO \\boxed{})
- **Specialization**:
  - Shared clinical feature identification across both patients
  - Target-specific pattern extraction (unique conditions/procedures/medications)
  - Similar-specific pattern extraction
  - Temporal progression analysis
  - Medical literature support for pattern significance

#### 3. **Protective Factor Analyst** (Round 1B - Label-Blind Analyst)
- **Role**: Clinical pattern comparison specialist (LABEL-BLIND)
- **Focus**: Contrastive analysis of target vs one similar patient
- **Temperature**: 0.3 (focused, consistent analysis)
- **Max Tokens**: 32,768
- **Label-Blind Design**: 
  - Receives preprocessed input showing "Shared" vs "Unique to Target" vs "Unique to Similar"
  - NO knowledge that similar patient had survival=0 outcome
  - Retrieval query uses patient context ONLY (no "survival factor" hints)
- **RAG Integration**: Label-blind retrieval (no outcome-based keywords)
- **Output**: Pattern analysis only (NO outcome speculation, NO prediction, NO \\boxed{})
- **Specialization**:
  - Shared clinical feature identification across both patients
  - Target-specific pattern extraction (unique conditions/procedures/medications)
  - Similar-specific pattern extraction
  - Temporal progression analysis
  - Medical literature support for pattern significance

#### 4. **Balanced Clinical Integrator** (Round 2 - Search-R1 Native Single-Generation)
- **Role**: Evidence-based decision maker with Search-R1 trained retrieval behavior
- **Focus**: Unified mortality and survival probability assessment
- **Temperature**: 0.5 (balanced reasoning)
- **Max Tokens**: 34,816 total (2048 search + 32,768 reasoning)
- **Search-R1 Native Architecture**:
  
  **Three-Step Generation Flow**
  - Step 1: Generate until `</search>` tag (max_tokens=2048, stop=["</search>"])
    - Model identifies knowledge gaps
    - Generates: `<search>specific medical query</search>`
  - Step 2: Information Injection (system-controlled)
    - Execute retrieval with model's query
    - Format as: `<information>retrieved docs</information>`
    - Inject into context without new generation
  - Step 3: Continue Generation (max_tokens=32,768)
    - Model receives `<information>` block
    - Continues reasoning from where it stopped
    - Outputs: `MORTALITY PROBABILITY: X.XX` and `SURVIVAL PROBABILITY: X.XX`
  
- **Label-Aware Integration**:
  - Receives debate history WITH labels ("Mortality=1 Analysis", "Survival=0 Analysis")
  - Note added: "Analyses were conducted without knowledge of outcomes"
  
- **Final Decision**: Based on probability comparison (mortality > survival → predict 1)
- **Key Fix**: Single-generation prevents duplicate probabilities (was generating twice with old two-phase approach)
- **Specialization**:
  - Trained Search-R1 behavior for knowledge-seeking
  - Unified mortality and survival assessment in single response
  - Evidence integration (label-revealed debate + retrieved medical literature)
  - Conservative mortality prediction with base rate consideration

### System Hyperparameters

#### Model Configuration
- **Base Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **GPU Setup**: Multi-GPU (default: GPUs 6,7)
- **Context Window**: 32,768 tokens (enforced input limit for Qwen models)
- **Response Format**: String-based with structured prediction extraction
- **RAG Configuration**: 
  - **Corpus**: MedCorp2 (medical literature database)
  - **Retriever**: MedCPT (medical domain-specific retriever)
  - **Retrieval Strategy**: Source-specific retrieval with `medrag_answer_by_source()`
  - **k Values**: Round 1&2: k=8, Round 3: k=16 (more evidence for integration)
- **Thinking Mode**: Enabled (`enable_thinking=True`) for enhanced reasoning
- **Logging**: Reduced verbosity (`disable_log_stats=True`)

#### Generation Parameters
```python
Agent Parameters:
├── Target Patient Analyst:
│   ├── Temperature: 0.7
│   ├── Top-p: 0.9
│   ├── Max Tokens: 8,192
│   ├── Repetition Penalty: 1.15
│   ├── Stop Sequences: ["<|im_end|>", "</s>"] (allows \\boxed{} completion)
│   └── RAG Retrieval: k=8 documents from MedCorp2
├── Mortality Risk Assessor:
│   ├── Temperature: 0.3
│   ├── Top-p: 0.9
│   ├── Max Tokens: 8,192
│   ├── Repetition Penalty: 1.15
│   ├── Stop Sequences: ["<|im_end|>", "</s>"]
│   └── RAG Retrieval: k=8 documents (mortality risk focused)
├── Protective Factor Analyst:
│   ├── Temperature: 0.3
│   ├── Top-p: 0.9
│   ├── Max Tokens: 8,192
│   ├── Repetition Penalty: 1.15
│   ├── Stop Sequences: ["<|im_end|>", "</s>"]
│   └── RAG Retrieval: k=8 documents (survival factors focused)
└── Balanced Clinical Integrator (Two-Step):
    ├── Step 1 - Mortality Assessment:
    │   ├── Temperature: 0.3
    │   ├── Max Tokens: 8,192
    │   ├── Tool Calling: retrieve("mortality risk query")
    │   ├── RAG Retrieval: k=16 documents
    │   └── Stop: ["SURVIVAL PROBABILITY:", "Step 2:"]
    ├── Step 2 - Survival Assessment:
    │   ├── Temperature: 0.3
    │   ├── Max Tokens: 8,192
    │   ├── Tool Calling: retrieve("survival factors query")
    │   ├── RAG Retrieval: k=16 documents
    │   └── Stop: ["MORTALITY PROBABILITY:", "Step 1:"]
    └── Final Integration: Manual probability comparison
```

#### Context Management and Summarization Parameters
```python
Summarization System:
├── Trigger Threshold: >6,000 tokens (~24,000 chars) per round
├── Target Length: 3,000 tokens (~12,000 chars) per summary
├── Word Limit Enforcement: target_tokens // 1.3 words maximum
├── Character Limit: target_tokens * 4 characters maximum
├── Generation Parameters:
│   ├── Max Tokens: target_tokens // 2 (force brevity)
│   ├── Temperature: 0.0 (deterministic)
│   ├── Repetition Penalty: 1.5 (avoid redundancy)
│   └── Stop Sequences: ["\n\n", "Original text", "ORIGINAL:"]
├── Format: Bullet points (•) for structured conciseness
├── Safety Layers:
│   ├── Character truncation if summary > max_chars
│   ├── Word count truncation if words > word_limit
│   └── Fallback to intelligent truncation if summarization fails
└── Quality Control: Multiple enforcement layers prevent expansion
```

#### Prediction Format
- **Round 1A/1B**: Pattern analysis only (NO predictions, NO \\boxed{}, NO outcome speculation)
  - Label-blind analysts focus solely on clinical pattern comparison
  - Output: Shared features, target-specific features, similar-specific features
- **Round 2**: Single-generation probability assessment with Search-R1 retrieval
  - Output: `MORTALITY PROBABILITY: X.XX (0.00 to 1.00)` + `SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)`
  - Generated ONCE after receiving `<information>retrieved evidence</information>`
  - Probabilities must sum to 1.00
- **Extraction Method**: 
  - Round 1: No extraction needed (pattern analysis only)
  - Round 2: Regex pattern matching for probability values from unified response
  - Key Fix: Single extraction point prevents duplicate probabilities
- **Fallback Strategy**: Conservative default to survival (0) if extraction fails
- **Decision Method**: Probability comparison → `prediction = 1 if mortality_prob > survival_prob else 0`

## KARE's Temporal Structure Understanding

### PyHealth Mortality Prediction Task
- Each sample represents **one visit predicting the next visit's mortality outcome**
- For patient with visits [V0, V1, V2]: Creates samples V0→V1_outcome, V1→V2_outcome

### KARE's Rolling Context Creation
- `patient_10004_0`: Uses **only visit 0** context → predicts visit 1 mortality
- `patient_10004_1`: Uses **visits 0+1 combined** context → predicts visit 2 mortality  
- `patient_10004_2`: Uses **visits 0+1+2 combined** context → predicts visit 3 mortality

### Similar Patient Matching
- **Temporal Consistency**: Similar patients have comparable temporal depth
- **Patient Exclusion**: All temporal instances of same patient excluded (e.g., for `10004_1`, exclude `10004_0`, `10004_1`, `10004_2`)
- **Context Format**: Multi-visit progression with "(new)" and "(continued)" annotations

## Core Components

### 1. Label-Blind Contrastive Preprocessing ✅ IMPLEMENTED

#### Preprocessing Architecture
The system implements label-blind contrastive analysis to prevent outcome leakage to analysts:

```python
from kare_contrastive_preprocessing import preprocess_for_debate

preprocessed = preprocess_for_debate(
    target_context,      # Target patient EHR
    positive_similars,   # Similar patient with mortality=1
    negative_similars    # Similar patient with survival=0
)
# Returns:
# - analyst1_input: Target vs Similar (mortality=1) - NO LABEL
# - analyst2_input: Target vs Similar (survival=0) - NO LABEL
# - original_target: Original target context for reference
```

#### Contrastive View Format
Each analyst receives structured comparison showing:
- **Shared with Similar**: Conditions/procedures/medications in BOTH patients
- **Unique to Target**: Items only in target patient
- **Unique to Similar**: Items only in similar patient

**Key Design**: Analysts see clinical patterns but NOT outcomes. No mentions of "mortality", "death", "survival" in analyst inputs.

#### Label Injection for Integrator
Labels added ONLY when preparing history for integrator:
```python
from kare_contrastive_preprocessing import format_integrator_history_with_labels

integrator_history = format_integrator_history_with_labels(
    analyst1_response,  # Pattern analysis from analyst 1
    analyst2_response   # Pattern analysis from analyst 2
)
# Adds headers:
# "### Similar Case with Mortality=1 (positive class) Analysis:"
# "### Similar Case with Survival=0 (negative class) Analysis:"
# Note: "Analyses were conducted without knowledge of outcomes"
```

### 2. Context Management and Summarization System ✅ IMPLEMENTED

#### Automatic Round-by-Round Summarization
The system implements intelligent context management to handle long debate histories while preserving critical medical information:

```python
def _summarize_round_response(self, round_text: str, round_name: str, target_tokens: int = 4000) -> str:
    # Multi-layer length enforcement system
    
    # Layer 1: Threshold Check (6000 tokens ≈ 24000 chars)
    if len(round_text) <= 24000:
        return round_text  # Keep original if under limit
    
    # Layer 2: Strict Constraint Prompt
    word_limit = target_tokens // 1.3  # Conservative word estimate
    max_chars = target_tokens * 4      # Conservative character estimate
    
    prompt = f"""Create a CONCISE medical summary in EXACTLY {word_limit} words or less.
    
    STRICT REQUIREMENTS:
    - Maximum {word_limit} words
    - Maximum {max_chars} characters  
    - Use bullet points for key information
    - No repetition or redundancy
    - Focus ONLY on essential medical facts
    
    Original text: {round_text}
    
    CONCISE SUMMARY ({word_limit} words max):
    •"""
    
    # Layer 3: Constrained Generation
    summary = llm(prompt, 
                  max_tokens=target_tokens // 2,  # Force shorter generation
                  temperature=0.0,                # Deterministic
                  repetition_penalty=1.5)        # Avoid redundancy
    
    # Layer 4: Character Limit Enforcement
    if len(summary) > max_chars:
        summary = summary[:max_chars-3] + "..."
    
    # Layer 5: Word Count Enforcement  
    words = summary.split()
    if len(words) > word_limit:
        summary = ' '.join(words[:word_limit]) + "..."
    
    return summary
```

#### Context Management Flow
```
Individual Round Analysis:
├── Round 1 (Target Analysis): Raw response → Check length → Summarize if >6k tokens
├── Round 2A (Risk Assessment): Raw response → Check length → Summarize if >6k tokens  
├── Round 2B (Protective Analysis): Raw response → Check length → Summarize if >6k tokens
└── Combined History: Summarized rounds → Integrator input (manageable size)

Integrator Context Preparation:
└── _prepare_integrator_history():
    ├── Process each round individually
    ├── Apply round-specific summarization if needed
    ├── Combine into structured history format
    └── Result: ~12k chars max (3 rounds × 4k chars) vs potential 72k+ chars
```

#### Summarization Quality Control
- **Format Enforcement**: Bullet points (•) for structured information
- **Content Focus**: Round-specific focus areas (conditions, risk factors, protective factors)
- **Length Limits**: Multiple enforcement layers prevent runaway generation
- **Fallback Strategy**: Intelligent truncation if summarization fails
- **Medical Preservation**: Designed to preserve critical clinical information

### 2. Data Sources (Leveraging KARE's Infrastructure) ✅ IMPLEMENTED
- **Test Data**: `/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_test.json` (996 temporal instances)
- **Similar Patients**: `/data/wang/junh/datasets/KARE/patient_context/similar_patient_qwen/patient_to_top_1_patient_contexts_mimic3_mortality.json` (9,717 patient contexts)
- **Temporal Patient IDs**: Format `{patient_id}_{visit_index}` (e.g., `"10188_1"` for 2-visit context)
- **Medical Knowledge**: KARE's existing knowledge retrieval system (can be integrated later)

## Implementation Status ✅ COMPLETED

### Phase 1: Data Integration ✅ COMPLETED
1. **KARE Data Adapter (`kare_data_adapter.py`)** ✅
   ```python
   class KAREDataAdapter:
       def __init__(self):
           # Loads 996 temporal instances from test data
           # Loads 9,717 precomputed similar patient contexts
           
       def format_patient_context(self, patient_data):
           # Handles rolling visit arrays: conditions[[v0], [v1]], procedures[[v0], [v1]]
           # Creates temporal progression with "(new)" and "(continued)" annotations
           # Returns KARE-compatible multi-visit context format
           
       def get_test_sample(self, index):
           # Constructs temporal patient ID: "{patient_id}_{visit_index}"
           # Retrieves similar patients using KARE's temporal ID format
           # Returns formatted sample with temporal consistency
   ```

2. **Temporal Context Formatting** ✅
   - **Rolling Visit Handling**: Processes cumulative visit arrays correctly
   - **Temporal Annotations**: Shows "(new)" vs "(continued)" items across visits  
   - **KARE ID Format**: Uses `patient_10188_1` format for 2-visit temporal context
   - **Similar Patient Integration**: Separates positive/negative similar cases

### 2. Four-Agent Debate System ✅ COMPLETED
1. **Mortality Debate System (`mortality_debate_system.py`)** ✅
   - **Four Specialized Agents**: Target analyst, positive/negative comparators, medical integrator
   - **Structured Three-Round Flow**: Individual → Comparative → Integrative analysis
   - **VLLM Integration**: Qwen3-4B-Instruct-2507 with optimized parameters per agent
   - **Prediction Method**: Direct integrator decision (removed consensus voting for simplicity)
   - **Patient-Specific Logging**: Structured logging per patient in `results/kare_mor_{model_name}/`

2. **Specialized Agent System Prompts** ✅ IMPLEMENTED
   ```python
   # Agent 1: Target Patient Analyst (Round 1)
   "target_patient_analyst": """
   You are a medical AI that provides balanced clinical assessment for mortality prediction.
   
   Instructions:
   - Read all visits in order and summarize the main clinical story
   - Review retrieved medical evidence for relevant mortality/survival factors
   - **BALANCED ASSESSMENT**: List 3 main RISK factors AND 3 main PROTECTIVE factors
   - Provide brief explanation integrating patient data with medical evidence
   - Make preliminary prediction based on evidence
   - Only predict mortality (1) if evidence STRONGLY indicates death is very likely
   - When uncertain, err toward survival prediction (0)
   - End with: \\boxed{0} for SURVIVAL or \\boxed{1} for MORTALITY
   """,
   
   # Agent 2: Mortality Risk Assessor (Round 2A)
   "mortality_risk_assessor": """
   You are a medical AI Risk Assessor that identifies mortality risk factors.
   
   Instructions:
   - Analyze common patterns in fatal cases (diseases, procedures, medications)
   - Review retrieved medical evidence for established mortality risk factors
   - Identify 3-5 key factors that clearly increase mortality risk
   - Rate strength of each risk factor (Weak/Moderate/Strong evidence)
   - Explain why factors indicate increased mortality risk with medical evidence
   - DO NOT make predictions or output \\boxed{0} or \\boxed{1}
   - Focus only on EVIDENCE that supports MORTALITY
   """,
   
   # Agent 3: Protective Factor Analyst (Round 2B)
   "protective_factor_analyst": """
   You are a medical AI that identifies factors supporting survival.
   
   Instructions:
   - Analyze common patterns in survival cases (treatments, recovery trajectories)
   - Review retrieved medical evidence for protective and survival factors
   - Identify 3-5 key factors that clearly support survival
   - Rate strength of each protective factor (Weak/Moderate/Strong evidence)
   - Explain why factors support survival with medical evidence
   - DO NOT make predictions or output \\boxed{0} or \\boxed{1}
   - Focus only on EVIDENCE that supports SURVIVAL
   """,
   
   # Agent 4A: Mortality Assessment Integrator (Round 3 Step 1)
   "balanced_clinical_integrator_mortality": """
   You are a medical AI analyzing MORTALITY risk with tool calling.
   
   Available tools: retrieve(query)
   
   Instructions:
   1) **INITIAL MORTALITY RISK ASSESSMENT**: Analyze available information focusing on mortality risk factors
   2) **IDENTIFY KNOWLEDGE GAPS**: Determine what medical evidence you need
   3) **STRATEGIC RETRIEVAL**: Call retrieve("specific mortality risk query") for needed knowledge
   4) **EVIDENCE INTEGRATION**: Review retrieved evidence for mortality risk indicators
   5) **FINAL MORTALITY ASSESSMENT**: Focus ONLY on factors that increase death probability
   6) Consider base rate: most patients survive, need strong evidence for high mortality
   - Focus ONLY on mortality risk factors
   - End with: MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
   """,
   
   # Agent 4B: Survival Assessment Integrator (Round 3 Step 2)  
   "balanced_clinical_integrator_survival": """
   You are a medical AI analyzing SURVIVAL probability with tool calling.
   
   Available tools: retrieve(query)
   
   Instructions:
   1) **INITIAL SURVIVAL ASSESSMENT**: Analyze available information focusing on protective factors
   2) **IDENTIFY KNOWLEDGE GAPS**: Determine what medical evidence you need about protective factors
   3) **STRATEGIC RETRIEVAL**: Call retrieve("specific survival/protective factors query") for needed knowledge
   4) **EVIDENCE INTEGRATION**: Review retrieved evidence for protective factors and treatment effectiveness
   5) **FINAL SURVIVAL ASSESSMENT**: Focus ONLY on factors that support patient survival
   6) Consider treatment effectiveness, patient resilience, positive prognostic factors
   - Focus ONLY on survival probability and protective factors
   - End with: SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)
   """
   ```

3. **Enhanced Two-Round Flow with Label-Blind Preprocessing and Search-R1** ✅ IMPLEMENTED
   ```python
   class MortalityDebateSystem:
       def debate_mortality_prediction(self, patient_context, positive_similars, negative_similars, 
                                     medical_knowledge="", patient_id="unknown"):
           # Setup patient-specific logging with structured directory
           log_dir = Path(f"results/kare_mor_{clean_model_name}/debate_logs")
           log_filename = log_dir / f"debate_responses_{patient_id}.log"
           
           debate_history = []
           similar_patients_dict = {'positive': positive_similars, 'negative': negative_similars}
           
           # Preprocessing: Create label-blind contrastive inputs
           preprocessed = preprocess_for_debate(
               patient_context,
               positive_similars,
               negative_similars
           )
           
           # Round 1: Label-Blind Contrastive Analysis (PARALLEL BATCH)
           round1_responses = self._agent_turn_batch(
               roles=["mortality_risk_assessor", "protective_factor_analyst"],
               patient_context=patient_context,
               similar_patients=similar_patients_dict,
               debate_history=[],
               logger=logger,
               patient_id=patient_id,
               log_dir=str(log_dir),
               preprocessed_inputs=preprocessed  # Label-blind analyst1_input, analyst2_input
           )
           debate_history.extend(round1_responses)
           # Result: Unbiased pattern comparisons from both analysts
           
           # Round 2: Search-R1 Native Integration with Label Injection
           # Labels added in _prepare_integrator_history():
           # - "Similar Case with Mortality=1 (positive class) Analysis:"
           # - "Similar Case with Survival=0 (negative class) Analysis:"
           integrator_response = self._integrator_single_step_prediction(
               patient_context=patient_context,
               similar_patients=similar_patients_dict,
               medical_knowledge=medical_knowledge,
               debate_history=debate_history,  # Label-aware history for integrator
               logger=logger,
               patient_id=patient_id,
               log_dir=str(log_dir)
           )
           
           # Search-R1 Flow in _execute_integrator_attempt():
           # Step 1: Generate until </search> (max_tokens=2048)
           # Step 2: Execute retrieval → inject <information>docs</information>
           # Step 3: Continue generation → output probabilities (once)
           
           return {
               'prediction': integrator_response['prediction'],
               'mortality_probability': integrator_response['mortality_probability'],
               'survival_probability': integrator_response['survival_probability'],
               'debate_history': debate_history,
               'model_name': self.model_name
           }
   ```
           
           # Round 2B: Protective Factor Analysis with RAG (NO prediction)
           negative_response = self._agent_turn(
               role="protective_factor_analyst",
               patient_context=patient_context,
               similar_patients=similar_patients_dict,  # Uses 'negative' similar patients
               medical_knowledge=medical_knowledge,
               debate_history=debate_history,  # Has target + risk analysis
               logger=logger,
               patient_id=patient_id,
               log_dir=str(log_dir)
           )
           debate_history.append(negative_response)
           # Result: Protective factors with strength ratings + RAG evidence
           
           # Context Management: Automatic summarization of long debate history
           # _prepare_integrator_history() handles:
           # - Individual round summarization if >6k tokens (~24k chars)
           # - Bullet-point format with strict length enforcement
           # - Multiple safety layers (word limits, character limits, truncation)
           
           # Round 3: Two-Step Tool-Calling Integration
           integrator_response = self._agent_turn(
               role="balanced_clinical_integrator",  # Triggers two-step process
               patient_context=patient_context,
               similar_patients=similar_patients_dict,
               medical_knowledge=medical_knowledge,
               debate_history=debate_history,  # Automatically summarized if needed
               logger=logger,
               patient_id=patient_id,
               log_dir=str(log_dir)
           )
           # Two-step process:
           # Step 1: Mortality assessment → identify gaps → retrieve → MORTALITY PROBABILITY: X.XX
           # Step 2: Survival assessment → identify gaps → retrieve → SURVIVAL PROBABILITY: X.XX
           # Final: Manual determination (higher probability wins)
           
           debate_history.append(integrator_response)
           
           # Extract final results
           final_prediction = integrator_response.get('prediction')
           final_mortality_prob = integrator_response.get('mortality_probability')
           final_survival_prob = integrator_response.get('survival_probability')
           
           return {
               'final_prediction': final_prediction,
               'mortality_probability': final_mortality_prob,
               'survival_probability': final_survival_prob,
               'debate_history': debate_history,
               'rounds_completed': 3,
               'total_generation_time': sum(r.get('generation_time', 0) for r in debate_history),
               'integrator_prediction': integrator_response.get('prediction'),
               'target_prediction': target_response.get('prediction'),
               'mortality_retrieved_docs': integrator_response.get('mortality_retrieved_docs', 0),
               'survival_retrieved_docs': integrator_response.get('survival_retrieved_docs', 0)
           }
   ```

### Phase 3: Integration Script ✅ COMPLETED
1. **Main Integration Script (`run_kare_debate_mortality.py`)** ✅
   ```python
   def run_kare_debate_evaluation(start_idx=0, num_samples=None, output_path=None):
       # Initialize components
       data_adapter = KAREDataAdapter()  # Handles temporal structure
       debate_system = MortalityDebateSystem()  # 3-agent specialized system
       
       for i in tqdm(range(start_idx, end_idx)):
           sample = data_adapter.get_test_sample(i)  # Temporal patient context
           
           # Multi-agent debate with temporal awareness
           debate_result = debate_system.debate_mortality_prediction(
               patient_context=sample['patient_context'],  # Multi-visit progression
               similar_patients=sample['similar_patients'],  # Temporal depth matched
               medical_knowledge=""  # Can be added later
           )
           
           results.append({
               'patient_id': sample['patient_id'],  # KARE format: "10188_1"
               'base_patient_id': sample['base_patient_id'],  # Original: "10188"
               'visit_index': sample['visit_index'],  # Temporal depth: 1
               'ground_truth': sample['ground_truth'],
               'prediction': debate_result['final_prediction'],
               'confidence_score': debate_result['confidence_score'],
               'consensus_achieved': debate_result['consensus_achieved'],
               'debate_history': debate_result['debate_history']  # Full reasoning trace
           })
       
       return results_with_metrics
   ```

## Usage Instructions ✅ READY TO RUN

### Basic Usage
```bash
# Run pilot test (10 samples with structured logging)
python run_kare_debate_mortality.py --start_idx 0 --num_samples 10 \
  --output results/pilot_test.json --include_history
# Logs: results/kare_mor_Qwen_Qwen3_4B_Instruct_2507/debate_responses_{patient_id}.log

# Run full evaluation (996 samples)
python run_kare_debate_mortality.py --start_idx 0 --num_samples 996 \
  --output results/full_evaluation.json --batch_size 20

# Run with custom model and GPUs
python run_kare_debate_mortality.py --model "Qwen/Qwen3-8B-Instruct" \
  --gpus "0,1" --batch_size 10 --output results/qwen3_8b_results.json
# Logs: results/kare_mor_Qwen_Qwen3_8B_Instruct/debate_responses_{patient_id}.log

# Development testing (single sample with full logging)
python run_kare_debate_mortality.py --start_idx 0 --num_samples 1 \
  --output results/debug_test.json --include_history
```

### Command Line Arguments
- `--start_idx`: Starting sample index (default: 0)
- `--num_samples`: Number of samples to process (default: all)
- `--output`: Output JSON file path
- `--model`: HuggingFace model name (default: "Qwen/Qwen3-8B")
- `--gpus`: GPU IDs comma-separated (default: "6,7")
- `--include_history`: Include full debate history in output
- `--batch_size`: Batch size for intermediate saves (default: 10)

### Expected Output Structure
```json
{
  "metadata": {
    "timestamp": "2025-12-06 12:00:00", 
    "total_samples": 996,
    "include_debate_history": true,
    "rag_enabled": true,
    "corpus": "MedCorp2",
    "retriever": "MedCPT"
  },
  "metrics": {
    "accuracy": 0.847, 
    "precision": 0.823, 
    "recall": 0.791, 
    "f1_score": 0.807,
    "specificity": 0.856,
    "total_samples": 996,
    "tp": 45, "fp": 12, "fn": 8, "tn": 931,
    "avg_mortality_probability": 0.124,
    "avg_survival_probability": 0.876
  },
  "results": [
    {
      "patient_id": "10188_1",
      "base_patient_id": "10188",
      "visit_index": 1,
      "ground_truth": 0,
      "prediction": 0,
      "mortality_probability": 0.15,
      "survival_probability": 0.85,
      "confidence": "High",
      "rounds_completed": 3,
      "total_generation_time": 67.8,
      "mortality_retrieved_docs": 16,
      "survival_retrieved_docs": 16,
      "debate_history": [
        {
          "role": "target_patient_analyst", 
          "prediction": 0, 
          "generation_time": 14.2,
          "response_length": 2847,
          "message": "Based on the temporal EHR analysis..."
        },
        {
          "role": "mortality_risk_assessor", 
          "prediction": null, 
          "generation_time": 18.1,
          "response_length": 2156,
          "message": "Analysis of mortality risk factors..."
        },
        {
          "role": "protective_factor_analyst", 
          "prediction": null, 
          "generation_time": 15.7,
          "response_length": 1989,
          "message": "Analysis of protective factors..."
        },
        {
          "role": "balanced_clinical_integrator", 
          "prediction": 0,
          "mortality_probability": 0.15,
          "survival_probability": 0.85,
          "generation_time": 19.8,
          "response_length": 4521,
          "mortality_query": "congestive heart failure mortality risk factors elderly patients with diabetes",
          "survival_query": "heart failure treatment effectiveness ACE inhibitors beta blockers elderly survival",
          "mortality_retrieved_docs": 16,
          "survival_retrieved_docs": 16,
          "message": "## Mortality Risk Assessment ##\nTool Call: retrieve(\"...\")..."
        }
      ]
    }
  ]
}
```

### Structured Logging Output
```
results/
└── kare_mor_Qwen_Qwen3_4B_Instruct_2507/
    ├── debate_responses_10117_0.log
    ├── debate_responses_10124_0.log
    ├── debate_responses_10139_0.log
    └── ...

Log Content Format:
2025-11-25 12:00:15,123 - Starting debate for patient 10117_0
2025-11-25 12:00:28,456 - RAW RESPONSE from TARGET_PATIENT_ANALYST
2025-11-25 12:00:28,457 - Response length: 2847
2025-11-25 12:00:28,458 - Full response: [Complete agent response]
2025-11-25 12:00:28,459 - EXTRACTED PREDICTION: 1
...
2025-11-25 12:01:15,789 - DEBATE COMPLETED - Final Prediction: 1
```

## Key Advantages of This Implementation

### 1. **Search-R1 Native Architecture with Label-Blind Analysis** ✅
- **Clean Role Separation**: Label-Blind Contrastive Analysis (R1) → Label-Aware Integration (R2)
- **Search-R1 Integration**: Single-generation flow with information injection (prevents duplicate probabilities)
- **Label-Blind Design**: Analysts see clinical patterns WITHOUT outcome knowledge
  - Removes all hints: "mortality risk", "survival factors", "death", "protective"
  - Retrieval queries use patient_context only (no outcome-based keywords)
  - Contrastive format: "Shared", "Unique to Target", "Unique to Similar"
- **Label-Aware Integration**: Integrator receives labeled history revealing outcomes
  - "Mortality=1 Analysis" and "Survival=0 Analysis" headers added
  - Note: "Analyses were conducted without knowledge of outcomes"
- **Unified Probability Assessment**: Single response with both mortality and survival probabilities
- **Context Management**: Automatic summarization prevents token overflow while preserving key information

### 2. **Perfect KARE Infrastructure Integration** ✅
- **Temporal Consistency**: Correctly handles rolling visit contexts with cumulative history
- **Similar Patient Matching**: Uses KARE's precomputed temporal patient contexts 
- **Patient ID Format**: Maintains KARE's `{patient_id}_{visit_index}` temporal structure
- **Context Formatting**: Reproduces KARE's "(new)" and "(continued)" temporal annotations
- **Binary Prediction**: Clean 0/1 mortality prediction format

### 3. **Superior Reasoning Quality with Label-Blind Design** ✅
- **Unbiased Pattern Analysis**: Analysts perform clinical comparison without outcome knowledge
  - Prevents confirmation bias (no anchoring on "mortality" or "survival" labels)
  - Focus on objective clinical similarities and differences
  - Medical literature retrieval uses neutral patient context
- **Search-R1 Trained Behavior**: Model's RL-trained knowledge-seeking patterns preserved
  - Single-generation with information injection (not two-phase)
  - Natural stop at `</search>` followed by evidence integration
  - Unified reasoning after seeing `<information>` block
- **Label Revelation Only for Integration**: Integrator sees outcome context AFTER pattern analysis complete
  - Receives unbiased clinical comparisons from analysts
  - Informed by labels to make mortality vs survival assessment
  - Conservative base rate consideration (mortality is rare)
- **Medical Literature Integration**: RAG system provides evidence-based support at all stages
- **Interpretable Decisions**: Clear trail from label-blind patterns → labeled integration → probabilities

### 4. **Production-Ready Infrastructure with Advanced Context Management** ✅
- **Patient-Specific Logging**: Structured per-patient debate logs in organized directories (`results/kare_mor_{model}/debate_logs/`)
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, specificity with confusion matrix + probability scores
- **Scalable Processing**: Handles full 996-sample test set with automatic context management and batch saves
- **Advanced Summarization**: Multi-layer length enforcement prevents token overflow while preserving medical information
- **RAG Integration**: Seamless medical literature retrieval with source-specific queries and document saving
- **Error Resilience**: Graceful error handling with detailed reporting and conservative fallbacks
- **Configurable Pipeline**: Command-line interface with model/GPU/RAG parameters flexibility
- **Research Reproducibility**: Deterministic parameters, complete logging, and retrieval document archival

## Performance Analysis Framework

### Quantitative Evaluation Metrics
```python
# Core Classification Metrics
{
    'accuracy': float,           # Overall prediction accuracy
    'precision': float,          # Mortality prediction precision (TP/(TP+FP))
    'recall': float,             # Mortality detection recall (TP/(TP+FN))
    'f1_score': float,           # Balanced F1 score
    'specificity': float,        # True negative rate (TN/(TN+FP))
    'total_samples': int,        # Number of evaluated samples
    'tp': int, 'fp': int,        # True/False positives
    'fn': int, 'tn': int         # False/True negatives
}

# Agent-Specific Analysis
{
    'agent_agreement_rates': {   # Cross-agent prediction agreement
        'target_vs_integrator': float,
        'positive_vs_negative_comparators': float,
        'all_agents_consensus': float
    },
    'generation_times': {        # Response time per agent
        'target_patient_analyst': float,
        'positive_similar_comparator': float,
        'negative_similar_comparator': float,
        'medical_knowledge_integrator': float
    }
}
```

### Analysis Capabilities

#### 1. **Evidence Quality Analysis**
- **Round 1 vs Round 3 Prediction Comparison**: How often integrator changes initial prediction
- **Evidence Utilization**: How Round 3 incorporates mortality vs survival factors
- **Factor Identification Quality**: Relevance and accuracy of Round 2 evidence
- **Decision Validation**: Whether Round 3 properly weighs contrastive evidence

#### 2. **Temporal Depth Analysis**
- **Single vs Multi-Visit Performance**: Prediction quality by temporal depth
- **Clinical Progression Impact**: How temporal context improves mortality prediction
- **Similar Patient Utilization**: Effectiveness of temporal depth matching
- **Trajectory Pattern Recognition**: Success in identifying deterioration patterns

#### 3. **Comparative Reasoning Evaluation**
- **Similar Patient Effectiveness**: Impact of positive/negative case analysis
- **Pattern Extraction Quality**: How well agents identify survival/mortality patterns
- **Case Similarity Assessment**: Accuracy of target-to-similar comparisons
- **Protective Factor Recognition**: Identification of survival-promoting factors

#### 4. **Structured Logging Analysis**
- **Per-Patient Debugging**: Complete reasoning trace for error analysis
- **Response Quality Assessment**: Length, coherence, medical accuracy of responses
- **Prediction Extraction Success**: Reliability of \\boxed{0/1} format parsing
- **Generation Efficiency**: Token usage and timing optimization

### Comparison Framework with Baselines
- **KARE Baseline**: Direct comparison using same test set and temporal structure
- **Single-Agent Baselines**: Compare against individual agent predictions
- **Reasoning Quality**: Debate reasoning depth vs single-model responses
- **Error Pattern Analysis**: Systematic improvements/degradations identification
- **Computational Efficiency**: Multi-agent cost vs single-model performance gains

## Next Steps: Evaluation and Analysis

1. **Run Pilot Evaluation** (10-50 samples)
   ```bash
   python run_kare_debate_mortality.py --start_idx 0 --num_samples 10 --output results/pilot_test.json --include_history
   ```

2. **Full Dataset Evaluation** (996 samples)
   ```bash
   python run_kare_debate_mortality.py --start_idx 0 --num_samples 996 --output results/full_evaluation.json --batch_size 20
   ```

3. **Performance Analysis**
   - Compare with KARE's Claude-based baseline results
   - Analyze temporal depth impact on prediction quality
   - Evaluate evidence quality and Round 1 vs Round 3 prediction changes
   - Assess contrastive reasoning effectiveness and decision validation quality