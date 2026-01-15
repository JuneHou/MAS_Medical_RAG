# GPT Swap Experiments for KARE Mortality Prediction

This directory contains experimental code to explore whether issues in the existing multi-agent debate framework are caused by:
1. **Lacking reasoning ability** (Qwen vs GPT-4)
2. **RAG inaccuracy** (retrieval quality issues)

## Directory Structure

```
gpt/
├── src/
│   ├── extract_metadata.py       # Build candidate table from dataset + multi-agent logs
│   ├── sample_select.py           # Select diagnostic slice (100 samples)
│   ├── gpt_utils.py               # Utility functions for GPT API and retrieval
│   ├── test_run.py                # TEST on 3 samples + cost estimation
│   ├── run_condition_A.py         # Condition A: GPT+GPT+GPT (ceiling)
│   ├── run_condition_B.py         # Condition B: GPT+Qwen+GPT (retrieval test)
│   ├── run_condition_C.py         # Condition C: Qwen+GPT+GPT (analyst test)
│   ├── inspect_samples.py         # Analyze selected samples
│   └── analyze_conditions.py      # Compare all ablation conditions
├── cache/
│   ├── candidate_table.parquet    # Unified metadata table (996 samples)
│   ├── condition_A/               # Condition A results
│   ├── condition_B/               # Condition B results
│   ├── condition_C/               # Condition C results
│   └── ablation_analysis.json     # Comparative analysis
├── manifests/
│   ├── samples_swap_core.csv      # Selected 100 samples (sample_id, label, split_tag)
│   ├── samples_swap_core_metadata.json  # Selection criteria and stats
│   ├── selected_samples_full.parquet    # Full data for selected samples
│   └── gpt_experiment_samples.json      # Ready for GPT experiments
├── README.md                      # This file
├── EXPERIMENT_SUMMARY.md          # Initial findings
└── QUICKSTART.md                  # Usage guide
```

---

## Ablation Experiments (GPT-4 vs Qwen)

### Four-Condition Design

To isolate whether issues are due to **reasoning ability** or **retrieval quality**, we run four ablation conditions:

| Condition | Analysts | Retrieval | Integrator | Purpose |
|-----------|----------|-----------|------------|---------|
| **A** | GPT-4 | GPT-4 | GPT-4 | Ceiling performance (all GPT) |
| **B** | GPT-4 | Qwen | GPT-4 | Test retrieval quality |
| **C** | Qwen | GPT-4 | GPT-4 | Test analyst reasoning |
| **D** | GPT-4 | GPT-4 | Qwen | Test integrator reasoning |

**Baseline**: Qwen-Qwen-Qwen (41.0% accuracy on 100 samples, existing results from `pred_multi_rag_qwen`)

### Research Questions

1. **Does GPT-4 improve over Qwen?** → Compare Condition A vs Baseline (41.0%)
2. **Is Qwen retrieval limiting performance?** → Compare Condition A vs Condition B
3. **Are Qwen analysts limiting performance?** → Compare Condition A vs Condition C
4. **Is Qwen integrator limiting performance?** → Compare Condition A vs Condition D

### Running the Experiments

**Prerequisites**:
- Set `OPENAI_API_KEY` environment variable or use `--api_key` argument
- Install OpenAI library: `pip install openai`
- Run `extract_metadata.py` and `sample_select.py` first

**Step 0: Test Run (RECOMMENDED)**

Before running the full experiments, test on 3 samples to verify everything works and estimate costs:

```bash
# Test all conditions (recommended)
python src/test_run.py --gpt_model gpt-4-turbo-preview

# Or test individual conditions
python src/test_run.py --condition A
python src/test_run.py --condition B
python src/test_run.py --condition C
```

This will:
- Run 3 diverse samples through each condition
- Verify cache and log structure
- Estimate total cost for 100 samples
- Show cost breakdown with safety margin

**Step 1: Run Condition A (Ceiling)**
```bash
cd /data/wang/junh/githubs/Debate/KARE/gpt

python src/run_condition_A.py \
  --gpt_model gpt-4-turbo-preview \
  --k 8
```

**Step 2: Run Condition B (Retrieval Test)**
```bash
python src/run_condition_B.py \
  --gpt_model gpt-4-turbo-preview \
  --qwen_log_dir /data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8/debate_logs
```

**Step 3: Run Condition C (Analyst Test)**
```bash
python src/run_condition_C.py \
  --gpt_model gpt-4o \
  --qwen_log_dir /data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8/debate_logs
```

**Step 4: Run Condition D (Integrator Test)**
```bash
# Uses vLLM with Qwen integrator (same setup as mortality_debate_rag.py)
python src/run_condition_D.py \
  --qwen_model Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0
```

**Step 5: Analyze Results**
```bash
python src/analyze_conditions.py
```

### Output Structure

```
cache/
├── condition_A/           # GPT+GPT+GPT
│   ├── {sample_id}.json   # Individual results
│   └── summary.json       # Aggregate metrics
├── condition_B/           # GPT+Qwen+GPT
│   ├── {sample_id}.json
│   └── summary.json
├── condition_C/           # Qwen+GPT+GPT
│   ├── {sample_id}.json
│   └── summary.json
├── condition_D/           # GPT+GPT+Qwen
│   ├── {sample_id}.json
│   └── summary.json
└── ablation_analysis.json # Comparative analysis
```

### Input Template Consistency

**All system prompts are copied verbatim, and all integrator input templates use identical block ordering, document formatting, and truncation rules across conditions.** This ensures fair comparison:

- **System prompts**: Copied exactly from `mortality_debate_rag.py`
  - `mortality_risk_assessor` (Analyst 1)
  - `protective_factor_analyst` (Analyst 2)
  - `balanced_clinical_integrator` (Integrator)

- **Input template structure** (consistent across all conditions):
  - Analyst inputs: `{system_prompt}\n\n## Target Patient:\n{target_context}\n\n## Similar Patient:\n{similar_context}\n\nProvide your clinical analysis...`
  - Integrator initial: `{system_prompt}\n\n## Target Patient:\n{target_context}\n\n## Previous Analysis:\n### Analysis 1...\n### Analysis 2...\n\nProvide your clinical analysis...`
  - Integrator final: Same structure + `<information>\n{retrieved_docs}\n</information>` block before final instruction

- **Document formatting**: Retrieved documents use identical `[Document {rank}, Score: {score:.3f}]\n{content}` format
- **No truncation differences**: All conditions use same max_tokens (32768 for all agents, matching Qwen)

This ensures that any performance differences are due to the model's reasoning ability or retrieval quality, not template engineering or formatting artifacts.

### Interpreting Results

**If Condition A >> Baseline**:
- GPT-4 provides substantial improvement
- Qwen has significant reasoning limitations

**If Condition A ≈ Condition B**:
- Retrieval quality is NOT the bottleneck
- Both GPT and Qwen retrieval work similarly

**If Condition A >> Condition B**:
- Qwen retrieval is problematic
- GPT-generated queries or retrieved docs are better

**If Condition A ≈ Condition C**:
- Analyst reasoning is NOT the bottleneck
- Both GPT and Qwen analysts provide similar inputs

**If Condition A >> Condition C**:
- Qwen analysts are limiting performance
- GPT analysts provide better clinical analysis

**If Condition A ≈ Condition D**:
- Integrator reasoning is NOT the bottleneck
- Both GPT and Qwen integrators aggregate similarly

**If Condition A >> Condition D**:
- Qwen integrator is limiting performance
- GPT integrator provides better evidence synthesis

---

## Sample Selection

### Selection Strategy (100 samples)
- **54 positives**: All samples with `label=1` (mortality cases)
- **46 negatives**: Balanced difficulty selection
  - **23 hard**: Wrong in 2-3 models (1 from N3 + 22 from N2)
  - **23 easy**: Correct in all 3 models (N0)

**Negative Selection Algorithm** (deterministic):

1. **Hard Pool (23 samples)**: 
   - Take all N3 (wrong in all 3 models: CoT, RAG-Qwen, RAG-R1) = 1 sample
   - Fill remaining from N2 (wrong in 2 models), sorted by prompt length = 22 samples
   
2. **Easy Pool (23 samples)**:
   - Take top 23 from N0 (correct in all 3 models), sorted by retrieval usage + prompt length

**Rationale**: This balanced design allows testing whether GPT-4 can improve on both:
- **Hard cases** where Qwen consistently struggled (N3+N2)
- **Easy cases** where Qwen succeeded, testing if GPT maintains baseline performance (N0)

### Qwen Baseline Performance (on 100 Selected Samples)

**Overall Metrics**:
- **Accuracy**: 41.0% (41/100)
- **Precision**: 33.3% (5/15 predicted positive)
- **Recall**: 9.3% (5/54 actual positive)
- **F1 Score**: 0.145
- **Retrieval Usage**: 0/100 (0.0%) - Note: This Qwen run had retrieval disabled

**Performance by Difficulty**:
| Split | Accuracy | Correct | Total |
|-------|----------|---------|-------|
| Positive (all) | 9.3% | 5/54 | All mortality cases |
| Hard Negatives | 56.5% | 13/23 | Wrong in 2-3 models |
| Easy Negatives | 100% | 23/23 | Correct in all models |

**Key Observations**:
- Strong survival bias: Only 9.3% recall on mortality cases
- Perfect accuracy on easy negatives (by design - these were selected for being correct)
- Hard negatives show 56.5% accuracy, indicating moderate difficulty

---

## Data Generation Pipeline

### Step 1: Extract Metadata (`src/extract_metadata.py`)

**Purpose**: Build unified candidate table joining KARE dataset with multi-agent results.

**Inputs**:
- KARE test dataset (996 samples)
- Three multi-agent logs: CoT, RAG-Qwen, RAG-R1
- Debate log files (for retrieval detection)

**Output**: `cache/candidate_table.parquet`

**Usage**:
```bash
python src/extract_metadata.py
```

### Step 2: Sample Selection (`src/sample_select.py`)

**Purpose**: Select 100-sample diagnostic slice for GPT experiments.

**Output**:
- `manifests/samples_swap_core.csv`: Minimal manifest
- `manifests/samples_swap_core_metadata.json`: Selection stats
- `manifests/selected_samples_full.parquet`: Full data

**Usage**:
```bash
python src/sample_select.py
```

### Step 3: Inspect Samples (`src/inspect_samples.py`)

**Purpose**: Analyze and export selected samples for experiments.

**Usage**:
```bash
python src/inspect_samples.py
```

---

## Dependencies

- `pandas`: Data manipulation
- `pyarrow`: Parquet file I/O
- `openai`: GPT-4 API access
- `tqdm`: Progress bars

Install:
```bash
pip install pandas pyarrow openai tqdm
```

---

## Cost Estimation

### Expected Costs (100 samples)

Based on GPT-4 Turbo pricing ($0.01/1K input, $0.03/1K output):

| Condition | Input Tokens | Output Tokens | Estimated Cost |
|-----------|--------------|---------------|----------------|
| **A** (GPT+GPT+GPT) | ~1M | ~300K | ~$19 |
| **B** (GPT+Qwen+GPT) | ~600K | ~180K | ~$11 |
| **C** (Qwen+GPT+GPT) | ~600K | ~180K | ~$11 |
| **Total** | ~2.2M | ~660K | **~$41** |

**Recommended budget with 50% safety margin: ~$62**

These are rough estimates. Actual costs depend on:
- Response length variation across samples
- Retrieval usage rate (affects integrator input length)
- OpenAI's exact token counting

**Use `test_run.py` to get precise estimates based on your samples!**

---

## Notes

- All selections are **deterministic** (sorted by stable keys) for reproducibility
- Prompts are **exactly the same** as Qwen system for fair comparison
- The candidate table includes raw patient data for later analysis
- Results are saved incrementally to handle API failures gracefully
- Rate limiting (1 second delay) is built-in to be nice to OpenAI API
- **Test run first** to verify setup and estimate costs before burning credits

---

## Contact & Support

**Scripts**: All located in `/data/wang/junh/githubs/Debate/KARE/gpt/src/`
**Data**: All located in `/data/wang/junh/githubs/Debate/KARE/gpt/manifests/`
**Documentation**: README.md, EXPERIMENT_SUMMARY.md, QUICKSTART.md

For questions, refer to code comments or documentation files.
