# Quick Start Guide - GPT Swap Experiments

## TL;DR - Run Before Full Experiments

```bash
# 1. Setup OpenAI API key
export OPENAI_API_KEY="your_key_here"

# 2. Test on 3 samples first (costs ~$0.50, estimates full cost)
cd /data/wang/junh/githubs/Debate/KARE/gpt
python src/test_run.py

# 3. If test passes, run full experiments (100 samples, ~$41)
python src/run_condition_A.py
python src/run_condition_B.py
python src/run_condition_C.py
python src/analyze_conditions.py
```

---

## Setup Complete! ✓

You now have a complete experimental framework to test whether issues in the multi-agent debate system are due to:
1. **Reasoning limitations** (Qwen 2.5-7B vs GPT-4)
2. **RAG accuracy issues** (retrieval quality problems)

---

## Generated Files

```
gpt/
├── cache/
│   └── candidate_table.parquet          # 996 samples with metadata
├── manifests/
│   ├── samples_swap_core.csv            # 100 selected samples (minimal)
│   ├── samples_swap_core_metadata.json  # Selection statistics
│   ├── selected_samples_full.parquet    # Full data for 100 samples
│   └── gpt_experiment_samples.json      # Ready for GPT experiments
├── src/
│   ├── extract_metadata.py              # Build candidate table
│   ├── sample_select.py                 # Select diagnostic slice
│   └── inspect_samples.py               # Analyze selected samples
├── README.md                            # Pipeline documentation
├── EXPERIMENT_SUMMARY.md                # Initial findings
└── QUICKSTART.md                        # This file
```

---

## Test Run (CRITICAL - Run This First!)

**Before spending ~$41 on the full experiment**, test on 3 samples:

```bash
export OPENAI_API_KEY="your_key_here"
cd /data/wang/junh/githubs/Debate/KARE/gpt

# Test all conditions
python src/test_run.py --gpt_model gpt-4-turbo-preview

# Or test individual conditions
python src/test_run.py --condition A
python src/test_run.py --condition B
python src/test_run.py --condition C
```

**What it does:**
1. Runs 3 diverse samples through each condition
2. Verifies cache directory structure is correct
3. Tests that Qwen logs can be loaded
4. Estimates total cost for 100 samples
5. Shows cost breakdown with safety margin

**Expected output:**
```
TESTING CONDITION A: GPT+GPT+GPT (Ceiling)
Processing sample 1/3: 10117_0
  ✓ Prediction: 0
  ✓ Mortality Prob: 0.35
  ✓ Retrieval used: True
...

COST ESTIMATE
Condition A (3 samples):
  Input tokens:  24,000
  Output tokens: 9,000
  Cost: $0.51

ESTIMATED FULL EXPERIMENT COST (100 samples):
Condition A: $17.00
Condition B: $10.00
Condition C: $10.00

TOTAL ESTIMATED COST: $41.00
⚠ RECOMMENDED BUDGET (with 50% safety margin): $61.50

✓ All tests passed! Ready to run full experiments.
```

**If test fails:**
- Check `OPENAI_API_KEY` is set correctly
- Verify Qwen log directory exists
- Check MedRAG is installed and accessible
- Review error messages in output

---

## Usage

### 1. Re-run Pipeline (if needed)

```bash
cd /data/wang/junh/githubs/Debate/KARE/gpt

# Step 1: Extract metadata from dataset + results
python src/extract_metadata.py

# Step 2: Select 100-sample diagnostic slice
python src/sample_select.py

# Step 3: Inspect and analyze selected samples
python src/inspect_samples.py
```

### 2. Load Selected Samples in Your Code

```python
import pandas as pd
import json

# Option 1: Load minimal manifest (sample_id, label, split_tag)
manifest = pd.read_csv('manifests/samples_swap_core.csv')

# Option 2: Load full data with patient context
full_data = pd.read_parquet('manifests/selected_samples_full.parquet')

# Option 3: Load JSON format for GPT experiments
with open('manifests/gpt_experiment_samples.json', 'r') as f:
    samples = json.load(f)

# Example: Get patient context for a specific sample
sample = full_data[full_data['sample_id'] == '10774_5'].iloc[0]
print(sample['patient_context'])
print(sample['positive_similars'])
print(sample['negative_similars'])
```

---

## Key Findings from Selected 100 Samples

### Difficulty Distribution
- **38 samples**: All 3 models wrong (hardest cases)
- **48 samples**: 2 models wrong (moderately hard)
- **14 samples**: 1 model wrong (edge cases)

### Critical Discovery: Mortality Prediction is Hardest
All top 10 hardest cases are **positive samples (mortality=1)** that all models predicted as survival (0).

This suggests:
- Models have a **strong survival bias**
- Mortality signals may be subtle/difficult to detect
- This is the key failure mode to investigate with GPT-4

### Retrieval Patterns in Selected Samples
- **RAG-Qwen**: 71/100 (71%) called retrieval
- **RAG-R1**: 46/100 (46%) called retrieval  
- **Both**: 31/100 (31%) both called retrieval

Higher than overall dataset average, suggesting harder cases trigger more retrieval attempts.

### Performance by Negative Difficulty

| Split Tag | Count | CoT Acc | RAG-Qwen Acc | RAG-R1 Acc | Avg Tokens |
|-----------|-------|---------|--------------|------------|------------|
| neg_wrong3 | 1 | 0% | 0% | 0% | 368 |
| neg_wrong2 | 35 | 34.3% | 14.3% | 51.4% | 1,643 |
| neg_wrong1 | 10 | 80% | 20% | 100% | 8,190 |

**Observation**: RAG-Qwen performs **significantly worse** than both CoT and RAG-R1, especially on moderately hard cases (neg_wrong2). This supports the hypothesis that RAG may be introducing errors rather than helping.

---

## Next Steps for GPT Experiments

### Experiment 1: Reasoning Ability Test
**Goal**: Test if GPT-4 can solve cases where Qwen 2.5-7B failed

**Method**:
1. Run GPT-4 on all 100 selected samples (especially the 38 mortality cases)
2. Compare GPT-4 vs Qwen 2.5-7B on same cases
3. Measure accuracy improvement

**Expected Outcome**:
- If reasoning is the issue: GPT-4 should improve significantly on mortality cases
- If not: GPT-4 and Qwen 2.5-7B should have similar failure modes

### Experiment 2: RAG Accuracy Test
**Goal**: Determine if retrieved information helps or hurts

**Method**:
1. Compare GPT-4 with RAG vs GPT-4 CoT-only on same 100 samples
2. Manually inspect retrieved documents for failed cases
3. Analyze correlation between retrieval and errors

**Expected Outcome**:
- If RAG accuracy is the issue: 
  - GPT-4 CoT might outperform GPT-4 RAG
  - Retrieved docs might be irrelevant/contradictory
- If RAG helps:
  - GPT-4 RAG should outperform GPT-4 CoT
  - Retrieved docs should be relevant

### Experiment 3: Retrieval Quality Analysis
**Goal**: Understand why RAG-Qwen performs worse despite more retrieval

**Method**:
1. Sample retrieved documents from neg_wrong2 cases (35 samples)
2. Score relevance and quality manually
3. Compare Qwen vs R1 retrieval quality

**Focus Areas**:
- Document relevance to mortality prediction
- Contradictions in retrieved evidence
- Noise vs signal ratio

---

## How to Use This for Your Paper

### Methods Section
- Cite `manifests/samples_swap_core_metadata.json` for selection criteria
- Reference deterministic selection algorithm
- Document difficulty stratification (N3, N2, N1 pools)

### Results Section
- Report baseline performance on 100 samples (from `EXPERIMENT_SUMMARY.md`)
- Compare GPT-4 vs Qwen 2.5-7B performance
- Analyze failure modes by split_tag

### Analysis Section
- Use `inspect_samples.py` to generate custom views
- Correlate retrieval usage with accuracy
- Identify mortality prediction as key failure mode

---

## Questions to Answer

1. **Can GPT-4 solve the 38 mortality cases where all models failed?**
   - If yes → reasoning limitation confirmed
   - If no → deeper issue with mortality signals in data

2. **Does retrieval help or hurt GPT-4?**
   - Compare GPT-4+RAG vs GPT-4 CoT
   - Measure impact on accuracy

3. **Why does RAG-Qwen underperform?**
   - Retrieval quality issue?
   - Model-specific reasoning limitation?
   - Combination of both?

4. **What makes mortality cases so hard?**
   - Lack of clear signals in EHR?
   - Subtle patterns requiring deeper reasoning?
   - Data quality/annotation issues?

---

## Contact & Support

**Scripts**: All located in `/data/wang/junh/githubs/Debate/KARE/gpt/src/`
**Data**: All located in `/data/wang/junh/githubs/Debate/KARE/gpt/manifests/`
**Documentation**: README.md, EXPERIMENT_SUMMARY.md, QUICKSTART.md

For questions or issues, refer to the code comments or documentation files.

---

*Happy experimenting! 🚀*
