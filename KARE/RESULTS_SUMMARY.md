# Single-Agent KARE Mortality Prediction Results

## Summary of Updated Results (After Re-parsing with Improved Extraction)

### Chain-of-Thought (CoT) Mode
**File**: `results/single_cot_mor_Qwen_Qwen2.5_7B_Instruct/results.json`

**Updated Metrics** (996 total patients):
- **Accuracy**: 0.7209 (718/996)
- **Precision**: 0.0447 (11/246)  
- **Recall**: 0.2037 (11/54)
- **F1 Score**: 0.0733
- **Specificity**: 0.7505 (707/942)

**Extraction Recovery**:
- Originally had 250 None predictions (counted as wrong)
- Recovered **241 predictions** with improved extraction patterns
- Only **13 fallback predictions** remain (use opposite of ground truth)
- **Valid predictions**: 983/996 (98.7%)

**Confusion Matrix**:
- True Positives (TP): 11
- False Positives (FP): 235
- True Negatives (TN): 707
- False Negatives (FN): 43

---

### RAG Mode (with MedCPT Retrieval)
**File**: `results/single_rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT/results.json`

**Updated Metrics** (996 total patients):
- **Accuracy**: 0.6556 (653/996)
- **Precision**: 0.0687 (23/335)
- **Recall**: 0.4259 (23/54)
- **F1 Score**: 0.1183
- **Specificity**: 0.6688 (630/942)

**Extraction Recovery**:
- Originally had 148 None predictions
- Previously recovered 132 predictions (from earlier run)
- Remaining **22 fallback predictions** (use opposite of ground truth)
- **Valid predictions**: 974/996 (97.8%)

**Confusion Matrix**:
- True Positives (TP): 23
- False Positives (FP): 312
- True Negatives (TN): 630
- False Negatives (FN): 31

---

## Mode Comparison

| Metric | CoT | RAG | Winner |
|--------|-----|-----|--------|
| Accuracy | **0.7209** | 0.6556 | CoT +6.5% |
| Precision | 0.0447 | **0.0687** | RAG +54% |
| Recall | 0.2037 | **0.4259** | RAG +109% |
| F1 Score | 0.0733 | **0.1183** | RAG +61% |
| Specificity | **0.7505** | 0.6688 | CoT +8.2% |
| Valid Predictions | **98.7%** | 97.8% | CoT |

**Key Findings**:

1. **CoT has higher accuracy** (72.1% vs 65.6%) - better at identifying survival cases
2. **RAG has better recall** (42.6% vs 20.4%) - catches more mortality cases
3. **RAG has higher F1 score** (0.118 vs 0.073) - better balanced performance
4. **Both modes are conservative** - low precision indicates many false alarms
5. **Extraction recovery was successful** - 97.8-98.7% valid predictions

**Clinical Interpretation**:
- CoT mode is more conservative (fewer mortality predictions, higher specificity)
- RAG mode with retrieval evidence is more sensitive (higher recall, catches more at-risk patients)
- Neither achieves high precision - both generate many false positives
- For mortality screening, RAG's higher recall may be more valuable despite lower accuracy

---

## Improved Fallback Logic

Both scripts now implement the same fallback rules:

1. **Both probabilities available**: Use mort_prob vs surv_prob comparison (normal case)
2. **Both probabilities None**: Fallback to opposite of ground truth (marked as fallback)
3. **One probability > 0.5**: Take that label as prediction
4. **One probability < 0.5**: Fallback to opposite of ground truth (marked as fallback)

All predictions are marked with `is_fallback` flag for transparency.

---

## Updated Code Files

1. **mortality_single_agent_cot.py**: Improved extraction patterns + fallback logic
2. **mortality_single_agent_rag.py**: Improved extraction patterns + fallback logic
3. **reparse_cot_results.py**: Re-parse CoT logs with improved extraction
4. **reparse_rag_results.py**: Re-parse RAG logs with improved extraction

