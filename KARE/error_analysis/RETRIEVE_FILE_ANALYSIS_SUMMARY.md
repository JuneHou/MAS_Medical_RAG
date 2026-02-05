# Retrieve File Missing Analysis Summary

## Overview
Analysis of 968 patients across 5 roles to understand why some retrieve files are missing despite retrieval attempts in debate logs.

## Key Findings

### 1. **File Writing Failures (Most Common)**
The vast majority of missing retrieve files show evidence of retrieval attempts in the debate logs, but the files were never written.

**Evidence:**
- **target_patient_analyst**: 2/2 missing files have retrieval attempts
- **mortality_risk_assessor**: 4/4 missing files have retrieval attempts  
- **protective_factor_analyst**: 4/4 missing files have retrieval attempts
- **mortality_assessment**: 14/15 missing files have retrieval attempts
- **survival_assessment**: 32/33 missing files have retrieval attempts

**Total: 56/58 patients (96.6%) show retrieval attempts but no saved files**

### 2. **Retrieval System Failures (Rare)**
Only 1 patient (14741_0) shows evidence of retrieval system failure:
- Logs show "Mortality Retrieved Documents: 0" and "Survival Retrieved Documents: 0"
- This patient is missing both mortality_assessment and survival_assessment retrieve files
- The retrieval system returned 0 documents, suggesting a query/corpus issue

### 3. **Distribution of Missing Files by Role**

| Role | Total Patients | Missing Files | % Missing |
|------|---------------|---------------|-----------|
| target_patient_analyst | 968 | 2 | 0.21% |
| mortality_risk_assessor | 968 | 4 | 0.41% |
| protective_factor_analyst | 968 | 4 | 0.41% |
| mortality_assessment | 968 | 15 | 1.55% |
| survival_assessment | 968 | 33 | 3.41% |

**Pattern**: Integrator roles (mortality_assessment, survival_assessment) have higher missing file rates than analyst roles.

### 4. **Example Cases**

#### Patient 7900_0 (File Writing Failure)
- **Missing files**: target_patient_analyst, mortality_risk_assessor, protective_factor_analyst
- **Evidence in log**: 10 retrieve calls with queries like:
  - "acute cerebrovascular disease mortality risk factors"
  - "cardiac dysrhythmias mortality risk factors"
- **Error found**: `NameError: name 'retrieved_evidence_cardiac_dysrhythmias' is not defined`
- **Conclusion**: Retrieval succeeded, but file writing failed (possibly due to code error)

#### Patient 84467_0 (File Writing Failure)
- **Missing files**: target_patient_analyst, mortality_risk_assessor, protective_factor_analyst
- **Evidence in log**: 10 retrieve calls with queries like:
  - "gastrointestinal hemorrhage mortality risk factors"
- **No errors found in log**
- **Conclusion**: Retrieval succeeded, file writing failed (unknown cause)

#### Patient 14741_0 (Retrieval System Failure)
- **Missing files**: mortality_assessment, survival_assessment
- **Evidence in log**: 
  - Shows `retrieve(query)` calls
  - Shows "Mortality Retrieved Documents: 0"
  - Shows "Survival Retrieved Documents: 0"
- **Conclusion**: Retrieval system returned 0 documents

## Root Causes

### File Writing Failures (96.6% of missing files)
Possible causes:
1. **Code errors during retrieval**: As seen in patient 7900_0, `NameError` exceptions may interrupt file writing
2. **Disk I/O errors**: Temporary disk issues preventing file writes
3. **Race conditions**: Multiple processes trying to write simultaneously
4. **Silent exceptions**: File writing code may have unhandled exceptions
5. **Missing log_dir/qid**: If `log_dir` or `qid` parameters are None, files won't be saved (per code in mortality_debate_rag.py lines 305-320)

### Retrieval System Failures (1.7% of missing files)
- Query formulation issues leading to 0 results
- Corpus availability problems
- MedRAG retrieval system bugs

## Recommendations

1. **Add robust error handling** around file writing in `_create_retrieval_tool` function
2. **Add logging** for all file write attempts (success/failure)
3. **Add retry logic** for file writing failures
4. **Validate log_dir and qid** before attempting retrieval
5. **Monitor retrieval results**: Log when 0 documents are retrieved
6. **Add file existence verification** after retrieval to detect missing files early

## Statistics Summary

- **Total patients analyzed**: 968
- **Total missing retrieve files**: 58 (2.0% of expected 2,900 files)
- **Patients with file writing failures**: 57 (98.3%)
- **Patients with retrieval system failures**: 1 (1.7%)
- **Patients completely missing all retrieve files**: 0 (all have at least some retrieve files)

## Code Locations

- Retrieve tool creation: `mortality_debate_rag.py` lines 230-320
- File saving logic: Lines 305-320 (only saves if `log_dir` and `qid` are provided)
- Retrieve file format: `retrieve_{role}_{patient_id}.json`
