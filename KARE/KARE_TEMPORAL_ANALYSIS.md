# KARE Multi-Visit Temporal Handling and Similar Patient Matching Analysis

## Overview
Based on analysis of KARE's codebase, I can now provide a comprehensive explanation of how KARE handles multi-visit patients and similar patient retrieval.

## 1. Multi-Visit Data Structure in KARE

### PyHealth Mortality Prediction Task (`mortality_prediction_mimic3_fn`)
```python
def mortality_prediction_mimic3_fn(patient):
    samples = []
    # Drop the last visit (no next visit to predict)
    for i in range(len(patient) - 1):
        visit = patient[i]
        next_visit = patient[i + 1]
        
        # Label = mortality outcome of NEXT visit
        mortality_label = int(next_visit.discharge_status) if next_visit.discharge_status in [0, 1] else 0
        
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,  
            "conditions": visit.get_code_list(table="DIAGNOSES_ICD"),
            "procedures": visit.get_code_list(table="PROCEDURES_ICD"),
            "drugs": visit.get_code_list(table="PRESCRIPTIONS"),
            "label": mortality_label,  # Next-visit mortality outcome
        })
    return samples
```

**Key Points:**
- Each sample represents a **single visit** predicting the **next visit's outcome**
- For a patient with visits [V0, V1, V2], PyHealth creates:
  - Sample 1: V0 data → V1 outcome
  - Sample 2: V1 data → V2 outcome
- This creates temporal prediction instances, NOT rolling visit windows

### KARE's Patient Context Creation (`base_context.py`)
```python
def process_dataset(sample_dataset, condition_dict, procedure_dict, drug_dict):
    patient_to_index = sample_dataset.patient_to_index
    
    for patient, idxs in patient_to_index.items():
        for i in range(len(idxs)):  # For each prediction instance
            patient_id = patient + f"_{i}"  # e.g., "10004_0", "10004_1"
            label = sample_dataset.samples[idxs[i]]['label']
            
            # Create ROLLING VISIT CONTEXT: visits [0] to [i]
            for j in range(i+1):
                idx = idxs[j]
                data = sample_dataset.samples[idx]
                patient_data[patient_id][f'visit {j}'] = {
                    'conditions': expand_and_map(data['conditions'], condition_dict),
                    'procedures': expand_and_map(data['procedures'], procedure_dict),
                    'drugs': expand_and_map(data['drugs'], drug_dict)
                }
```

**CRITICAL INSIGHT:**
KARE creates **ROLLING VISIT CONTEXTS**, not single visits:
- `patient_10004_0`: Uses only visit 0 data → predicts visit 1 outcome
- `patient_10004_1`: Uses visits 0+1 data → predicts visit 2 outcome  
- `patient_10004_2`: Uses visits 0+1+2 data → predicts visit 3 outcome

## 2. Similar Patient Definition and Retrieval

### Patient ID Structure
```
Original Patient: "10004" with 3 visits
KARE Patient IDs: 
- "10004_0" (using visit 0, predicting visit 1 outcome)
- "10004_1" (using visits 0+1, predicting visit 2 outcome)  
- "10004_2" (using visits 0+1+2, predicting visit 3 outcome)
```

### Similar Patient Matching (`sim_patient_ret_faiss.py`)
```python
for patient_id in target_patient_ids:  # e.g., "10004_1"
    # Skip same patient OR other instances of same patient
    for neighbor_idx in I[0]:
        neighbor_id = patient_ids[neighbor_idx]  # e.g., "15967_0"
        
        # KEY EXCLUSION LOGIC:
        if neighbor_id == patient_id or neighbor_id.split("_")[0] == patient_id.split("_")[0]:
            continue  # Skip "10004_1" itself AND "10004_0", "10004_2"
```

**Similar Patient Selection:**
1. **Target**: `10004_1` (patient 10004 using visits 0+1)
2. **Valid Similar Patients**: Any other patient instances like `15967_0`, `22831_2`, etc.
3. **Excluded**: All instances of patient 10004 (`10004_0`, `10004_1`, `10004_2`)

## 3. Temporal Context Handling

### Example Multi-Visit Patient Context:
```
Patient ID: 10004_1

Visit 0:
Conditions:
1. Acute myocardial infarction
2. Heart failure
Procedures:  
1. Cardiac catheterization
Medications:
1. ACE inhibitors
2. Beta blockers

Visit 1:  
Conditions:
1. Acute myocardial infarction (continued from previous visit)
2. Heart failure (continued from previous visit)  
3. Acute kidney injury (new)
Procedures:
1. Hemodialysis (new)
Medications:
1. ACE inhibitors (continued from previous visit)
2. Beta blockers (continued from previous visit)
3. Diuretics (new)

Label: 1  # Mortality in visit 2
```

### Similar Patient Example:
```
Similar Patient (15967_0):
Patient ID: 15967_0

Visit 0:
Conditions: 
1. Intracranial injury
2. Cardiac dysrhythmias
3. Essential hypertension
...
Label: 1  # Also had mortality in next visit
```