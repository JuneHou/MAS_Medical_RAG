                    #!/usr/bin/env python3
"""
KARE Data Adapter for Multi-Agent Debate Integration
Loads KARE test data and similar patient contexts for mortality prediction.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class KAREDataAdapter:
    """
    Adapter class to load and format KARE's mortality prediction data
    for integration with multi-agent debate system.
    """
    
    def __init__(self, base_path: str = "./data", split: str = "test"):
        """
        Initialize KARE data adapter.
        
        Args:
            base_path: Base path to KARE dataset directory (relative to script location)
            split: Data split to load ('train', 'val', or 'test')
        """
        self.base_path = Path(base_path)
        self.ehr_data_path = self.base_path / "ehr_data"
        self.patient_context_path = self.base_path / "patient_context" / "similar_patient_qwen"
        self.split = split
        
        # Load appropriate split data
        if split == "train":
            self.data = self._load_train_data()
        elif split == "val":
            self.data = self._load_val_data()
        else:
            self.data = self._load_test_data()
        
        # Keep test_data reference for backward compatibility
        self.test_data = self.data
        
        self.similar_patients = self._load_similar_patients()
        
        print(f"Loaded {len(self.data)} {split} samples")
        print(f"Loaded similar patient contexts for {len(self.similar_patients)} patients")
    
    def _load_train_data(self) -> List[Dict[str, Any]]:
        """Load MIMIC-III mortality training data."""
        train_file = self.ehr_data_path / "mimic3_mortality_samples_train.json"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Train data not found: {train_file}")
        
        with open(train_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _load_val_data(self) -> List[Dict[str, Any]]:
        """Load MIMIC-III mortality validation data."""
        val_file = self.ehr_data_path / "mimic3_mortality_samples_val.json"
        
        if not val_file.exists():
            raise FileNotFoundError(f"Val data not found: {val_file}")
        
        with open(val_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load MIMIC-III mortality test data."""
        test_file = self.ehr_data_path / "mimic3_mortality_samples_test.json"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _load_similar_patients(self) -> Dict[str, str]:
        """Load precomputed similar patient contexts."""
        # Use improved similar patient file with better positive/negative balance (top-1)
        similar_file = self.base_path / "patient_context" / "similar_patient_debate" / "patient_to_top_1_patient_contexts_mimic3_mortality_improved.json"
        
        # Fallback to original file if improved version not available
        if not similar_file.exists():
            similar_file = self.patient_context_path / "patient_to_top_1_patient_contexts_mimic3_mortality.json"
            print(f"Using original similar patient file: {similar_file}")
        else:
            print(f"Using improved similar patient file: {similar_file}")
        
        if not similar_file.exists():
            print(f"Warning: Similar patient contexts not found: {similar_file}")
            return {}
        
        with open(similar_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def format_patient_context(self, patient_data: Dict[str, Any]) -> str:
        """
        Format patient data into KARE's temporal rolling visit context format.
        
        Args:
            patient_data: Patient visit data from test set with rolling visits
            
        Returns:
            Formatted patient context string following KARE's base_context.py format
        """
        patient_id = patient_data.get('patient_id', 'Unknown')
        visit_id = patient_data.get('visit_id', 'Unknown')
        
        # Extract rolling visit arrays
        conditions_visits = patient_data.get('conditions', [[]])
        procedures_visits = patient_data.get('procedures', [[]])  
        drugs_visits = patient_data.get('drugs', [[]])
        
        # Determine number of visits (rolling temporal context)
        num_visits = max(len(conditions_visits), len(procedures_visits), len(drugs_visits))
        
        # Format according to KARE's base_context.py temporal style
        context_parts = [f"Patient ID: {patient_id}"]
        
        # Track items across visits for "new" vs "continued" annotations
        seen_conditions = set()
        seen_procedures = set()
        seen_drugs = set()
        
        # Process each visit in temporal order
        for visit_num in range(num_visits):
            context_parts.extend(["", f"Visit {visit_num}:"])
            
            # Process conditions for this visit
            context_parts.append("Conditions:")
            visit_conditions = conditions_visits[visit_num] if visit_num < len(conditions_visits) else []
            if visit_conditions:
                for i, condition in enumerate(visit_conditions, 1):
                    if condition in seen_conditions:
                        context_parts.append(f"{i}. {condition} (continued from previous visit)")
                    elif visit_num > 0:
                        context_parts.append(f"{i}. {condition} (new)")
                        seen_conditions.add(condition)
                    else:
                        context_parts.append(f"{i}. {condition}")
                        seen_conditions.add(condition)
            else:
                context_parts.append("No documented conditions.")
            
            # Process procedures for this visit
            context_parts.extend(["", "Procedures:"])
            visit_procedures = procedures_visits[visit_num] if visit_num < len(procedures_visits) else []
            if visit_procedures:
                for i, procedure in enumerate(visit_procedures, 1):
                    if procedure in seen_procedures:
                        context_parts.append(f"{i}. {procedure} (continued from previous visit)")
                    elif visit_num > 0:
                        context_parts.append(f"{i}. {procedure} (new)")
                        seen_procedures.add(procedure)
                    else:
                        context_parts.append(f"{i}. {procedure}")
                        seen_procedures.add(procedure)
            else:
                context_parts.append("No documented procedures.")
            
            # Process medications for this visit
            context_parts.extend(["", "Medications:"])
            visit_drugs = drugs_visits[visit_num] if visit_num < len(drugs_visits) else []
            if visit_drugs:
                for i, drug in enumerate(visit_drugs, 1):
                    if drug in seen_drugs:
                        context_parts.append(f"{i}. {drug} (continued from previous visit)")
                    elif visit_num > 0:
                        context_parts.append(f"{i}. {drug} (new)")
                        seen_drugs.add(drug)
                    else:
                        context_parts.append(f"{i}. {drug}")
                        seen_drugs.add(drug)
            else:
                context_parts.append("No documented medications.")
        
        return "\n".join(context_parts)
    
    def get_test_sample(self, index: int) -> Dict[str, Any]:
        """
        Get formatted test sample for debate system.
        
        Args:
            index: Index of test sample
            
        Returns:
            Dictionary containing patient context, similar patients, and ground truth
        """
        if index >= len(self.test_data):
            raise IndexError(f"Index {index} out of range (max: {len(self.test_data)-1})")
        
        patient_data = self.test_data[index]
        
        # KARE uses patient_id + "_" + visit_index format for temporal instances
        base_patient_id = str(patient_data['patient_id'])
        visit_id = str(patient_data['visit_id'])
        
        # Determine visit index from rolling context (number of visits - 1)
        conditions_visits = patient_data.get('conditions', [[]])
        procedures_visits = patient_data.get('procedures', [[]])
        drugs_visits = patient_data.get('drugs', [[]])
        num_visits = max(len(conditions_visits), len(procedures_visits), len(drugs_visits))
        visit_index = num_visits - 1  # 0-based index
        
        # Construct KARE's temporal patient ID format
        kare_patient_id = f"{base_patient_id}_{visit_index}"
        
        # Format patient context with temporal rolling visits
        patient_context = self.format_patient_context(patient_data)
        
        # Get similar patient contexts using KARE's patient ID format
        similar_patients_data = self.similar_patients.get(kare_patient_id, {})
        
        # Format separate positive and negative similar patient blocks
        if similar_patients_data:
            positive_patients = similar_patients_data.get('positive', [])
            negative_patients = similar_patients_data.get('negative', [])
            
            # Create separate blocks for positive and negative similar patients
            if positive_patients and positive_patients[0] != "None":
                positive_block = "\n\n".join(positive_patients)
            else:
                positive_block = "No positive similar patients available."
                
            if negative_patients and negative_patients[0] != "None":
                negative_block = "\n\n".join(negative_patients)
            else:
                negative_block = "No negative similar patients available."
        else:
            positive_block = "No positive similar patients available."
            negative_block = "No negative similar patients available."
        
        # Extract ground truth label
        ground_truth = patient_data.get('label', 0)
        
        return {
            'patient_id': kare_patient_id,
            'base_patient_id': base_patient_id,
            'visit_id': visit_id,
            'visit_index': visit_index,
            'target_context': patient_context,
            'positive_similars': positive_block,
            'negative_similars': negative_block,
            'ground_truth': ground_truth,
            'original_data': patient_data
        }
    
    def get_batch_samples(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get batch of test samples.
        
        Args:
            start_idx: Starting index
            batch_size: Number of samples to retrieve
            
        Returns:
            List of formatted test samples
        """
        end_idx = min(start_idx + batch_size, len(self.test_data))
        return [self.get_test_sample(i) for i in range(start_idx, end_idx)]
    
    def get_task_description(self) -> str:
        """Get mortality prediction task description."""
        return """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
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

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.

CRITICAL CONSERVATIVE GUIDELINE: Mortality is relatively rare in this population. When uncertain, err toward survival prediction (0). Strong evidence is required to predict mortality (1).
"""
    
    def format_as_retrieval_query(self, patient_context: str, ground_truth: int = None) -> str:
        """
        Format patient context as retrieval query with label conversion.
        
        Args:
            patient_context: Formatted patient temporal context
            ground_truth: Ground truth label (0 or 1), will be converted to text
            
        Returns:
            Patient context with converted labels for retrieval
        """
        query = patient_context
        
        # Convert labels in the query text
        query = query.replace("label: 0", "label: survive")
        query = query.replace("label:0", "label: survive")
        query = query.replace("labels: 0", "labels: survive")  
        query = query.replace("mortality = 0", "mortality = survive")
        query = query.replace("mortality=0", "mortality = survive")
        
        query = query.replace("label: 1", "label: mortality")
        query = query.replace("label:1", "label: mortality")
        query = query.replace("labels: 1", "labels: mortality")
        query = query.replace("mortality = 1", "mortality = mortality") 
        query = query.replace("mortality=1", "mortality = mortality")
        
        # If ground truth is provided, append it in converted format
        if ground_truth is not None:
            label_text = "survive" if ground_truth == 0 else "mortality"
            query += f"\n\nGround truth outcome: {label_text}"
        
        return query
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_samples = len(self.test_data)
        
        # Count labels
        label_counts = {}
        for sample in self.test_data:
            label = sample.get('label', 0)
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Count patients with similar contexts
        patients_with_similar = sum(1 for patient_id in [str(s['patient_id']) for s in self.test_data] 
                                  if patient_id in self.similar_patients)
        
        return {
            'total_samples': total_samples,
            'label_distribution': label_counts,
            'patients_with_similar_contexts': patients_with_similar,
            'similar_context_coverage': patients_with_similar / total_samples if total_samples > 0 else 0
        }

# Test the adapter
if __name__ == "__main__":
    try:
        adapter = KAREDataAdapter()
        
        # Print statistics
        stats = adapter.get_statistics()
        print("\nDataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Similar context coverage: {stats['similar_context_coverage']:.2%}")
        
    except Exception as e:
        print(f"Error testing adapter: {e}")
        import traceback
        traceback.print_exc()