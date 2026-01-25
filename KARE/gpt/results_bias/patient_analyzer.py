"""
Patient EHR Data Analyzer

This module provides comprehensive analysis of patient EHR data including:
- Ground truth labels
- Visit and concept counts
- Model predictions
- ICD-9 to CCS category mapping for next visit prediction
"""

import json
import os
import pandas as pd


class PatientAnalyzer:
    """Analyzer for patient EHR data with MIMIC-III integration."""
    
    def __init__(self, mimic_path="/data/wang/junh/datasets/physionet.org/files/mimiciii/1.4",
                 ccs_resources="/data/wang/junh/githubs/KARE/kg_construct/resources"):
        """
        Initialize the analyzer with paths to data resources.
        
        Args:
            mimic_path: Path to MIMIC-III data directory
            ccs_resources: Path to CCS mapping resource files
        """
        self.mimic_path = mimic_path
        self.ccs_resources = ccs_resources
        
        # File paths
        self.diagnoses_file = os.path.join(mimic_path, "DIAGNOSES_ICD.csv")
        self.admissions_file = os.path.join(mimic_path, "ADMISSIONS.csv")
        self.icd9_to_ccs_file = os.path.join(ccs_resources, "ICD9CM_to_CCSCM.csv")
        self.ccs_labels_file = os.path.join(ccs_resources, "CCSCM.csv")
        
        # Load mapping data
        self._load_mappings()
    
    def _load_mappings(self):
        """Load CCS mapping files."""
        try:
            icd9_to_ccs = pd.read_csv(self.icd9_to_ccs_file, dtype=str)
            ccs_labels = pd.read_csv(self.ccs_labels_file)
            
            self.icd9_to_ccs_dict = dict(zip(icd9_to_ccs['ICD9CM'], icd9_to_ccs['CCSCM']))
            self.ccs_code_to_name = dict(zip(ccs_labels['code'].astype(str), ccs_labels['name']))
        except Exception as e:
            print(f"Warning: Could not load CCS mappings: {e}")
            self.icd9_to_ccs_dict = {}
            self.ccs_code_to_name = {}
    
    @staticmethod
    def format_icd9_code(code):
        """
        Format ICD-9 code by adding decimal point.
        
        MIMIC-III stores codes without decimals (e.g., '5789')
        CCS mapping expects format with decimals (e.g., '578.9')
        
        Rules:
        - E codes (E####): E + 3 digits + decimal + remaining (e.g., E9359 -> E935.9)
        - V codes (V####): V + 2 digits + decimal + remaining (e.g., V4581 -> V45.81)
        - Regular codes: 3 digits + decimal + remaining (e.g., 5789 -> 578.9)
        
        Args:
            code: ICD-9 code without decimal
            
        Returns:
            Formatted ICD-9 code with decimal
        """
        code = str(code).strip()
        
        if code.startswith('E'):
            # E codes: Exxx.x format
            if len(code) > 4:
                return code[:4] + '.' + code[4:]
            return code
        elif code.startswith('V'):
            # V codes: Vxx.xx format
            if len(code) > 3:
                return code[:3] + '.' + code[3:]
            return code
        else:
            # Regular diagnosis codes: xxx.xx format
            if len(code) > 3:
                return code[:3] + '.' + code[3:]
            return code
    
    def analyze_patient(self, patient_data, log_path=None, show_predictions=True, 
                       show_next_visit_icd=True):
        """
        Perform comprehensive analysis of patient data.
        
        Args:
            patient_data: Dictionary containing patient EHR data
            log_path: Path to experiment results (e.g., condition_A_gpt_4o)
            show_predictions: Whether to show model predictions
            show_next_visit_icd: Whether to show next visit ICD codes
        """
        patient_id = patient_data['patient_id']
        ehr_data = patient_data['ehr_data']
        
        print("=" * 80)
        print(f"PATIENT ANALYSIS: {patient_id}")
        print("=" * 80)
        print()
        
        # 1. Ground Truth Label
        self._print_ground_truth(ehr_data)
        
        # 2. Visit Statistics
        visit_stats = self._analyze_visits(ehr_data)
        
        # 3. Display per-visit details
        self._display_visit_details(visit_stats)
        
        # 4. Overall Summary
        self._display_summary(patient_id, ehr_data, visit_stats)
        
        # 5. Model Predictions (if requested)
        if show_predictions and log_path:
            self._display_predictions(patient_id, log_path)
        
        # 6. Next Visit ICD Codes (if requested)
        if show_next_visit_icd:
            self._display_next_visit_icd(patient_data)
    
    def _print_ground_truth(self, ehr_data):
        """Print ground truth label."""
        ground_truth_label = ehr_data['label']
        outcome = 'MORTALITY' if ground_truth_label == 1 else 'SURVIVAL'
        
        print("GROUND TRUTH LABEL")
        print("-" * 80)
        print(f"Label: {ground_truth_label} ({outcome})")
        print()
    
    def _analyze_visits(self, ehr_data):
        """Analyze visits and count concepts."""
        visit_stats = []
        
        for key in ehr_data.keys():
            if key.startswith('visit '):
                visit_num = key.split(' ')[1]
                visit_data = ehr_data[key]
                
                num_conditions = len(visit_data.get('conditions', []))
                num_procedures = len(visit_data.get('procedures', []))
                num_drugs = len(visit_data.get('drugs', []))
                
                visit_stats.append({
                    'visit_num': int(visit_num),
                    'conditions': num_conditions,
                    'procedures': num_procedures,
                    'drugs': num_drugs
                })
        
        visit_stats.sort(key=lambda x: x['visit_num'])
        return visit_stats
    
    def _display_visit_details(self, visit_stats):
        """Display concept counts per visit."""
        print("CONCEPT COUNTS PER VISIT")
        print("-" * 80)
        
        for stat in visit_stats:
            total = stat['conditions'] + stat['procedures'] + stat['drugs']
            print(f"Visit {stat['visit_num']}:")
            print(f"  - Conditions:  {stat['conditions']:3d} concepts")
            print(f"  - Procedures:  {stat['procedures']:3d} concepts")
            print(f"  - Drugs:       {stat['drugs']:3d} concepts")
            print(f"  - Total:       {total:3d} concepts")
            print()
    
    def _display_summary(self, patient_id, ehr_data, visit_stats):
        """Display overall summary statistics."""
        ground_truth_label = ehr_data['label']
        outcome = 'MORTALITY' if ground_truth_label == 1 else 'SURVIVAL'
        
        total_visits = len(visit_stats)
        total_conditions = sum(s['conditions'] for s in visit_stats)
        total_procedures = sum(s['procedures'] for s in visit_stats)
        total_drugs = sum(s['drugs'] for s in visit_stats)
        total_concepts = total_conditions + total_procedures + total_drugs
        
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Patient ID:           {patient_id}")
        print(f"Ground Truth Label:   {ground_truth_label} ({outcome})")
        print(f"Total Visits:         {total_visits}")
        print(f"Total Conditions:     {total_conditions}")
        print(f"Total Procedures:     {total_procedures}")
        print(f"Total Drugs:          {total_drugs}")
        print(f"Total Concepts:       {total_concepts}")
        if total_visits > 0:
            print(f"Average per Visit:    {total_concepts / total_visits:.1f} concepts")
        print("=" * 80)
        print()
    
    def _display_predictions(self, patient_id, log_path):
        """Display model predictions from experiment results."""
        results_file = os.path.join(log_path, "results.json")
        
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            print()
            return
        
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            if patient_id not in results_data:
                print(f"Patient {patient_id} not found in results")
                print()
                return
            
            patient_result = results_data[patient_id]
            
            print("MODEL PREDICTION RESULTS")
            print("=" * 80)
            print(f"Experiment:          {os.path.basename(log_path)}")
            print(f"Patient ID:          {patient_id}")
            print(f"Predicted Prob:      {patient_result.get('pred_prob', 'N/A')}")
            print(f"Prediction:          {patient_result.get('pred', 'N/A')}")
            print(f"Ground Truth:        {patient_result.get('label', 'N/A')}")
            
            if 'pred' in patient_result and 'label' in patient_result:
                is_correct = patient_result['pred'] == patient_result['label']
                print(f"Correct Prediction:  {is_correct}")
            
            print("=" * 80)
            print()
        except Exception as e:
            print(f"Error loading predictions: {e}")
            print()
    
    def _display_next_visit_icd(self, patient_data):
        """Display ICD-9 codes for next visit (prediction target)."""
        patient_id = patient_data['patient_id']
        
        try:
            # Parse patient ID
            parts = patient_id.split('_')
            subject_id = int(parts[0])
            current_visit_num = int(parts[1])
            next_visit_num = current_visit_num + 1
            
            print("NEXT VISIT ICD-9 DIAGNOSIS CODES (PREDICTION TARGET)")
            print("=" * 80)
            print(f"Current visit (input): Visit {current_visit_num}")
            print(f"Prediction target:     Visit {next_visit_num}")
            print()
            
            if not os.path.exists(self.diagnoses_file) or not os.path.exists(self.admissions_file):
                print("Required MIMIC-III files not found")
                print()
                return
            
            # Load MIMIC-III data
            print(f"Loading MIMIC-III data for Subject ID: {subject_id}...")
            diagnoses_df = pd.read_csv(self.diagnoses_file)
            admissions_df = pd.read_csv(self.admissions_file)
            
            # Get patient admissions sorted by time
            patient_admissions = admissions_df[admissions_df['SUBJECT_ID'] == subject_id].copy()
            patient_admissions['ADMITTIME'] = pd.to_datetime(patient_admissions['ADMITTIME'])
            patient_admissions = patient_admissions.sort_values('ADMITTIME')
            
            if len(patient_admissions) <= current_visit_num:
                print(f"No visit {next_visit_num} found for patient {patient_id}")
                print(f"Patient has only {len(patient_admissions)} visits in total")
                print()
                return
            
            # Get next visit admission
            next_visit_hadm_id = patient_admissions.iloc[current_visit_num]['HADM_ID']
            next_visit_dx = diagnoses_df[diagnoses_df['HADM_ID'] == next_visit_hadm_id].copy()
            next_visit_dx = next_visit_dx.sort_values('SEQ_NUM')
            
            print("-" * 80)
            print(f"Hospital Admission ID: {int(next_visit_hadm_id)}")
            print(f"Admission Date:        {patient_admissions.iloc[current_visit_num]['ADMITTIME']}")
            print(f"Number of diagnoses:   {len(next_visit_dx)}")
            print("-" * 80)
            print()
            
            # Display ICD codes with CCS mappings
            print(f"{'Seq':<5} {'ICD-9 Raw':<10} {'ICD-9 Fmt':<11} {'CCS':<6} {'CCS Category Description'}")
            print("-" * 80)
            
            for idx, row in next_visit_dx.iterrows():
                seq_num = row['SEQ_NUM']
                icd_code_raw = str(row['ICD9_CODE'])
                icd_code_formatted = self.format_icd9_code(icd_code_raw)
                
                # Get CCS mapping
                ccs_code = self.icd9_to_ccs_dict.get(icd_code_formatted, 'N/A')
                ccs_name = self.ccs_code_to_name.get(ccs_code, 'Unknown') if ccs_code != 'N/A' else 'Not mapped'
                
                seq_str = f"{int(seq_num):<5}" if pd.notna(seq_num) else f"{'--':<5}"
                print(f"{seq_str} {icd_code_raw:<10} {icd_code_formatted:<11} {ccs_code:<6} {ccs_name}")
            
            print("-" * 80)
            print()
            
            # Summary of unique CCS categories
            ccs_categories = []
            for idx, row in next_visit_dx.iterrows():
                icd_code_raw = str(row['ICD9_CODE'])
                icd_code_formatted = self.format_icd9_code(icd_code_raw)
                ccs_code = self.icd9_to_ccs_dict.get(icd_code_formatted, None)
                if ccs_code and ccs_code in self.ccs_code_to_name:
                    ccs_categories.append(self.ccs_code_to_name[ccs_code].lower())
            
            unique_categories = sorted(set(ccs_categories))
            print(f"Unique CCS Categories in Visit {next_visit_num}: {len(unique_categories)}")
            if unique_categories:
                print()
                for cat in unique_categories:
                    print(f"  • {cat}")
            print()
            
        except Exception as e:
            import traceback
            print(f"Error analyzing next visit: {e}")
            traceback.print_exc()
            print()


def analyze_patient_from_file(file_path, log_path=None, show_predictions=True, 
                              show_next_visit_icd=True):
    """
    Convenience function to analyze a patient from a JSON file.
    
    Args:
        file_path: Path to patient data JSON file
        log_path: Path to experiment results directory
        show_predictions: Whether to show model predictions
        show_next_visit_icd: Whether to show next visit ICD codes
    """
    with open(file_path, 'r') as f:
        patient_data = json.load(f)
    
    analyzer = PatientAnalyzer()
    analyzer.analyze_patient(patient_data, log_path, show_predictions, show_next_visit_icd)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python patient_analyzer.py <patient_json_file> [log_path]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    log_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_patient_from_file(file_path, log_path)
