#!/usr/bin/env python3
"""
Retrieve patient EHR data and similar patient information.

Usage:
    python get_patient_data.py <patient_id>
    python get_patient_data.py 10774_5
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class PatientDataRetriever:
    """Class to retrieve patient EHR data and similar patient information."""
    
    def __init__(self, dataset="mimic3", task="mortality"):
        """
        Initialize the retriever.
        
        Args:
            dataset: Dataset name (mimic3 or mimic4)
            task: Task name (mortality or readmission)
        """
        self.dataset = dataset
        self.task = task
        
        # Define data paths
        self.base_path = Path("/data/wang/junh/datasets/KARE")
        self.ehr_data_path = self.base_path / "ehr_data" / f"pateint_{dataset}_{task}.json"
        self.similar_patients_path = (
            self.base_path / "patient_context" / "similar_patient_debate" / 
            f"patient_to_top_1_patient_contexts_{dataset}_{task}_improved.json"
        )
        
        # Cache for loaded data
        self._patient_data = None
        self._similar_patients = None
    
    def load_patient_data(self) -> Dict[str, Any]:
        """Load all patient EHR data."""
        if self._patient_data is None:
            print(f"Loading patient data from {self.ehr_data_path}...")
            with open(self.ehr_data_path, 'r') as f:
                self._patient_data = json.load(f)
            print(f"Loaded {len(self._patient_data)} patients")
        return self._patient_data
    
    def load_similar_patients(self) -> Dict[str, Any]:
        """Load similar patient mappings."""
        if self._similar_patients is None:
            if self.similar_patients_path.exists():
                print(f"Loading similar patients from {self.similar_patients_path}...")
                
                # For large files, we'll read them line by line
                try:
                    with open(self.similar_patients_path, 'r') as f:
                        self._similar_patients = json.load(f)
                    print(f"Loaded similar patients for {len(self._similar_patients)} patients")
                except Exception as e:
                    print(f"Warning: Could not load similar patients file: {e}")
                    print("Attempting to parse line by line...")
                    self._similar_patients = {}
            else:
                print(f"Similar patients file not found: {self.similar_patients_path}")
                self._similar_patients = {}
        return self._similar_patients
    
    def get_patient_ehr(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get EHR data for a specific patient.
        
        Args:
            patient_id: Patient ID (e.g., "10774_5")
            
        Returns:
            Dictionary containing patient EHR data or None if not found
        """
        patient_data = self.load_patient_data()
        return patient_data.get(patient_id)
    
    def get_similar_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get similar patient information for a specific patient.
        
        Args:
            patient_id: Patient ID (e.g., "10774_5")
            
        Returns:
            Dictionary containing similar patient data or None if not found
        """
        similar_patients = self.load_similar_patients()
        return similar_patients.get(patient_id)
    
    def extract_similar_patient_id(self, patient_id: str) -> Optional[str]:
        """
        Extract the similar patient ID from the similar patient info.
        
        Args:
            patient_id: Target patient ID
            
        Returns:
            Similar patient ID or None
        """
        similar_info = self.get_similar_patient_info(patient_id)
        if not similar_info:
            return None
        
        # The similar patient context often contains patient ID in the text
        # We'll need to parse it from the context
        context = similar_info.get("similar_patient_context", "")
        
        # Try to find patient ID pattern in the context
        # This is a simple heuristic - might need adjustment based on actual format
        import re
        match = re.search(r'patient[_\s]+(\d+_\d+)', context.lower())
        if match:
            return match.group(1)
        
        # Return the first key that's not metadata
        for key in similar_info.keys():
            if key != "similar_patient_context" and "_" in key:
                return key
        
        return None
    
    def display_patient_data(self, patient_id: str, verbose: bool = True):
        """
        Display comprehensive patient information.
        
        Args:
            patient_id: Patient ID to display
            verbose: If True, show detailed information
        """
        print("=" * 80)
        print(f"PATIENT DATA: {patient_id}")
        print("=" * 80)
        
        # Get patient EHR
        ehr_data = self.get_patient_ehr(patient_id)
        if not ehr_data:
            print(f"❌ Patient {patient_id} not found in EHR data")
            return
        
        # Display label
        label = ehr_data.get("label")
        label_str = "MORTALITY" if label == 1 else "SURVIVAL"
        print(f"\n📋 LABEL: {label} ({label_str})")
        
        # Display visits
        visits = [key for key in ehr_data.keys() if key.startswith("visit")]
        print(f"\n🏥 NUMBER OF VISITS: {len(visits)}")
        
        for visit_key in sorted(visits):
            visit_data = ehr_data[visit_key]
            print(f"\n  {visit_key.upper()}:")
            
            if verbose:
                # Conditions
                conditions = visit_data.get("conditions", [])
                print(f"    Conditions ({len(conditions)}):")
                for i, cond in enumerate(conditions[:5], 1):
                    print(f"      {i}. {cond}")
                if len(conditions) > 5:
                    print(f"      ... and {len(conditions) - 5} more")
                
                # Procedures
                procedures = visit_data.get("procedures", [])
                print(f"    Procedures ({len(procedures)}):")
                for i, proc in enumerate(procedures[:5], 1):
                    print(f"      {i}. {proc}")
                if len(procedures) > 5:
                    print(f"      ... and {len(procedures) - 5} more")
                
                # Drugs
                drugs = visit_data.get("drugs", [])
                print(f"    Medications ({len(drugs)}):")
                for i, drug in enumerate(drugs[:5], 1):
                    print(f"      {i}. {drug}")
                if len(drugs) > 5:
                    print(f"      ... and {len(drugs) - 5} more")
            else:
                conditions = visit_data.get("conditions", [])
                procedures = visit_data.get("procedures", [])
                drugs = visit_data.get("drugs", [])
                print(f"    {len(conditions)} conditions, {len(procedures)} procedures, {len(drugs)} medications")
        
        # Display similar patient info
        print("\n" + "=" * 80)
        print("SIMILAR PATIENT INFORMATION")
        print("=" * 80)
        
        similar_info = self.get_similar_patient_info(patient_id)
        if similar_info:
            print("✓ Similar patient data found")
            
            # Check for positive and negative similar patients
            for key in ["positive", "negative"]:
                if key in similar_info and similar_info[key]:
                    print(f"\n{key.upper()} SIMILAR PATIENT(S):")
                    similar_list = similar_info[key]
                    
                    # The data is a list of strings (patient context texts)
                    if isinstance(similar_list, list):
                        for idx, context in enumerate(similar_list, 1):
                            print(f"\n  Similar Patient #{idx}:")
                            
                            # Extract patient ID from the context text
                            import re
                            match = re.search(r'Patient ID:\s*(\d+_\d+)', context)
                            similar_id = match.group(1) if match else "Unknown"
                            print(f"    Patient ID: {similar_id}")
                            
                            # Show the context
                            print(f"\n    Context ({len(context)} characters):")
                            # Show first 800 chars
                            print("    " + "-" * 74)
                            preview = context[:800].replace("\n", "\n    ")
                            print(f"    {preview}")
                            if len(context) > 800:
                                print(f"    ... ({len(context) - 800} more characters)")
                            print("    " + "-" * 74)
                            
                            # If we have the similar patient ID, show their EHR summary
                            if similar_id != "Unknown" and self._patient_data and similar_id in self._patient_data:
                                similar_ehr = self._patient_data[similar_id]
                                similar_label = similar_ehr.get("label")
                                label_str = "MORTALITY" if similar_label == 1 else "SURVIVAL"
                                visits = [k for k in similar_ehr.keys() if k.startswith("visit")]
                                print(f"\n    Similar Patient EHR Data:")
                                print(f"      Label: {similar_label} ({label_str})")
                                print(f"      Number of visits: {len(visits)}")
        else:
            print("❌ No similar patient data found")
    
    def export_patient_json(self, patient_id: str, output_path: Optional[str] = None):
        """
        Export patient data to JSON file.
        
        Args:
            patient_id: Patient ID to export
            output_path: Output file path (default: patient_id.json)
        """
        if output_path is None:
            output_path = f"{patient_id}_data.json"
        
        data = {
            "patient_id": patient_id,
            "ehr_data": self.get_patient_ehr(patient_id),
            "similar_patient_info": self.get_similar_patient_info(patient_id)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported patient data to {output_path}")


def main():
    """Main function to run from command line."""
    if len(sys.argv) < 2:
        print("Usage: python get_patient_data.py <patient_id> [dataset] [task]")
        print("Example: python get_patient_data.py 10774_5")
        print("Example: python get_patient_data.py 10774_5 mimic3 mortality")
        sys.exit(1)
    
    patient_id = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else "mimic3"
    task = sys.argv[3] if len(sys.argv) > 3 else "mortality"
    
    # Create retriever
    retriever = PatientDataRetriever(dataset=dataset, task=task)
    
    # Display patient data
    retriever.display_patient_data(patient_id, verbose=True)
    
    # Ask if user wants to export
    print("\n" + "=" * 80)
    response = input(f"Export data to JSON file? (y/n): ").strip().lower()
    if response == 'y':
        retriever.export_patient_json(patient_id)


if __name__ == "__main__":
    main()
