"""
Generate GRPO training dataset for mortality integrator format enforcement.

This script creates a parquet file with prompts in Search-R1 format for training
the integrator model to output mortality probabilities in the correct format:
"MORTALITY PROBABILITY: X.XX"

The dataset follows the structure from Search-R1's GRPO training approach.
"""

import os
import json
import random
import re
import pandas as pd
from typing import List, Dict
from parse_debate_logs import DebateLogParser
from parse_retrieval_logs import RetrievalLogParser


class TrainingDataGenerator:
    """Generate GRPO training dataset from debate logs."""
    
    def __init__(
        self,
        ehr_file: str,
        primary_debate_dir: str,
        fallback_debate_dir: str,
        output_dir: str,
        hard_mode: bool = False,
        class_type: str = "mortality"
    ):
        """
        Initialize generator.
        
        Args:
            ehr_file: Path to EHR data JSON file
            primary_debate_dir: Primary debate logs directory
            fallback_debate_dir: Fallback debate logs directory
            output_dir: Output directory for parquet files
            hard_mode: If True, only use samples with format extraction failures
            class_type: Either 'mortality' or 'survival' - determines prompt format
        """
        self.ehr_file = ehr_file
        self.output_dir = output_dir
        self.hard_mode = hard_mode
        self.class_type = class_type.lower()
        
        # Validate class_type
        if self.class_type not in ["mortality", "survival"]:
            raise ValueError(f"class_type must be 'mortality' or 'survival', got: {class_type}")
        
        # Initialize parsers
        self.debate_parser = DebateLogParser(primary_debate_dir, fallback_debate_dir)
        self.retrieval_parser = RetrievalLogParser(
            primary_retrieval_dir=os.path.join(primary_debate_dir, "debate_logs"),
            fallback_retrieval_dir=os.path.join(fallback_debate_dir, "debate_logs")
        )
        
        # Load EHR data
        with open(ehr_file, 'r') as f:
            self.ehr_data = json.load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # If hard mode, find samples with failed extraction
        if hard_mode:
            self.hard_samples = self._find_hard_samples()
            print(f"Hard mode enabled: Found {len(self.hard_samples)} samples with format issues")
    
    def _find_hard_samples(self) -> set:
        """
        Find samples where mortality probability extraction failed.
        
        Returns:
            Set of (patient_id, visit_id) tuples
        """
        hard_samples = set()
        
        # Scan both primary and fallback debate logs
        for log_dir in [self.debate_parser.primary_dir, 
                        self.debate_parser.fallback_dir]:
            if not os.path.exists(log_dir):
                continue
            
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            
            for log_file in log_files:
                # Extract patient_id and visit_id from filename
                match = re.search(r'debate_responses_(\d+)_(\d+)\.log', log_file)
                if not match:
                    continue
                
                patient_id, visit_id = match.groups()
                log_path = os.path.join(log_dir, log_file)
                
                # Check if extraction failed
                try:
                    with open(log_path, 'r') as f:
                        if "EXTRACTED MORTALITY PROBABILITY: None" in f.read():
                            hard_samples.add((patient_id, visit_id))
                except Exception:
                    continue
        
        return hard_samples
    
    def format_patient_context(self, patient_id: str, visit_id: str) -> str:
        """
        Format patient EHR context.
        
        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            
        Returns:
            Formatted patient context string
        """
        # Find the patient in EHR data
        patient_key = f"{patient_id}_{visit_id}"
        
        for sample in self.ehr_data:
            if sample.get("patient_id") == int(patient_id) and sample.get("visit_id") == int(visit_id):
                # Format the EHR data
                context_parts = []
                context_parts.append(f"Patient ID: {patient_id}, Visit ID: {visit_id}")                
                # Add conditions
                if "conditions" in sample:
                    context_parts.append("\nConditions:")
                    for cond in sample["conditions"]:
                        context_parts.append(f"  - {cond}")
                
                # Add procedures
                if "procedures" in sample:
                    context_parts.append("\nProcedures:")
                    for proc in sample["procedures"]:
                        context_parts.append(f"  - {proc}")
                
                # Add medications
                if "drugs" in sample:
                    context_parts.append("\nMedications:")
                    for drug in sample["drugs"]:
                        context_parts.append(f"  - {drug}")
                
                return "\n".join(context_parts)
        
        return f"Patient ID: {patient_id}, Visit ID: {visit_id} (EHR data not found)"
    
    def create_integrator_prompt(
        self,
        patient_id: str,
        visit_id: str,
        debate_responses,
        retrieved_docs
    ) -> str:
        """
        Create the complete integrator prompt matching the original system.
        
        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            debate_responses: DebateResponses object
            retrieved_docs: List of RetrievedDocument objects
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # System prompt - different for mortality vs survival
        if self.class_type == "mortality":
            system_prompt = """You are a medical AI Clinical Assistant analyzing MORTALITY risk for hospital patients. Your task is to provide a probability assessment for whether the patient will die during their NEXT hospital visit.

CRITICAL OUTPUT REQUIREMENT:
You MUST end your response with exactly this format:
MORTALITY PROBABILITY: X.XX
where X.XX is a number between 0.00 and 1.00

Guidelines:
- 0.00 = Definitely will survive
- 0.50 = Uncertain/Equal risk
- 1.00 = Definitely will die

Provide comprehensive clinical reasoning analyzing ALL available evidence, then conclude with the required probability format."""
        else:  # survival
            system_prompt = """You are a medical AI Clinical Assistant analyzing SURVIVAL probability for hospital patients. Your task is to provide a probability assessment for whether the patient will survive their NEXT hospital visit.

CRITICAL OUTPUT REQUIREMENT:
You MUST end your response with exactly this format:
SURVIVAL PROBABILITY: X.XX
where X.XX is a number between 0.00 and 1.00

Guidelines:
- 0.00 = Definitely will die
- 0.50 = Uncertain/Equal risk
- 1.00 = Definitely will survive

Provide comprehensive clinical reasoning analyzing ALL available evidence, then conclude with the required probability format."""
        
        prompt_parts.append("# System Instructions")
        prompt_parts.append(system_prompt)
        prompt_parts.append("")
        
        # Patient context
        patient_context = self.format_patient_context(patient_id, visit_id)
        prompt_parts.append("# Patient Context")
        prompt_parts.append(patient_context)
        prompt_parts.append("")
        
        # Debate history
        prompt_parts.append("# Debate Agent Analyses")
        prompt_parts.append("")
        prompt_parts.append("## Target Patient Analyst Assessment")
        prompt_parts.append(debate_responses.target_patient_analyst)
        prompt_parts.append("")
        prompt_parts.append("## Mortality Risk Assessor Analysis")
        prompt_parts.append(debate_responses.mortality_risk_assessor)
        prompt_parts.append("")
        prompt_parts.append("## Protective Factor Analyst Analysis")
        prompt_parts.append(debate_responses.protective_factor_analyst)
        prompt_parts.append("")
        
        # Retrieved documents
        prompt_parts.append("# Retrieved Medical Evidence")
        formatted_docs = self.retrieval_parser.format_retrieved_documents(retrieved_docs, max_docs=5)
        prompt_parts.append(formatted_docs)
        prompt_parts.append("")
        
        # Final instruction
        prompt_parts.append("# Task")
        if self.class_type == "mortality":
            prompt_parts.append("Based on all the evidence above, provide your comprehensive mortality risk assessment.")
            prompt_parts.append("You MUST conclude with: MORTALITY PROBABILITY: X.XX (where X.XX is between 0.00 and 1.00)")
        else:  # survival
            prompt_parts.append("Based on all the evidence above, provide your comprehensive survival probability assessment.")
            prompt_parts.append("You MUST conclude with: SURVIVAL PROBABILITY: X.XX (where X.XX is between 0.00 and 1.00)")
        
        return "\n".join(prompt_parts)
    
    def generate_single_sample(
        self,
        patient_id: str,
        visit_id: str
    ) -> Dict:
        """
        Generate a single training sample.
        
        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            
        Returns:
            Dictionary with prompt field for parquet
        """
        # Get debate responses
        debate_responses = self.debate_parser.get_debate_responses(patient_id, visit_id)
        if not debate_responses:
            return None
        
        # Get retrieved documents
        retrieved_docs = self.retrieval_parser.get_retrieved_documents(patient_id, visit_id)
        if not retrieved_docs:
            print(f"Warning: No retrieved docs for {patient_id}_{visit_id}, using empty list")
            retrieved_docs = []
        
        # Create prompt
        prompt_text = self.create_integrator_prompt(
            patient_id, visit_id, debate_responses, retrieved_docs
        )
        
        # Format as chat (Search-R1 style)
        prompt = [
            {
                "role": "user",
                "content": prompt_text
            }
        ]
        
        # Return with VERL-required fields
        return {
            "prompt": prompt,
            "patient_id": patient_id,
            "visit_id": visit_id,
            "source_dir": debate_responses.source_dir,
            "data_source": debate_responses.source_dir,  # VERL requires data_source
            "reward_model": {"ground_truth": "__FORMAT_ONLY__"}  # Format-only training placeholder
        }
    
    def generate_dataset(
        self,
        num_samples: int = 500,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        """
        Generate full training dataset.
        
        Args:
            num_samples: Number of samples to generate
            train_ratio: Ratio for train/val split
            random_seed: Random seed for reproducibility
        """
        random.seed(random_seed)
        
        # Get all available samples
        all_samples = self.debate_parser.get_available_samples()
        print(f"Total available samples: {len(all_samples)}")
        
        # Filter for hard samples if in hard mode
        if self.hard_mode:
            all_samples = [s for s in all_samples if s in self.hard_samples]
            print(f"Hard mode: Filtered to {len(all_samples)} hard samples")
        
        # Randomly sample
        if num_samples > len(all_samples):
            print(f"Warning: Requested {num_samples} but only {len(all_samples)} available")
            num_samples = len(all_samples)
        
        selected_samples = random.sample(all_samples, num_samples)
        print(f"Selected {len(selected_samples)} samples")
        
        # Generate samples
        dataset = []
        failed = []
        
        for i, (patient_id, visit_id) in enumerate(selected_samples):
            if (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(selected_samples)}...")
            
            sample = self.generate_single_sample(patient_id, visit_id)
            if sample:
                dataset.append(sample)
            else:
                failed.append((patient_id, visit_id))
        
        print(f"\nSuccessfully generated {len(dataset)} samples")
        print(f"Failed: {len(failed)} samples")
        
        if failed:
            print(f"Failed samples: {failed[:10]}...")  # Show first 10
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Split into train/test (80/20) - GRPO doesn't need validation set
        train_size = int(train_ratio * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Verify VERL-required fields are present
        required_fields = ['prompt', 'data_source', 'reward_model']
        for field in required_fields:
            if field not in train_df.columns:
                raise ValueError(f"Missing required VERL field: {field}")
        
        print(f"\n✓ VERL-required fields verified:")
        print(f"  - data_source: {train_df.iloc[0]['data_source']}")
        print(f"  - reward_model.ground_truth: {train_df.iloc[0]['reward_model']['ground_truth']}")
        
        # Save to parquet
        train_path = os.path.join(self.output_dir, "train.parquet")
        test_path = os.path.join(self.output_dir, "test.parquet")
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        print(f"\nSaved training data:")
        print(f"  Train: {train_path} ({len(train_df)} samples)")
        print(f"  Test: {test_path} ({len(test_df)} samples)")
        print(f"\n✓ Data ready for VERL GRPO training!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GRPO training data")
    parser.add_argument("--hard", action="store_true", 
                        help="Hard mode: only include samples where format extraction failed")
    parser.add_argument("--class_type", type=str, default="mortality", choices=["mortality", "survival"],
                        help="Type of probability to predict: 'mortality' or 'survival' (default: mortality)")
    args = parser.parse_args()
    
    # Paths
    ehr_file = "/data/wang/junh/githubs/Debate/KARE/data/ehr_data/mimic3_mortality_samples_test.json"
    primary_debate_dir = "/data/wang/junh/githubs/Debate/KARE/results/arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8"
    fallback_debate_dir = "/data/wang/junh/githubs/Debate/KARE/results/fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8"
    
    # Set output directory based on mode and class_type
    base_dir = f"/data/wang/junh/githubs/Debate/KARE/verl/data_generation/{args.class_type}_grpo_data"
    if args.hard:
        output_dir = f"{base_dir}_hard"
        print(f"🔥 HARD MODE: Only samples with format extraction failures ({args.class_type.upper()})")
    else:
        output_dir = base_dir
        print(f"📋 STANDARD MODE: All samples ({args.class_type.upper()})")
    
    # Create generator
    generator = TrainingDataGenerator(
        ehr_file=ehr_file,
        primary_debate_dir=primary_debate_dir,
        fallback_debate_dir=fallback_debate_dir,
        output_dir=output_dir,
        hard_mode=args.hard,
        class_type=args.class_type
    )
    
    # Test on single sample first
    test_patient = "14038" if args.hard else "34"
    test_visit = "0"
    print(f"\nTesting on sample (patient {test_patient}_{test_visit})...")
    sample = generator.generate_single_sample(test_patient, test_visit)
    
    if sample:
        print("\n✓ Successfully generated test sample!")
        print(f"  Source directory: {sample['source_dir']}")
        print(f"  Prompt length: {len(sample['prompt'][0]['content'])} characters")
        print(f"\n  First 500 characters of prompt:")
        print(sample['prompt'][0]['content'][:500])
        print("...")
        
        # Generate full dataset
        print("\n" + "="*80)
        if args.hard:
            print(f"Generating HARD {args.class_type.upper()} dataset (format extraction failures)...")
        else:
            print(f"Generating STANDARD {args.class_type.upper()} dataset (all available samples)...")
        print("="*80 + "\n")
        generator.generate_dataset(num_samples=999, train_ratio=0.8)
    else:
        print("Failed to generate test sample!")


if __name__ == "__main__":
    main()
