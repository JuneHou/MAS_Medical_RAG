#!/usr/bin/env python3
"""
Generate GRPO training data for mortality prediction task.
Runs full 3-round debate pipeline on train/val splits and captures integrator prompts.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import argparse

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from kare_data_adapter import KAREDataAdapter
from mortality_debate_rag_grpo import MortalityDebateSystem


class PredictionDataGenerator:
    """
    Generate GRPO training data by running 3-round debate and capturing integrator prompts.
    """
    
    def __init__(self, 
                 data_split: str = 'train',
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 gpu_ids: str = "6,7",
                 rag_enabled: bool = True,
                 corpus_name: str = "MedCorp2",
                 retriever_name: str = "MedCPT",
                 db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"):
        """
        Initialize data generator.
        
        Args:
            data_split: Data split to use ('train' or 'val')
            model_name: Model name for agents 1-3
            gpu_ids: GPU IDs (comma-separated)
            rag_enabled: Enable RAG retrieval
            corpus_name: MedRAG corpus name
            retriever_name: MedRAG retriever name
            db_dir: MedRAG database directory
        """
        self.data_split = data_split
        
        # Initialize data adapter for the specified split
        print(f"Loading {data_split} split data...")
        self.data_adapter = KAREDataAdapter(
            base_path="/data/wang/junh/datasets/KARE",
            split=data_split
        )
        
        # Initialize debate system (only for agents 1-3)
        print("Initializing debate system...")
        self.debate_system = MortalityDebateSystem(
            model_name=model_name,
            gpu_ids=gpu_ids,
            rag_enabled=rag_enabled,
            corpus_name=corpus_name,
            retriever_name=retriever_name,
            db_dir=db_dir
        )
    
    def generate_training_examples(self, 
                                   num_samples: int = None, 
                                   output_dir: str = "./data/grpo_training",
                                   start_idx: int = 0,
                                   balanced_file: str = None,
                                   sample_indices: List[int] = None) -> str:
        """
        Generate training examples by running 3-round debate.
        
        Args:
            num_samples: Number of samples to process (None = all)
            output_dir: Output directory for parquet files
            start_idx: Starting index for processing
            balanced_file: Path to balanced sample JSON file (overrides num_samples/start_idx)
            sample_indices: List of specific indices to process (overrides num_samples/start_idx)
            
        Returns:
            Path to generated parquet file
        """
        # Load balanced sample file if provided
        if balanced_file:
            print(f"Loading balanced sample file: {balanced_file}")
            with open(balanced_file, 'r') as f:
                balanced_data = json.load(f)
            
            sample_indices = balanced_data['metadata']['all_indices']
            print(f"Loaded {len(sample_indices)} balanced samples:")
            print(f"  Positive (mortality=1): {balanced_data['metadata']['n_positive']}")
            print(f"  Negative (survival=0): {balanced_data['metadata']['n_negative']}")
        
        # Determine sample range
        if sample_indices is not None:
            # Use specific indices
            indices_to_process = sample_indices
            print(f"Processing {len(indices_to_process)} specific indices")
        else:
            # Use range
            total_samples = len(self.data_adapter.data)
            if num_samples is None:
                num_samples = total_samples - start_idx
            else:
                num_samples = min(num_samples, total_samples - start_idx)
            
            end_idx = start_idx + num_samples
            indices_to_process = list(range(start_idx, end_idx))
            print(f"Processing samples {start_idx} to {end_idx-1} ({num_samples} total)")
        
        training_examples = []
        error_count = 0
        
        # Process each patient
        for i in tqdm(indices_to_process, desc=f"Generating {self.data_split} data"):
            try:
                # Get formatted patient sample
                sample = self.data_adapter.get_test_sample(i)
                
                # Run 3-round debate using the same pipeline as downstream deployment
                debate_result = self._run_debate_rounds(sample, log_dir=output_dir)
                
                if debate_result is None:
                    error_count += 1
                    continue
                
                # Generate TWO training examples per patient:
                # 1. Mortality probability assessment prompt
                # 2. Survival probability assessment prompt
                
                # Example 1: Mortality assessment
                mortality_prompt = self._construct_integrator_prompt(
                    sample, 
                    debate_result, 
                    assessment_type='mortality'
                )
                
                training_examples.append({
                    'prompt': mortality_prompt,
                    'data_source': 'kare_mortality_prediction',
                    'ground_truth': sample['ground_truth'],
                    'extra_info': {
                        'patient_id': sample['patient_id'],
                        'visit_id': sample['visit_id'],
                        'split': self.data_split,
                        'assessment_type': 'mortality'
                    }
                })
                
                # Example 2: Survival assessment
                survival_prompt = self._construct_integrator_prompt(
                    sample, 
                    debate_result, 
                    assessment_type='survival'
                )
                
                training_examples.append({
                    'prompt': survival_prompt,
                    'data_source': 'kare_survival_prediction',
                    'ground_truth': sample['ground_truth'],
                    'extra_info': {
                        'patient_id': sample['patient_id'],
                        'visit_id': sample['visit_id'],
                        'split': self.data_split,
                        'assessment_type': 'survival'
                    }
                })
                
            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
        
        print(f"\nGenerated {len(training_examples)} training examples ({error_count} errors)")
        
        # Save to Parquet
        output_path = self._save_parquet(training_examples, output_dir)
        
        return output_path
    
    def _run_debate_rounds(self, sample: Dict[str, Any], log_dir: str = None) -> Dict[str, Any]:
        """
        Run 3-round debate (agents 1-3 only) and perform retrieval WITHOUT running integrator.
        This ensures we don't leak integrator responses into training data.
        
        Args:
            sample: Formatted patient sample from data adapter
            log_dir: Directory for saving debate logs
            
        Returns:
            Dictionary with debate_history (rounds 1-3), patient context, and retrieved docs
        """
        try:
            # Run ONLY the first 3 rounds (agents 1-3)
            # We'll manually replicate the retrieval that the integrator would do
            
            import logging
            from pathlib import Path
            
            # Setup logging
            if log_dir:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                log_filename = log_path / f"debate_responses_{sample['patient_id']}.log"
                logger = logging.getLogger(f"debate_{sample['patient_id']}")
                logger.setLevel(logging.INFO)
                logger.handlers.clear()
                file_handler = logging.FileHandler(log_filename, mode='w')
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info(f"Starting debate for patient {sample['patient_id']}")
            else:
                logger = None
            
            debate_history = []
            similar_patients_dict = {
                'positive': sample['positive_similars'],
                'negative': sample['negative_similars']
            }
            
            # Round 1: Target Patient Analysis
            print(f"\n--- ROUND 1: TARGET PATIENT ANALYSIS ---")
            target_response = self.debate_system._agent_turn(
                role="target_patient_analyst",
                patient_context=sample['target_context'],
                similar_patients=similar_patients_dict,
                medical_knowledge="",
                debate_history=[],
                logger=logger,
                patient_id=sample['patient_id'],
                log_dir=str(log_dir) if log_dir else None
            )
            debate_history.append(target_response)
            
            # Round 2: Similar Patient Comparisons (BATCH)
            print(f"\n--- ROUND 2: SIMILAR PATIENT COMPARISONS (BATCH) ---")
            round2_responses = self.debate_system._agent_turn_batch(
                roles=["mortality_risk_assessor", "protective_factor_analyst"],
                patient_context=sample['target_context'],
                similar_patients=similar_patients_dict,
                medical_knowledge="",
                debate_history=debate_history,
                logger=logger,
                patient_id=sample['patient_id'],
                log_dir=str(log_dir) if log_dir else None
            )
            debate_history.extend(round2_responses)
            
            # Round 3: Do ONLY the retrieval that integrator would do (no LLM call)
            print(f"\n--- ROUND 3: RETRIEVAL ONLY (NO INTEGRATOR RESPONSE) ---")
            
            # Prepare retrieval queries based on debate history
            # Replicate the integrator's retrieval logic without running the LLM
            mortality_query = self._generate_mortality_query(debate_history, sample['target_context'])
            survival_query = self._generate_survival_query(debate_history, sample['target_context'])
            
            # Perform retrieval
            mortality_docs = []
            survival_docs = []
            
            if self.debate_system.rag_enabled:
                print(f"Retrieving mortality evidence: '{mortality_query}'")
                mortality_docs = self.debate_system.retriever.retrieve(
                    question=mortality_query,
                    k=self.debate_system.k,
                    rrf_k=self.debate_system.rrf_k
                )
                
                print(f"Retrieving survival evidence: '{survival_query}'")
                survival_docs = self.debate_system.retriever.retrieve(
                    question=survival_query,
                    k=self.debate_system.k,
                    rrf_k=self.debate_system.rrf_k
                )
            
            return {
                'debate_history': debate_history,  # Only rounds 1-3
                'patient_context': sample['target_context'],
                'similar_patients': similar_patients_dict,
                'mortality_retrieved_docs': mortality_docs,
                'survival_retrieved_docs': survival_docs,
                'mortality_query': mortality_query,
                'survival_query': survival_query
            }
            
        except Exception as e:
            print(f"\nError in debate rounds for patient {sample['patient_id']}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_retrieved_docs(self, docs_list: List[Dict[str, Any]]) -> str:
        """
        Format list of retrieved documents into a readable string.
        
        Args:
            docs_list: List of document dictionaries from MedRAG retrieval
            
        Returns:
            Formatted string with document contents
        """
        if not docs_list:
            return ""
        
        formatted_docs = []
        for i, doc in enumerate(docs_list, 1):
            title = doc.get('title', 'Unknown')
            content = doc.get('content', doc.get('text', ''))[:1000]  # Limit content length
            formatted_docs.append(f"[Document {i}] {title}\n{content}...")
        
        return "\n\n".join(formatted_docs)
    
    def _construct_integrator_prompt(self, 
                                    sample: Dict[str, Any], 
                                    debate_result: Dict[str, Any], 
                                    assessment_type: str = 'mortality') -> str:
        """
        Construct the exact integrator prompt that will be used during GRPO training.
        This replicates the prompt construction in mortality_debate_rag.py's _execute_integrator_attempt.
        
        Args:
            sample: Patient sample data
            debate_result: Results from 3-round debate
            assessment_type: 'mortality' or 'survival'
            
        Returns:
            Complete integrator prompt string
        """
        # Use the debate system's method to prepare debate history
        debate_history_text = self.debate_system._prepare_integrator_history(
            debate_result['debate_history']
        )
        
        # Get the appropriate system prompt based on assessment type
        if assessment_type == 'mortality':
            system_prompt = self.debate_system.agent_prompts['balanced_clinical_integrator_mortality']
        else:
            system_prompt = self.debate_system.agent_prompts['balanced_clinical_integrator_survival']
        
        # Construct patient information section (only target patient - similar patients already in debate history)
        patient_info = f"""## Target Patient Information:
{debate_result['patient_context']}"""
        
        # Extract retrieved documents from debate_result (NOT from integrator response)
        # These come from the retrieval we did manually, not from integrator
        retrieval_context = ""
        if assessment_type == 'mortality':
            mortality_docs_list = debate_result.get('mortality_retrieved_docs', [])
            if mortality_docs_list and isinstance(mortality_docs_list, list):
                mortality_docs_text = self._format_retrieved_docs(mortality_docs_list)
                if mortality_docs_text:
                    retrieval_context = f"\n\n## Retrieved Evidence for Mortality Assessment ##\n{mortality_docs_text}"
        else:  # survival
            survival_docs_list = debate_result.get('survival_retrieved_docs', [])
            if survival_docs_list and isinstance(survival_docs_list, list):
                survival_docs_text = self._format_retrieved_docs(survival_docs_list)
                if survival_docs_text:
                    retrieval_context = f"\n\n## Retrieved Evidence for Survival Assessment ##\n{survival_docs_text}"
        
        # Combine all parts into the full prompt
        full_prompt = f"""{system_prompt}

{patient_info}

{debate_history_text}{retrieval_context}

Now provide your {assessment_type} probability assessment with comprehensive reasoning."""
        
        return full_prompt
    
    def _generate_mortality_query(self, debate_history: List[Dict], patient_context: str) -> str:
        """
        Generate mortality-focused retrieval query based on debate history.
        Replicates integrator's query generation logic.
        """
        # Extract key risk factors from debate history
        risk_factors = []
        for round_data in debate_history:
            message = round_data.get('message', '')
            # Simple extraction of risk-related content
            if 'risk' in message.lower() or 'mortality' in message.lower():
                # Extract first sentence or first 100 chars as risk factor
                risk_snippet = message[:200].split('.')[0]
                risk_factors.append(risk_snippet)
        
        # Combine into query
        if risk_factors:
            query = f"mortality risk {' '.join(risk_factors[:3])}"
        else:
            query = "mortality risk factors complications"
        
        return query[:200]  # Limit query length
    
    def _generate_survival_query(self, debate_history: List[Dict], patient_context: str) -> str:
        """
        Generate survival-focused retrieval query based on debate history.
        Replicates integrator's query generation logic.
        """
        # Extract key protective factors from debate history
        protective_factors = []
        for round_data in debate_history:
            message = round_data.get('message', '')
            # Simple extraction of protective/survival content
            if 'protective' in message.lower() or 'survival' in message.lower():
                # Extract first sentence or first 100 chars
                protective_snippet = message[:200].split('.')[0]
                protective_factors.append(protective_snippet)
        
        # Combine into query
        if protective_factors:
            query = f"survival outcomes {' '.join(protective_factors[:3])}"
        else:
            query = "survival prognosis recovery outcomes"
        
        return query[:200]  # Limit query length
    
    def _save_parquet(self, training_examples: List[Dict[str, Any]], output_dir: str) -> str:
        """
        Save training examples to Parquet format.
        
        Args:
            training_examples: List of training examples
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_examples)
        
        # Convert extra_info dict to JSON string for Parquet compatibility
        df['extra_info'] = df['extra_info'].apply(json.dumps)
        
        # Save to Parquet
        output_file = output_path / f"{self.data_split}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"Saved {len(training_examples)} examples to {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training data for mortality prediction")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Data split to process (default: train for GRPO training)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (None = all)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for processing')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/wang/junh/githubs/Debate/KARE/verl/data_generation/prediction',
                       help='Output directory for parquet files')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Model name for agents 1-3')
    parser.add_argument('--gpus', type=str, default='3,4',
                       help='GPU IDs (comma-separated)')
    parser.add_argument('--no_rag', action='store_true',
                       help='Disable RAG retrieval')
    parser.add_argument('--corpus_name', type=str, default='MedCorp2',
                       help='MedRAG corpus name')
    parser.add_argument('--retriever_name', type=str, default='MedCPT',
                       help='MedRAG retriever name')
    parser.add_argument('--db_dir', type=str, 
                       default='/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus',
                       help='MedRAG database directory')
    parser.add_argument('--balanced_file', type=str, default=None,
                       help='Path to balanced sample JSON file (overrides num_samples/start_idx)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PredictionDataGenerator(
        data_split=args.split,
        model_name=args.model,
        gpu_ids=args.gpus,
        rag_enabled=not args.no_rag,
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        db_dir=args.db_dir
    )
    
    # Generate training data
    output_file = generator.generate_training_examples(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        balanced_file=args.balanced_file
    )
    
    print(f"\n✅ Data generation complete!")
    print(f"Output file: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Verify the generated data: python -c \"import pandas as pd; df=pd.read_parquet('{output_file}'); print(df.head())\"")
    print(f"2. Generate validation data if needed: python {__file__} --split val")
    print(f"3. Implement reward function: kare_prediction_reward.py")
    print(f"4. Start GRPO training with the generated data")


if __name__ == "__main__":
    main()
