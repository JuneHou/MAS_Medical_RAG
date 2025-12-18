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
                    'ground_truth': sample['ground_truth'],  # Keep for reference
                    'extra_info': {
                        'patient_id': sample['patient_id'],
                        'visit_id': sample['visit_id'],
                        'split': self.data_split,
                        'assessment_type': 'mortality'  # Used by reward function to select mortality prob
                    },
                    'reward_model': {
                        'ground_truth': sample['ground_truth']  # VERL passes this to reward function
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
                    'ground_truth': sample['ground_truth'],  # Keep for reference
                    'extra_info': {
                        'patient_id': sample['patient_id'],
                        'visit_id': sample['visit_id'],
                        'split': self.data_split,
                        'assessment_type': 'survival'  # Used by reward function to select survival prob
                    },
                    'reward_model': {
                        'ground_truth': sample['ground_truth']  # VERL passes this to reward function
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
                # Create debate_logs subdirectory for all debate-related logs
                log_path = Path(log_dir) / "debate_logs"
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
                log_dir=str(log_path) if log_dir else None
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
                log_dir=str(log_path) if log_dir else None
            )
            debate_history.extend(round2_responses)
            
            # Round 3: Let integrator LLM generate queries via tool calling, then retrieve
            print(f"\n--- ROUND 3: INTEGRATOR TOOL CALLING FOR RETRIEVAL ---")
            
            # Prepare integrator history for query generation
            history_text = self.debate_system._prepare_integrator_history(debate_history)
            primary_context = f"## Target Patient EHR Context ##\n{sample['target_context']}"
            
            # Step 3a: Generate MORTALITY query via LLM tool calling
            print("Step 3a: Generating mortality query via integrator LLM...")
            mortality_prompt = f"""{self.debate_system.agent_prompts["balanced_clinical_integrator_mortality"]}

{primary_context}

## Previous Debate Analysis ##
{history_text}

Start by calling retrieve() to gather medical evidence:"""
            
            mortality_tool_response = self.debate_system.integrator_llm(
                mortality_prompt,
                max_tokens=32768,
                temperature=0.3,
                top_p=0.9,
                return_format='string',
                stop_sequences=["<|im_end|>", "</s>", "End of response.", "---"],
                repetition_penalty=1.15,
                enable_thinking=True
            )
            
            # Parse tool call to extract query
            tool_name, mortality_query = self.debate_system._parse_tool_call(mortality_tool_response)
            if not mortality_query or tool_name != "retrieve":
                print(f"[WARNING] Failed to parse mortality query, using fallback")
                mortality_query = "mortality risk factors severe illness prognosis"
            
            print(f"Generated mortality query: '{mortality_query}' [{len(mortality_query)} chars]")
            
            # Step 3b: Generate SURVIVAL query via LLM tool calling
            print("Step 3b: Generating survival query via integrator LLM...")
            survival_prompt = f"""{self.debate_system.agent_prompts["balanced_clinical_integrator_survival"]}

{primary_context}

## Previous Debate Analysis ##
{history_text}

Start by calling retrieve() to gather medical evidence:"""
            
            survival_tool_response = self.debate_system.integrator_llm(
                survival_prompt,
                max_tokens=32768,
                temperature=0.3,
                top_p=0.9,
                return_format='string',
                stop_sequences=["<|im_end|>", "</s>", "End of response.", "---"],
                repetition_penalty=1.15,
                enable_thinking=True
            )
            
            # Parse tool call to extract query
            tool_name, survival_query = self.debate_system._parse_tool_call(survival_tool_response)
            if not survival_query or tool_name != "retrieve":
                print(f"[WARNING] Failed to parse survival query, using fallback")
                survival_query = "survival protective factors treatment outcomes"
            
            print(f"Generated survival query: '{survival_query}' [{len(survival_query)} chars]")
            
            # Step 3c: Execute retrieval with LLM-generated queries
            mortality_docs = []
            survival_docs = []
            
            if self.debate_system.rag_enabled:
                # Use the retrieval tool directly so logs are saved
                retrieval_tool = self.debate_system._setup_retrieval_tool(k=self.debate_system.k)
                
                print(f"Retrieving mortality evidence with LLM-generated query...")
                mortality_docs = retrieval_tool['func'](
                    mortality_query,
                    qid=f"integrator_mortality_{sample['patient_id']}",
                    log_dir=str(log_path) if log_dir else None
                )
                
                print(f"Retrieving survival evidence with LLM-generated query...")
                survival_docs = retrieval_tool['func'](
                    survival_query,
                    qid=f"integrator_survival_{sample['patient_id']}",
                    log_dir=str(log_path) if log_dir else None
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
        This EXACTLY replicates the prompt from mortality_debate_rag.py Step 1c/2c (mortality_reasoning_prompt/survival_reasoning_prompt).
        
        The prompt includes:
        1. System prompt with tool definition
        2. Primary context (## Target Patient EHR Context ##)
        3. Previous debate analysis
        4. "You called: retrieve(query)" - simulating the tool call
        5. Retrieved Evidence
        6. Final instruction to provide probability
        
        Args:
            sample: Patient sample data
            debate_result: Results from 3-round debate (contains simulated query and retrieved docs)
            assessment_type: 'mortality' or 'survival'
            
        Returns:
            Complete integrator prompt string matching downstream deployment
        """
        # Use the debate system's method to prepare debate history
        debate_history_text = self.debate_system._prepare_integrator_history(
            debate_result['debate_history']
        )
        
        # Get the appropriate system prompt based on assessment type
        if assessment_type == 'mortality':
            system_prompt = self.debate_system.agent_prompts['balanced_clinical_integrator_mortality']
            query = debate_result.get('mortality_query', 'mortality risk factors')
            retrieved_docs = debate_result.get('mortality_retrieved_docs', [])
        else:
            system_prompt = self.debate_system.agent_prompts['balanced_clinical_integrator_survival']
            query = debate_result.get('survival_query', 'survival protective factors')
            retrieved_docs = debate_result.get('survival_retrieved_docs', [])
        
        # Construct primary context - EXACTLY as in downstream (## Target Patient EHR Context ##)
        primary_context = f"## Target Patient EHR Context ##\n{debate_result['patient_context']}"
        
        # Format retrieved documents using the debate system's method
        # This ensures same formatting as deployment
        retrieved_docs_text = self.debate_system._format_retrieved_docs_for_context(retrieved_docs)
        
        # Construct the EXACT prompt format from mortality_debate_rag.py Step 1c/2c
        # This is what the model will see during GRPO training and must match deployment
        full_prompt = f"""{system_prompt}

{primary_context}

## Previous Debate Analysis ##
{debate_history_text}

You called: retrieve("{query}")

Retrieved Evidence:
{retrieved_docs_text}

Now provide your complete {assessment_type} probability assessment based on the retrieved evidence:"""
        
        return full_prompt
    
    def _generate_mortality_query(self, debate_history: List[Dict], patient_context: str) -> str:
        """
        Generate mortality-focused retrieval query based on debate history and patient context.
        Simulates what the integrator LLM would generate as a tool call query.
        
        This should be a SHORT, focused query (not the full EHR) that the integrator
        would generate via tool calling, matching the actual deployment behavior.
        """
        # Extract key conditions from patient context to create a realistic short query
        # Parse the patient context to identify main conditions
        import re
        conditions = []
        
        # Look for conditions section
        conditions_match = re.search(r'Conditions:\s*(.*?)(?:Procedures:|Medications:|$)', patient_context, re.DOTALL)
        if conditions_match:
            conditions_text = conditions_match.group(1)
            # Extract first few conditions (typically most severe)
            condition_lines = [line.strip() for line in conditions_text.split('\n') if line.strip() and re.match(r'^\d+\.', line.strip())]
            conditions = [re.sub(r'^\d+\.\s*', '', line) for line in condition_lines[:3]]  # Top 3 conditions
        
        # Create a realistic short query that an LLM would generate
        if conditions:
            # Create focused query based on main conditions
            main_condition = conditions[0] if conditions else "patient"
            query = f"{main_condition} mortality risk prognosis"
            
            # Add additional context if multiple severe conditions
            if len(conditions) > 1:
                query = f"{conditions[0]} {conditions[1]} mortality outcomes"
        else:
            # Fallback if parsing fails
            query = "mortality risk factors severe illness prognosis"
        
        return query
    
    def _generate_survival_query(self, debate_history: List[Dict], patient_context: str) -> str:
        """
        Generate survival-focused retrieval query based on debate history and patient context.
        Simulates what the integrator LLM would generate as a tool call query.
        
        This should be a SHORT, focused query (not the full EHR) that the integrator
        would generate via tool calling, matching the actual deployment behavior.
        """
        # Extract key conditions from patient context to create a realistic short query
        import re
        conditions = []
        procedures = []
        
        # Look for conditions section
        conditions_match = re.search(r'Conditions:\s*(.*?)(?:Procedures:|Medications:|$)', patient_context, re.DOTALL)
        if conditions_match:
            conditions_text = conditions_match.group(1)
            condition_lines = [line.strip() for line in conditions_text.split('\n') if line.strip() and re.match(r'^\d+\.', line.strip())]
            conditions = [re.sub(r'^\d+\.\s*', '', line) for line in condition_lines[:3]]
        
        # Look for procedures section
        procedures_match = re.search(r'Procedures:\s*(.*?)(?:Medications:|$)', patient_context, re.DOTALL)
        if procedures_match:
            procedures_text = procedures_match.group(1)
            procedure_lines = [line.strip() for line in procedures_text.split('\n') if line.strip() and re.match(r'^\d+\.', line.strip())]
            procedures = [re.sub(r'^\d+\.\s*', '', line) for line in procedure_lines[:2]]
        
        # Create a realistic short query focusing on treatment and recovery
        if conditions and procedures:
            query = f"{conditions[0]} treatment {procedures[0]} recovery outcomes"
        elif conditions:
            query = f"{conditions[0]} treatment survival recovery"
        elif procedures:
            query = f"{procedures[0]} patient recovery outcomes"
        else:
            query = "treatment recovery survival outcomes"
        
        return query
    
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
        
        # NOTE: Keep extra_info as dict - Parquet handles nested structures fine
        # VERL's dataset loader expects dict, not JSON string
        # Do NOT convert: df['extra_info'] = df['extra_info'].apply(json.dumps)
        
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
    
    # Set CUDA_VISIBLE_DEVICES before any CUDA imports to prevent GPU leakage
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f"Setting CUDA_VISIBLE_DEVICES={args.gpus}")
    
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
