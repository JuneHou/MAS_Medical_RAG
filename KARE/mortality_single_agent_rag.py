#!/usr/bin/env python3
"""
Single-Agent RAG System for KARE Mortality Prediction
Adapted from mortality_debate_rag_fast.py with minimal changes.
Uses KARE zero_shot_base style prompting with MedRAG retrieval.
"""

import os
import sys
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add MedRAG paths for VLLM wrapper and retrieval
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"

sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))
sys.path.insert(0, mirage_src)

from run_medrag_vllm import VLLMWrapper, patch_medrag_for_vllm
from medrag import MedRAG
from vllm import SamplingParams

class MortilitySingleAgentRAG:
    """
    Single-agent system for KARE mortality prediction with MedRAG retrieval.
    Minimal changes from mortality_debate_rag_fast.py - only integrator agent remains.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 gpu_ids: str = "6,7",
                 corpus_name: str = "MedCorp2",
                 retriever_name: str = "MedCPT",
                 db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
                 in_context: str = "zero-shot"):
        """
        Initialize the single-agent RAG system.
        
        Args:
            model_name: HuggingFace model name for VLLM
            gpu_ids: GPU IDs to use (comma-separated)
            corpus_name: MedRAG corpus name
            retriever_name: MedRAG retriever name
            db_dir: MedRAG database directory
            in_context: 'zero-shot' or 'few-shot' (whether to use similar patients)
        """
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.in_context = in_context
        
        # Set CUDA_VISIBLE_DEVICES to all GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_ids}")
        print(f"In-context mode: {in_context}")
        
        # Count available GPUs for tensor parallelism
        gpu_list = gpu_ids.split(',')
        num_gpus = len(gpu_list)
        print(f"Using {num_gpus} GPU(s) - Retriever on cuda:0, LLM with tensor_parallel_size={num_gpus}")
        
        # Store RAG configuration
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.db_dir = db_dir
        
        # Initialize MedRAG for retrieval only (retriever uses cuda:0)
        print(f"Initializing MedRAG with {corpus_name} corpus and {retriever_name} retriever...")
        patch_medrag_for_vllm()
        self.medrag = MedRAG(
            llm_name=model_name,
            rag=True,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir,
            corpus_cache=True,
            HNSW=True,
            retriever_device="cuda:0"  # Use first visible GPU for retriever
        )
        # Create retrieval tool
        self.retrieval_tool = self._create_retrieval_tool(k=8)
        print("MedRAG initialization complete.")
        
        # Initialize VLLM for generation with tensor parallelism across all GPUs
        print(f"Initializing VLLM with tensor parallelism across {num_gpus} GPU(s)...")
        # Use lower gpu_memory_utilization for tensor parallelism (model is split across GPUs)
        gpu_util = 0.35 if num_gpus >= 2 else 0.5  # TP mode needs less per-GPU, single GPU needs more headroom for retriever
        self.llm = VLLMWrapper(
            model_name=model_name, 
            enable_thinking=True, 
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util
        )
        print(f"VLLM initialized successfully with TP={num_gpus}, gpu_util={gpu_util}")
        
        # KARE task description
        self.task_description = """Mortality Prediction Task:
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

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1."""
        
        # Instruction for using retrieval
        self.retrieval_instruction = """Before making your prediction, you should retrieve relevant medical evidence using retrieve(query) to support your reasoning."""
    
    def _create_retrieval_tool(self, k=8):
        """Create a retrieval tool for medical evidence"""
        def retrieve_tool(query, qid=None, log_dir=None):
            """Retrieve medical evidence using MedRAG"""
            print(f"[RETRIEVE] Query: {query[:100]}...")
            
            # Truncate query if too long (max 200 chars)
            if len(query) > 200:
                query = query[:200]
                print(f"[RETRIEVE] Truncated query to 200 chars")
            
            try:
                # Use MedRAG's source retrievers directly (following original debate pattern)
                if hasattr(self.medrag, 'source_retrievers') and self.medrag.corpus_name == "MedCorp2":
                    print(f"[RETRIEVE] Using MedCorp2 source retrievers")
                    
                    # Split k between sources
                    k_medcorp = k // 2 + k % 2  # Give extra to general literature if odd
                    k_umls = k // 2
                    
                    all_snippets = []
                    all_scores = []
                    
                    for source, k_source in [("medcorp", k_medcorp), ("umls", k_umls)]:
                        if source in self.medrag.source_retrievers:
                            source_retrieval_system = self.medrag.source_retrievers[source]
                            snippets, scores = source_retrieval_system.retrieve(query, k=k_source, rrf_k=60)
                            all_snippets.extend(snippets)
                            all_scores.extend(scores)
                    
                    retrieved_snippets = all_snippets
                    scores = all_scores
                    
                elif hasattr(self.medrag, 'retrieval_system') and self.medrag.retrieval_system:
                    print(f"[RETRIEVE] Using retrieval system")
                    retrieved_snippets, scores = self.medrag.retrieval_system.retrieve(query, k=k, rrf_k=60)
                else:
                    print(f"[RETRIEVE ERROR] No retrieval system available")
                    return []
                
                # Format results
                results = []
                for i, (snippet, score) in enumerate(zip(retrieved_snippets, scores)):
                    if isinstance(snippet, dict):
                        content = snippet.get('content', snippet.get('contents', str(snippet)))
                    else:
                        content = str(snippet)
                    
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'content': content[:500]  # Limit content length
                    })
                
                print(f"[RETRIEVE] Retrieved {len(results)} documents")
                return results
                
            except Exception as e:
                print(f"[RETRIEVE ERROR] {e}")
                return []
        
        return {"name": "retrieve", "func": retrieve_tool}
    
    def _parse_tool_call(self, response_text):
        """Parse tool call from agent response"""
        patterns = [
            r'retrieve\s*\(\s*["\']([^"\']+)["\'\s]*\)',
            r'Tool Call:\s*retrieve\s*\(\s*["\']([^"\']+)["\'\s]*\)',
            r'RETRIEVE\s*\(\s*["\']([^"\']+)["\'\s]*\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                print(f"[TOOL CALL] Detected retrieve('{query[:50]}...')")
                return "retrieve", query
        
        return None, None
    
    def _execute_tool_call(self, tool_name, query, qid=None, log_dir=None):
        """Execute a tool call"""
        if tool_name == "retrieve":
            return self.retrieval_tool["func"](query, qid, log_dir)
        return []
    
    def _format_retrieved_docs_for_context(self, retrieved_docs):
        """Format retrieved documents for inclusion in prompt"""
        if not retrieved_docs:
            return "No medical evidence retrieved."
            
        docs_context = ""
        for i, doc in enumerate(retrieved_docs):
            docs_context += f"\n[Document {i+1}, Score: {doc.get('score', 0):.3f}]\n{doc.get('content', '')}\n"
        
        return docs_context.strip()
    
    def _extract_prediction_and_probabilities(self, response, ground_truth=None) -> Dict[str, Any]:
        """Extract prediction from agent response (KARE format: 1/0)"""
        if isinstance(response, list):
            response = response[0].get('generated_text', '') if response else ''
        elif not isinstance(response, str):
            response = str(response)
        
        result = {
            'prediction': None,
            'is_fallback': False,
        }
        
        # Extract prediction (1/0 format) - KARE style
        prediction_patterns = [
            r'#\s*Prediction\s*#[\s:]*([01])',
            r'Prediction[:\s]+([01])',
            r'\*\*Prediction\*\*[:\s]+([01])',
            r'prediction[:\s]+([01])',
        ]
        
        for pattern in prediction_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                pred = int(match.group(1))
                result['prediction'] = pred
                result['is_fallback'] = False
                return result
        
        # Fallback: use opposite of ground truth
        if ground_truth is not None:
            result['prediction'] = 1 - ground_truth
        else:
            result['prediction'] = 0  # Default to survival
        
        result['is_fallback'] = True
        
        return result
    
    def predict_mortality(self, 
                         patient_context: str,
                         positive_similars: str,
                         negative_similars: str,
                         patient_id: str = "unknown",
                         output_dir: str = None,
                         ground_truth: int = None) -> Dict[str, Any]:
        """
        Single-agent mortality prediction with RAG retrieval.
        
        Args:
            patient_context: Target patient clinical information
            positive_similars: Similar patients who died
            negative_similars: Similar patients who survived
            patient_id: Patient ID
            output_dir: Output directory for logs
            ground_truth: Ground truth label (0=survive, 1=mortality)
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Setup logging
        if output_dir:
            log_dir = Path(output_dir) / f"debate_logs_{self.in_context.replace('-', '_')}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"single_agent_rag_{patient_id}.log"
            
            logger = logging.getLogger(f"single_agent_rag_{patient_id}")
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_file, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        else:
            logger = None
        
        # Construct KARE-style prompt for initial retrieval request
        if self.in_context == "zero-shot":
            # Zero-shot: no similar patients
            retrieval_prompt = f"""Given the following task and patient context, first retrieve relevant medical evidence, then make a prediction.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

{self.retrieval_instruction}"""
        else:  # few-shot
            # Few-shot: include similar patients
            retrieval_prompt = f"""Given the following task, patient EHR context, and similar patients, first retrieve relevant medical evidence, then make a prediction.

========================================
# Task #
{self.task_description}

========================================
# Patient EHR Context #
{patient_context}

========================================
# Similar Patients #
Similar Patients Who Died:
{positive_similars}

Similar Patients Who Survived:
{negative_similars}

========================================

{self.retrieval_instruction}"""

        if logger:
            logger.info("="*80)
            logger.info(f"SINGLE AGENT RAG ({self.in_context.upper()}) - Patient: {patient_id}")
            logger.info("="*80)
            logger.info(f"\nRETRIEVAL PROMPT:\n{retrieval_prompt}")
        
        # First call - agent should request retrieval
        if "qwen" in self.model_name.lower():
            formatted_prompt = f"<|im_start|>user\n{retrieval_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = retrieval_prompt
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["<|im_end|>", "</s>"],
            repetition_penalty=1.2
        )
        
        print(f"[AGENT] Generating initial response...")
        outputs = self.llm.llm.generate([formatted_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        if logger:
            logger.info(f"\n\nAGENT INITIAL RESPONSE:\n{response_text}")
        
        # Check for tool call
        tool_name, query = self._parse_tool_call(response_text)
        
        if tool_name == "retrieve":
            # Execute retrieval
            retrieved_docs = self._execute_tool_call(tool_name, query, qid=patient_id, log_dir=output_dir)
            docs_context = self._format_retrieved_docs_for_context(retrieved_docs)
            
            if logger:
                logger.info(f"\n\nRETRIEVED EVIDENCE:\n{docs_context}")
            
            # Second call with KARE-style prompt including supplementary information
            if self.in_context == "zero-shot":
                final_prompt = f"""Given the following task description, patient context, and relevant supplementary information, please make a prediction with reasoning based on the patient's context and relevant medical knowledge.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

# Supplementary Information #
{docs_context}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:"""
            else:  # few-shot
                final_prompt = f"""Given the following task description, patient EHR context, similar patients, and relevant supplementary information, please make a prediction with reasoning based on the patient's context and relevant medical knowledge.

========================================
# Task #
{self.task_description}

========================================
# Patient EHR Context #
{patient_context}

========================================
# Similar Patients #
Similar Patients Who Died:
{positive_similars}

Similar Patients Who Survived:
{negative_similars}

========================================
# Supplementary Information #
{docs_context}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:"""

            if "qwen" in self.model_name.lower():
                formatted_prompt = f"<|im_start|>user\n{final_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = final_prompt
            
            print(f"[AGENT] Generating final response with evidence...")
            outputs = self.llm.llm.generate([formatted_prompt], sampling_params)
            final_response = outputs[0].outputs[0].text
            
            if logger:
                logger.info(f"\n\nFINAL PROMPT:\n{final_prompt}")
                logger.info(f"\n\nAGENT FINAL RESPONSE:\n{final_response}")
        else:
            final_response = response_text
        
        # Extract prediction
        result = self._extract_prediction_and_probabilities(final_response, ground_truth=ground_truth)
        
        total_time = time.time() - start_time
        
        return {
            'final_prediction': result['prediction'],
            'is_fallback': result.get('is_fallback', False),
            'total_generation_time': total_time,
            'response': final_response
        }


# Test the single-agent system
if __name__ == "__main__":
    try:
        print("Testing Single-Agent RAG System...")
        system = MortilitySingleAgentRAG(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_ids="6,7"
        )
        
        # Test patient
        patient_context = """Patient: 85-year-old male
- Chief complaint: Septic shock
- Vital signs: BP 80/40, HR 120, Temp 39.5°C
- Lab: WBC 22k, Lactate 4.5
- Medical history: CHF, CKD stage 3"""

        positive_similars = """Similar patient 1 (died): 82F, septic shock, APACHE II 28
Similar patient 2 (died): 88M, septic shock with ARDS"""

        negative_similars = """Similar patient 1 (survived): 79M, sepsis, early antibiotics
Similar patient 2 (survived): 81F, UTI sepsis, good response"""

        result = system.predict_mortality(
            patient_context=patient_context,
            positive_similars=positive_similars,
            negative_similars=negative_similars,
            patient_id="test_001"
        )
        
        print(f"\nPrediction: {result['final_prediction']}")
        print(f"Mortality Prob: {result['mortality_probability']}")
        print(f"Survival Prob: {result['survival_probability']}")
        print(f"Time: {result['total_generation_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
