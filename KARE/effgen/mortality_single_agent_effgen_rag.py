#!/usr/bin/env python3
"""
Single-Agent RAG System for KARE Mortality Prediction using effGen
With MedRAG retrieval from MedCorp2 corpus
Supports both zero-shot and few-shot modes

ReACT agent pattern: follows effgen example `agentic_search_agent.py`:
- Agent with tools (MedRAGRetrievalTool), system_prompt, max_iterations=5
- Single agent.run(prompt) — effGen runs the ReAct loop (tool use + reasoning) internally
- No explicit mode parameter; sub-agents and memory disabled for reproducibility
"""

import os
import sys
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory and MedRAG paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"
sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))
sys.path.insert(0, mirage_src)

# Import effgen components
try:
    from effgen import Agent, load_model
    from effgen.core.agent import AgentConfig
except ImportError as e:
    print(f"Error importing effgen: {e}")
    print("Please install effgen: pip install effgen")
    sys.exit(1)

# Import MedRAG
try:
    from run_medrag_vllm import patch_medrag_for_vllm
    from medrag import MedRAG
except ImportError as e:
    print(f"Error importing MedRAG: {e}")
    sys.exit(1)

# Import custom MedRAG tool
try:
    from effgen_medrag_tool import MedRAGRetrievalTool
except ImportError:
    print("Error: effgen_medrag_tool.py not found")
    sys.exit(1)


class MortilitySingleAgentEffGenRAG:
    """
    Single-agent system for KARE mortality prediction with MedRAG retrieval.
    Supports zero-shot (no similar patients) and few-shot (with similar patients) modes.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 gpu_ids: str = "0,1",
                 in_context: str = "zero-shot",
                 model_cache_dir: str = "/data/wang/junh/.cache/huggingface",
                 corpus_name: str = "MedCorp2",
                 retriever_name: str = "MedCPT",
                 db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"):
        """
        Initialize the single-agent RAG system.
        
        Args:
            model_name: HuggingFace model name
            gpu_ids: GPU IDs to use (comma-separated)
            in_context: 'zero-shot' or 'few-shot'
            model_cache_dir: Cache directory for models
            corpus_name: MedRAG corpus name
            retriever_name: MedRAG retriever name
            db_dir: MedRAG database directory
        """
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.in_context = in_context
        self.model_cache_dir = model_cache_dir
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.db_dir = db_dir
        
        print(f"[INIT] Single-agent RAG model ({model_name}) - GPUs: {self.gpu_ids}")
        print(f"[INIT] In-context mode: {in_context}")
        print(f"[INIT] MedRAG Corpus: {corpus_name}, Retriever: {retriever_name}")
        
        # Set CUDA_VISIBLE_DEVICES to all GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        print(f"[CUDA] Set CUDA_VISIBLE_DEVICES={self.gpu_ids}")
        
        # Get first GPU for retriever
        gpu_list = self.gpu_ids.split(',')
        self.main_gpu = gpu_list[0]
        
        # Initialize MedRAG FIRST (before model loading)
        print(f"[MedRAG] Initializing MedRAG retrieval system...")
        try:
            patch_medrag_for_vllm()
            self.medrag = MedRAG(
                llm_name=model_name,
                rag=True,
                retriever_name=retriever_name,
                corpus_name=corpus_name,
                db_dir=db_dir,
                corpus_cache=True,
                HNSW=True,
                retriever_device=f"cuda:{self.main_gpu}"
            )
            print(f"[MedRAG] MedRAG initialization complete")
        except Exception as e:
            print(f"[ERROR] MedRAG initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Load model using effgen (full precision, no quantization)
        print(f"[MODEL] Loading model with effGen...")
        try:
            # Use model name directly - let effgen/transformers find it in cache
            # Set HF_HOME to point to cache directory
            os.environ['HF_HOME'] = model_cache_dir
            print(f"[MODEL] Set HF_HOME={model_cache_dir}")
            print(f"[MODEL] Loading model: {model_name}")
            
            # Configure model to use GPU explicitly
            import torch
            print(f"[MODEL] CUDA available: {torch.cuda.is_available()}")
            print(f"[MODEL] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"[MODEL] Number of visible GPUs: {num_gpus}")
                for i in range(num_gpus):
                    print(f"[MODEL] GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Load model with explicit device mapping
            # When CUDA_VISIBLE_DEVICES=1, GPU 1 becomes device 0 in torch
            # So we use device_map={"": 0} to explicitly use the first visible device
            self.model = load_model(
                model_name,
                device_map={"": 0},  # Explicitly use device 0 (which is physical GPU based on CUDA_VISIBLE_DEVICES)
                trust_remote_code=True,  # Required for some models like Qwen
                attn_implementation="eager"  # Use standard attention instead of flash attention
            )
            print(f"[MODEL] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # KARE task description
        self.task_description = """Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 1 = mortality, 0 = survival
IMPORTANT: Mortality is rare - only predict mortality probability > 0.5 if evidence STRONGLY supports it. When uncertain, predict survival probability > 0.5. The Target patient is the source of truth. Do not treat Similar-only items as present in the Target.


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
        
        # Retrieval instruction
        self.retrieval_instruction = """Before making your prediction, you should retrieve relevant medical evidence using retrieve(query) to support your reasoning."""
        
        # Will be set per-prediction with log_dir
        self.retrieval_tool = None
        self.agent = None
    
    def _create_agent_with_tool(self, log_dir: str):
        """Create agent with retrieval tool (needs log_dir for tool initialization)."""
        # Create retrieval tool with conservative query length (200 chars for single-agent)
        self.retrieval_tool = MedRAGRetrievalTool(
            medrag_instance=self.medrag,
            k=8,
            max_query_tokens=50,  # ~200 chars (conservative for single-agent)
            log_dir=log_dir
        )
        
        # Create agent with retrieval tool (ReACT pattern from effgen examples/agentic_search_agent.py)
        # System prompt instructs tool use; max_iterations=5 allows retrieval + reasoning steps
        system_prompt = """You are a medical AI Clinical Assistant with access to medical evidence retrieval.

CRITICAL: You MUST use the retrieve_medical_evidence tool to search for clinical evidence BEFORE making any prediction. Never make predictions without first retrieving evidence.

When you receive a mortality prediction task:
1) FIRST: Use retrieve_medical_evidence tool to search for relevant medical evidence about the patient's conditions
2) THEN: After receiving the retrieved evidence, analyze the information
3) FINALLY: Make your prediction based on the evidence

The retrieve_medical_evidence tool searches medical literature (MedCorp2) and terminology (UMLS) for clinical evidence and prognosis information.

IMPORTANT: Mortality is rare - only predict high mortality probability when evidence STRONGLY supports it."""
        
        self.agent = Agent(
            config=AgentConfig(
                name="mortality_predictor_rag",
                model=self.model,
                tools=[self.retrieval_tool],  # Include retrieval tool
                system_prompt=system_prompt,  # Enable ReAct loop for tool usage
                max_iterations=5,  # Allow for retrieval + reasoning (match effgen example)
                temperature=0.5,  # Changed from 0.7 to 0.5
                enable_sub_agents=False,  # Disable sub-agents
                enable_memory=False  # Disable memory for reproducibility
            )
        )
        print(f"[AGENT] Agent created with retrieval tool")
    
    def _build_zero_shot_retrieval_prompt(self, patient_context: str) -> str:
        """Build zero-shot retrieval request prompt."""
        return f"""Given the following task and patient context, first retrieve relevant medical evidence, then make a prediction.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

{self.retrieval_instruction}"""
    
    def _build_few_shot_retrieval_prompt(self, patient_context: str, 
                                         positive_similars: str, 
                                         negative_similars: str) -> str:
        """Build few-shot retrieval request prompt."""
        return f"""Given the following task, patient EHR context, and similar patients, first retrieve relevant medical evidence, then make a prediction.

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
    
    def _build_zero_shot_final_prompt(self, patient_context: str, docs_context: str) -> str:
        """Build zero-shot final prediction prompt with supplementary information."""
        return f"""Given the following task description, patient context, and relevant supplementary information, please make a prediction with reasoning based on the patient's context and relevant medical knowledge.

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
    
    def _build_few_shot_final_prompt(self, patient_context: str, 
                                     positive_similars: str, 
                                     negative_similars: str, 
                                     docs_context: str) -> str:
        """Build few-shot final prediction prompt with supplementary information."""
        return f"""Given the following task description, patient EHR context, similar patients, and relevant supplementary information, please make a prediction with reasoning based on the patient's context and relevant medical knowledge.

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
    
    def _extract_prediction(self, response: str, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """Extract prediction from agent response (KARE format: 1/0)."""
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
        print(f"[WARNING] No prediction found in response, using fallback: {result['prediction']}")
        
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
            
            logger = logging.getLogger(f"single_agent_effgen_rag_{patient_id}")
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            handler = logging.FileHandler(log_file, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            
            # Create agent with tool now that we have log_dir
            self._create_agent_with_tool(str(log_dir))
        else:
            logger = None
            # Create agent without log_dir
            self._create_agent_with_tool(None)
        
        try:
            # Build initial retrieval request prompt
            if self.in_context == "zero-shot":
                prompt = self._build_zero_shot_retrieval_prompt(patient_context)
            else:  # few-shot
                prompt = self._build_few_shot_retrieval_prompt(patient_context, positive_similars, negative_similars)
            
            if logger:
                logger.info("="*80)
                logger.info(f"SINGLE AGENT RAG ({self.in_context.upper()}) - Patient: {patient_id}")
                logger.info("="*80)
                logger.info(f"\nRETRIEVAL PROMPT:\n{prompt}")
            
            print(f"[AGENT] Running RAG agent for {patient_id}...")
            
            # Run agent (effGen will handle tool calling automatically via ReAct)
            # Don't specify mode - let agent use ReAct loop with tools
            result = self.agent.run(prompt)
            
            # Extract response text
            if hasattr(result, 'output'):
                response_text = result.output
            elif isinstance(result, dict):
                response_text = result.get('output', str(result))
            else:
                response_text = str(result)
            
            if logger:
                logger.info(f"\n\nAGENT RESPONSE:\n{response_text}")
            
            # Extract prediction
            prediction_result = self._extract_prediction(response_text, ground_truth=ground_truth)
            
            if logger:
                logger.info(f"\n\nEXTRACTED PREDICTION: {prediction_result['prediction']}")
                logger.info(f"IS FALLBACK: {prediction_result['is_fallback']}")
        
        except Exception as e:
            print(f"[ERROR] Prediction failed for {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Use fallback
            prediction_result = {
                'prediction': 1 - ground_truth if ground_truth is not None else 0,
                'is_fallback': True
            }
            response_text = f"Error: {str(e)}"
        
        # Clean up logger
        if logger:
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                logger.removeHandler(handler)
        
        total_time = time.time() - start_time
        
        return {
            'final_prediction': prediction_result['prediction'],
            'is_fallback': prediction_result.get('is_fallback', False),
            'total_generation_time': total_time,
            'response': response_text
        }


# Test the single-agent system
if __name__ == "__main__":
    try:
        print("Testing Single-Agent effGen RAG System...")
        system = MortilitySingleAgentEffGenRAG(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_ids="0,1",
            in_context="zero-shot"
        )
        
        # Test patient
        patient_context = """Patient: 85-year-old male

Visit 0:
Conditions:
1. Septic shock
2. Acute respiratory failure
3. Chronic heart failure

Procedures:
1. Mechanical ventilation
2. Central line insertion

Medications:
1. Norepinephrine
2. Broad-spectrum antibiotics
3. Furosemide"""

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
        print(f"Is Fallback: {result['is_fallback']}")
        print(f"Time: {result['total_generation_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
