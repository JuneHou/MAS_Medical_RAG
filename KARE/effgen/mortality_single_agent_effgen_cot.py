#!/usr/bin/env python3
"""
Single-Agent CoT System for KARE Mortality Prediction using effGen
Chain-of-Thought reasoning without external retrieval
Supports both zero-shot and few-shot modes
"""

import os
import sys
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import effgen components
try:
    from effgen import Agent, load_model
    from effgen.core.agent import AgentConfig
except ImportError as e:
    print(f"Error importing effgen: {e}")
    print("Please install effgen: pip install effgen")
    sys.exit(1)


class MortilitySingleAgentEffGenCoT:
    """
    Single-agent system for KARE mortality prediction with Chain-of-Thought only.
    Supports zero-shot (no similar patients) and few-shot (with similar patients) modes.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 gpu_id: str = "0",
                 in_context: str = "zero-shot",
                 model_cache_dir: str = "/data/wang/junh/.cache/huggingface"):
        """
        Initialize the single-agent CoT system.
        
        Args:
            model_name: HuggingFace model name
            gpu_id: GPU ID to use (single GPU only)
            in_context: 'zero-shot' or 'few-shot' (whether to use similar patients)
            model_cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.in_context = in_context
        self.model_cache_dir = model_cache_dir
        
        print(f"[INIT] Single-agent CoT model ({model_name}) - GPU: {self.gpu_id}")
        print(f"[INIT] In-context mode: {in_context}")
        
        # Set CUDA_VISIBLE_DEVICES once (following MEDRAG_GPU_SETUP_FIX.md)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        print(f"[CUDA] Set CUDA_VISIBLE_DEVICES={self.gpu_id}")
        
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
        
        # Create agent (no system prompt - KARE uses task in user prompt)
        self.agent = Agent(
            config=AgentConfig(
                name="mortality_predictor_cot",
                model=self.model,
                tools=[],  # No tools for CoT mode
                system_prompt="",  # No system prompt for KARE style
                max_iterations=1,  # Single-turn execution
                enable_sub_agents=False,  # Disable sub-agents
                enable_memory=False  # Disable memory for reproducibility
            )
        )
        print(f"[AGENT] Agent created successfully")
    
    def _build_zero_shot_prompt(self, patient_context: str) -> str:
        """Build zero-shot KARE-style prompt (no similar patients)."""
        return f"""Given the following task description and patient context, please make a prediction with reasoning based on the patient's context.

# Task # 
{self.task_description}
========================================

# Patient Context #
{patient_context}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #  
[Your prediction here (1/0)]

Output:"""
    
    def _build_few_shot_prompt(self, patient_context: str, 
                               positive_similars: str, 
                               negative_similars: str) -> str:
        """Build few-shot KARE-style prompt (with similar patients)."""
        return f"""Given the following task description, patient EHR context, similar patients, please make a prediction with reasoning based on the patient's context.

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

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #  
[Your prediction here (1/0)]


Output:"""
    
    def _extract_prediction(self, response: str, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract prediction from agent response (KARE format: 1/0).
        
        Args:
            response: Agent response text
            ground_truth: Ground truth label for fallback
            
        Returns:
            Dictionary with prediction and is_fallback flag
        """
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
        Single-agent mortality prediction with Chain-of-Thought only.
        
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
            log_file = log_dir / f"single_agent_cot_{patient_id}.log"
            
            logger = logging.getLogger(f"single_agent_effgen_cot_{patient_id}")
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            handler = logging.FileHandler(log_file, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        else:
            logger = None
        
        # Build KARE-style prompt based on in_context mode
        if self.in_context == "zero-shot":
            prompt = self._build_zero_shot_prompt(patient_context)
        else:  # few-shot
            prompt = self._build_few_shot_prompt(patient_context, positive_similars, negative_similars)
        
        if logger:
            logger.info("="*80)
            logger.info(f"SINGLE AGENT COT ({self.in_context.upper()}) - Patient: {patient_id}")
            logger.info("="*80)
            logger.info(f"\nUSER PROMPT:\n{prompt}")
        
        # Run agent with effGen
        try:
            print(f"[AGENT] Generating CoT response for {patient_id}...")
            # Import AgentMode to explicitly specify single agent execution
            from effgen.core.agent import AgentMode
            result = self.agent.run(
                prompt,
                mode=AgentMode.SINGLE,
                temperature=0.7,  # Match original vllm
                top_p=0.9,  # Match original vllm
                max_tokens=32768,
                repetition_penalty=1.2,  # Match original vllm
                stop_sequences=["<|im_end|>", "</s>"]  # Match original vllm
            )
            
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
        print("Testing Single-Agent effGen CoT System...")
        system = MortilitySingleAgentEffGenCoT(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_id="0",
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
