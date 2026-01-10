#!/usr/bin/env python3
"""
Single-Agent CoT System for KARE Mortality Prediction
Adapted from mortality_debate_rag_fast.py with minimal changes.
Uses KARE zero_shot_base style prompting with Chain-of-Thought only (no RAG).
"""

import os
import sys
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Use VLLM directly without MedRAG
from vllm import LLM, SamplingParams

class MortilitySingleAgentCoT:
    """
    Single-agent system for KARE mortality prediction with Chain-of-Thought only.
    Minimal changes from mortality_debate_rag_fast.py - only integrator agent remains, no RAG.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 gpu_id: str = "6",
                 in_context: str = "zero-shot"):
        """
        Initialize the single-agent CoT system.
        
        Args:
            model_name: HuggingFace model name for VLLM
            gpu_id: GPU ID to use (single GPU only)
            in_context: 'zero-shot' or 'few-shot' (whether to use similar patients)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.in_context = in_context
        
        print(f"Single-agent CoT model ({model_name}) will use GPU: {self.gpu_id}")
        print(f"In-context mode: {in_context}")
        
        # Set CUDA_VISIBLE_DEVICES to single GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        print(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id}")
        
        # Initialize VLLM directly (no MedRAG needed for CoT)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enforce_eager=True
        )
        print(f"VLLM initialized for {model_name}")
        
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
            
            logger = logging.getLogger(f"single_agent_cot_{patient_id}")
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_file, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        else:
            logger = None
        
        # Construct KARE-style prompt based on in_context mode
        if self.in_context == "zero-shot":
            # Zero-shot: no similar patients
            user_prompt = f"""Given the following task description and patient context, please make a prediction with reasoning based on the patient's context.

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
        else:  # few-shot
            # Few-shot: include similar patients
            user_prompt = f"""Given the following task description, patient EHR context, similar patients, please make a prediction with reasoning based on the patient's context.

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

        if logger:
            logger.info("="*80)
            logger.info(f"SINGLE AGENT COT ({self.in_context.upper()}) - Patient: {patient_id}")
            logger.info("="*80)
            logger.info(f"\nUSER PROMPT:\n{user_prompt}")
        
        # Format prompt for model (no system prompt needed for KARE style)
        if "qwen" in self.model_name.lower():
            formatted_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = user_prompt
        
        # Generate response with CoT
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=32768,
            stop=["<|im_end|>", "</s>"],
            repetition_penalty=1.2
        )
        
        print(f"[AGENT] Generating CoT response...")
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        if logger:
            logger.info(f"\n\nAGENT RESPONSE:\n{response_text}")
        
        # Extract prediction
        result = self._extract_prediction_and_probabilities(response_text, ground_truth=ground_truth)
        
        total_time = time.time() - start_time
        
        return {
            'final_prediction': result['prediction'],
            'is_fallback': result.get('is_fallback', False),
            'total_generation_time': total_time,
            'response': response_text
        }


# Test the single-agent system
if __name__ == "__main__":
    try:
        print("Testing Single-Agent CoT System...")
        system = MortilitySingleAgentCoT(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_id="6"
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
