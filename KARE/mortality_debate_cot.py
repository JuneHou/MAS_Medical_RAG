#!/usr/bin/env python3
"""
Enhanced Multi-Agent Debate System for KARE Mortality Prediction
Integrates with KARE's precomputed similar patients and medical knowledge.
"""

import os
import sys
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add MedRAG paths for VLLM wrapper - make configurable for different servers
medrag_root = os.environ.get("MEDRAG_ROOT", "/data/wang/junh/githubs/mirage_medrag/MedRAG")
sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))

from run_medrag_vllm import VLLMWrapper

class MortalityDebateSystem:
    """
    Enhanced debate system specifically designed for KARE mortality prediction task.
    Uses three specialized agents with distinct analytical perspectives.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507", 
                 gpu_ids: str = "6,7",
                 integrator_model_name: str = None,
                 integrator_gpu: str = None):
        """
        Initialize the mortality debate system.
        
        Args:
            model_name: HuggingFace model name for VLLM (agents 1-3)
            gpu_ids: GPU IDs to use for main model (comma-separated)
            integrator_model_name: Model name for integrator agent. If None, uses model_name
            integrator_gpu: GPU ID for integrator model. If None, uses second GPU from gpu_ids
        """
        # Store model configurations
        self.model_name = model_name
        self.integrator_model_name = integrator_model_name or model_name
        self.gpu_ids = gpu_ids
        
        # Determine integrator GPU
        gpu_list = gpu_ids.split(',')
        self.main_gpu = gpu_list[0]
        self.integrator_gpu = integrator_gpu or (gpu_list[1] if len(gpu_list) > 1 else gpu_list[0])
        
        print(f"Main model ({model_name}) will use GPU: {self.main_gpu}")
        print(f"Integrator model ({self.integrator_model_name}) will use GPU: {self.integrator_gpu}")
        
        # Initialize main VLLM wrapper for agents 1-3
        os.environ['CUDA_VISIBLE_DEVICES'] = self.main_gpu
        self.llm = VLLMWrapper(model_name=model_name)
        
        # Initialize integrator VLLM wrapper if different model
        if self.integrator_model_name != model_name:
            print(f"Initializing separate integrator model: {self.integrator_model_name}")
            
            # Handle multi-GPU for integrator model with tensor parallelism
            integrator_gpu_list = self.integrator_gpu.split(',') if ',' in str(self.integrator_gpu) else [str(self.integrator_gpu)]
            tensor_parallel_size = len(integrator_gpu_list)
            
            print(f"DEBUG: Integrator will use {tensor_parallel_size} GPU(s): {integrator_gpu_list}")
            os.environ['CUDA_VISIBLE_DEVICES'] = self.integrator_gpu
            
            # Initialize with tensor parallelism if multiple GPUs
            if tensor_parallel_size > 1:
                self.integrator_llm = VLLMWrapper(
                    model_name=self.integrator_model_name, 
                    tensor_parallel_size=tensor_parallel_size
                )
            else:
                self.integrator_llm = VLLMWrapper(model_name=self.integrator_model_name)
        else:
            print("Using same model for integrator")
            self.integrator_llm = self.llm
        
        # Agent configuration - Four specialized agents
        self.agent_roles = [
            "target_patient_analyst",
            "positive_similar_comparator", 
            "negative_similar_comparator",
            "medical_knowledge_integrator"
        ]
        self.max_rounds = 3
        
        # System prompts for each agent
        self.agent_prompts = self._initialize_agent_prompts()
        
    def _initialize_agent_prompts(self) -> Dict[str, str]:
        """Initialize specialized prompts for each agent role."""
        
        return {
            "target_patient_analyst": """You are a medical AI that predicts whether a patient will die in the NEXT hospital visit.

You see ONLY the target patient's temporal EHR context.

Your job:
1) Read all visits in order and summarize the main clinical story (conditions, procedures, medications).
2) List totally 5 main factors that contribute to mortality or survival of patient.
3) Provide a brief explanation (2–3 sentences).
4) Make an initial prediction.

CRITICAL FORMAT REQUIREMENT: You MUST end your response with exactly one of these lines:
\\boxed{0} - if you predict SURVIVAL (patient will NOT die in next visit)
\\boxed{1} - if you predict MORTALITY (patient WILL die in next visit)

Do not include anything after the \\boxed{} prediction.""",

            "positive_similar_comparator": """You are a medical AI that analyzes similar patients who DIED (mortality = 1).

Your job (EVIDENCE ONLY, NO FINAL LABEL):
1) Summarize the common patterns in these fatal cases (diseases, procedures, medications, trajectories).
2) List key factors that appear often in these deaths and clearly increase mortality risk.
3) Briefly explain (2–3 sentences) why these factors are warning signs for death in the next visit.

IMPORTANT:
- DO NOT make a final prediction.
- DO NOT output \\boxed{0} or \\boxed{1}.
- Your job is only to provide EVIDENCE that supports MORTALITY (death in the next visit).""",

            "negative_similar_comparator": """You are a medical AI that analyzes similar patients who SURVIVED (mortality = 0).

Your job (EVIDENCE ONLY, NO FINAL LABEL):
1) Summarize the common patterns in these survivor cases (conditions, treatments, recovery trajectories).
2) List key factors that appear often in these survivors and clearly support survival or stability.
3) Briefly explain (2–3 sentences) why these factors are associated with LOW mortality risk in the next visit.

IMPORTANT:
- DO NOT make a final prediction.
- DO NOT output \\boxed{0} or \\boxed{1}.
- Your job is only to provide EVIDENCE that supports SURVIVAL (no death in the next visit).""",

            "medical_knowledge_integrator": """You are a medical AI that makes the FINAL decision about outcome in the NEXT hospital visit.

You see:
- The target patient's analysis and initial prediction (Round 1).
- General mortality-risk patterns from fatal cases (positive agent).
- General survival-risk patterns from survivor cases (negative agent).

Your job:
1) Check if the Round 1 reasoning matches the general death and survival patterns.
2) Weigh both sides and decide if the initial prediction should stay the same or be changed.
3) Explain your decision in 2–4 short bullet points and 2–3 sentences.

CRITICAL FINAL STEP: You MUST end your response with exactly one of these lines:
\\boxed{0} - if your FINAL prediction is SURVIVAL (patient will NOT die in next visit)
\\boxed{1} - if your FINAL prediction is MORTALITY (patient WILL die in next visit)

Do not include anything after the \\boxed{} prediction."""
        }
    
    def _extract_prediction(self, response) -> Optional[int]:
        """
        Extract prediction from agent response.
        
        Args:
            response: Agent response (string or list format)
            
        Returns:
            Extracted prediction (0 or 1) or None if not found
        """
        # Ensure response is a string
        if isinstance(response, list):
            if len(response) > 0 and isinstance(response[0], dict):
                response = response[0].get('generated_text', '')
            else:
                response = str(response)
        elif not isinstance(response, str):
            response = str(response)
        
        # Look for \\boxed{0} or \\boxed{1} pattern - EXACTLY 0 or 1, not other characters
        boxed_pattern = r'\\boxed\{([01])\}'
        match = re.search(boxed_pattern, response)
        
        if match:
            return int(match.group(1))
        
        # Also try without double backslash (in case it's single backslash) - EXACTLY 0 or 1
        simple_pattern = r'boxed\{([01])\}'
        match = re.search(simple_pattern, response)
        
        if match:
            return int(match.group(1))
        
        # No fallback guessing - return None if no explicit boxed prediction found
        return None
    
    def _agent_turn(self, 
                   role: str, 
                   patient_context: str, 
                   similar_patients: Dict[str, str], 
                   medical_knowledge: str = "",
                   debate_history: List[Dict[str, Any]] = None,
                   logger = None) -> Dict[str, Any]:
        """
        Execute a single agent turn.
        
        Args:
            role: Agent role identifier
            patient_context: Target patient's EHR context
            similar_patients: Similar patient contexts
            medical_knowledge: Retrieved medical knowledge (optional)
            debate_history: Previous debate messages
            
        Returns:
            Agent response dictionary
        """
        print(f"\n--- {role.upper()} TURN ---")
        
        # Get system prompt for this role
        system_prompt = self.agent_prompts.get(role, self.agent_prompts["target_patient_analyst"])
        
        # Format debate history for context (only for Round 1 and Round 3 agents)
        history_text = ""
        if debate_history and role not in ["positive_similar_comparator", "negative_similar_comparator"]:
            history_text = "\n## Previous Analysis:\n"
            for entry in debate_history[-4:]:  # Last 4 entries for context
                agent_role = entry.get('role', 'Unknown')
                message = entry.get('message', '')
                prediction = entry.get('prediction', 'Unknown')
                history_text += f"{agent_role}: {message} [Prediction: {prediction}]\n"
            history_text += "\n"
        
        # Debug: Check similar patient data
        print(f"\n--- DEBUG: Data for {role.upper()} ---")
        print(f"Target context length: {len(patient_context) if patient_context else 0}")
        print(f"Similar patients keys: {list(similar_patients.keys()) if similar_patients else 'None'}")
        if similar_patients:
            print(f"Positive similars length: {len(similar_patients.get('positive', ''))}")
            print(f"Negative similars length: {len(similar_patients.get('negative', ''))}")
            print(f"Length of history text: {len(history_text)}")
            print(f"Tail of history text: {history_text[-200:]}...")
        print(f"--- END DEBUG ---\n")
        
        # Build context based on agent role
        if role == "target_patient_analyst":
            primary_context = f"## Target Patient EHR Context ##\n{patient_context}"
            secondary_context = ""
        elif role == "positive_similar_comparator":
            primary_context = f"## Positive Similar Patients (Mortality = 1) ##\n{similar_patients.get('positive', 'No positive similar patients available.')}"
            # Let agent validate Round 1 analysis instead of re-analyzing raw target context
            secondary_context = ""
        elif role == "negative_similar_comparator":
            primary_context = f"## Negative Similar Patients (Mortality = 0) ##\n{similar_patients.get('negative', 'No negative similar patients available.')}"
            # Let agent validate Round 1 analysis instead of re-analyzing raw target context
            secondary_context = ""
        else:  # medical_knowledge_integrator
            primary_context = f"## Target Patient EHR Context ##\n{patient_context}"
            secondary_context = f"\n## Positive Similar Patients ##\n{similar_patients.get('positive', 'None')}"
            secondary_context += f"\n## Negative Similar Patients ##\n{similar_patients.get('negative', 'None')}"
            if medical_knowledge:
                secondary_context += f"\n## Medical Knowledge ##\n{medical_knowledge}"
        
        # Build full prompt
        prompt = f"""{system_prompt}

{primary_context}{secondary_context}

{history_text}

Provide your clinical analysis and mortality risk assessment:"""
        
        # Generate response
        try:
            start_time = time.time()
            
            # Use different temperatures for each agent to promote diversity
            agent_temps = {
                "target_patient_analyst": 0.7,
                "positive_similar_comparator": 0.3,
                "negative_similar_comparator": 0.3,
                "medical_knowledge_integrator": 0.5
            }
            temperature = agent_temps.get(role, 0.5)
            
            # Set token limits based on debate round and role
            if role == "target_patient_analyst":
                max_tokens = 8192  # Round 1: Comprehensive target analysis with full context
            elif role in ["positive_similar_comparator", "negative_similar_comparator"]:
                max_tokens = 8192  # Round 2: Detailed comparison analysis with full similar patient context
            else:  # medical_knowledge_integrator
                max_tokens = 32768  # Round 3: Maximum available tokens for comprehensive integration
            
            # Select appropriate model based on agent role
            if role == "medical_knowledge_integrator":
                # Use integrator model (potentially larger 70B model)
                selected_llm = self.integrator_llm
                print(f"Using integrator model: {self.integrator_model_name}")
            else:
                # Use main model for other agents
                selected_llm = self.llm
                
            response = selected_llm(
                prompt,
                max_tokens=max_tokens,  # Use max_tokens for new tokens to generate
                temperature=temperature,
                top_p=0.9,
                return_format='string',  # Ensure we get a string response
                stop_sequences=["<|im_end|>", "</s>"],  # Remove boxed stop sequences to allow completion
                enable_think=True  # Enable thinking mode for better reasoning
            )
            
            generation_time = time.time() - start_time
            
            # Log raw response for debugging
            log_message = f"\n{'='*50}\nRAW RESPONSE from {role.upper()}\n{'='*50}\nResponse type: {type(response)}\nResponse length: {len(response) if response else 0}\nFull response: {response}\n{'='*50}"
            if logger:
                logger.info(log_message)
            
            # Extract prediction
            prediction = self._extract_prediction(response)
            if logger:
                logger.info(f"EXTRACTED PREDICTION: {prediction}")
            
            return {
                'role': role,
                'message': response,
                'prediction': prediction if role not in ["positive_similar_comparator", "negative_similar_comparator"] else None,
                'generation_time': generation_time,
                'prompt_length': len(prompt),
                'response_length': len(response)
            }
            
        except Exception as e:
            print(f"Error in {role} turn: {e}")
            return {
                'role': role,
                'message': f"Error occurred: {str(e)}",
                'prediction': None,
                'generation_time': 0,
                'error': str(e)
            }
    

    
    def debate_mortality_prediction(self, 
                                  patient_context: str, 
                                  positive_similars: str,
                                  negative_similars: str,
                                  medical_knowledge: str = "",
                                  patient_id: str = "unknown",
                                  model_name: str = None,
                                  output_dir: str = None,
                                  ground_truth: int = None) -> Dict[str, Any]:
        """
        Conduct structured three-round multi-agent debate for mortality prediction.
        
        Round 1: Target patient analysis only
        Round 2: Similar patient comparisons (positive and negative)
        Round 3: Medical knowledge integration and final consensus
        
        Args:
            patient_context: Target patient's EHR context
            positive_similars: Positive similar patient contexts (mortality=1)
            negative_similars: Negative similar patient contexts (mortality=0)
            medical_knowledge: Retrieved medical knowledge (optional)
            ground_truth: Ground truth label (0=survival, 1=mortality), used for fallback when prediction fails
            
        Returns:
            Debate results dictionary
        """
        print(f"\n{'='*80}")
        print("STARTING STRUCTURED MORTALITY PREDICTION DEBATE")
        print(f"{'='*80}")
        
        # Setup patient-specific logging with structured directory
        import logging
        from pathlib import Path
        
        # Use provided model name or default from instance
        if model_name is None:
            model_name = getattr(self, 'model_name', 'Qwen/Qwen3-4B-Instruct-2507')
        
        # Clean model name for directory (replace / with _)
        clean_model_name = model_name.replace('/', '_').replace('-', '_')
        
        # Create structured log directory with absolute path
        # Use absolute paths to ensure consistency regardless of working directory
        if output_dir:
            # If output_dir is provided, create logs in the same parent directory
            output_path = Path(output_dir)
            if output_path.is_file() or output_path.suffix == '.json':
                # output_dir is actually a file path, use its parent directory
                log_dir = output_path.parent / "debate_logs"
            else:
                # output_dir is a directory path
                log_dir = output_path / "debate_logs"
        else:
            # Default to results directory - use absolute path based on script location
            script_dir = Path(__file__).parent.resolve()  # Get absolute path
            log_dir = script_dir / "results" / f"cot_mor_{clean_model_name}" / "debate_logs"
        
        # Ensure the log directory is created with proper permissions
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            # Test write permissions by creating a temporary file
            test_file = log_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot create log directory at {log_dir}: {e}")
            # Fallback to a temporary directory in /tmp
            import tempfile
            log_dir = Path(tempfile.mkdtemp(prefix=f"debate_logs_{clean_model_name}_"))
            print(f"Using fallback log directory: {log_dir}")
        
        print(f"COT Debug: Output dir parameter: {output_dir}")
        print(f"COT Debug: Creating log directory at: {log_dir.absolute()}")
        
        # Create patient-specific log file
        log_filename = log_dir / f"debate_responses_{patient_id}.log"
        logger = logging.getLogger(f'debate_{patient_id}_{clean_model_name}')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create file handler with patient-specific filename
        try:
            file_handler = logging.FileHandler(log_filename, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Starting debate for patient {patient_id}")
            print(f"COT Debug: Successfully created log file: {log_filename}")
        except Exception as e:
            print(f"Warning: Failed to create log file {log_filename}: {e}")
            # Create a console handler as fallback
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.info(f"Starting debate for patient {patient_id} (console logging only)")
        
        debate_history = []
        similar_patients_dict = {
            'positive': positive_similars,
            'negative': negative_similars
        }
        
        # Round 1: Target Patient Analysis Only
        print(f"\n--- ROUND 1: TARGET PATIENT ANALYSIS ---")
        target_response = self._agent_turn(
            role="target_patient_analyst",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=[],
            logger=logger
        )
        debate_history.append(target_response)
        print(f"Target Analysis: {target_response.get('message', 'No response')[:200]}...")
        print(f"Initial Prediction: {target_response.get('prediction')}")
        
        # Round 2: Similar Patient Comparisons
        print(f"\n--- ROUND 2: SIMILAR PATIENT COMPARISONS ---")
        
        # Positive similar patient comparator
        positive_response = self._agent_turn(
            role="positive_similar_comparator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger
        )
        debate_history.append(positive_response)
        print(f"Positive Comparator: {positive_response.get('message', 'No response')[:200]}...")
        
        # Negative similar patient comparator  
        negative_response = self._agent_turn(
            role="negative_similar_comparator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger
        )
        debate_history.append(negative_response)
        print(f"Negative Comparator: {negative_response.get('message', 'No response')[:200]}...")
        
        # Round 3: Integration and Final Consensus
        print(f"\n--- ROUND 3: INTEGRATION AND CONSENSUS ---")
        integrator_response = self._agent_turn(
            role="medical_knowledge_integrator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger
        )
        debate_history.append(integrator_response)
        print(f"Integration: {integrator_response.get('message', 'No response')[:200]}...")
        print(f"Final Prediction: {integrator_response.get('prediction')}")
        
        # Use integrator's prediction as the final result (no fallback to consensus)
        final_prediction = integrator_response.get('prediction')
        
        # Only fallback to target if integrator completely fails
        if final_prediction is None:
            print("Warning: No prediction from final integrator, using target prediction as fallback")
            final_prediction = target_response.get('prediction')
            if final_prediction is None:
                # Predict opposite of ground truth as final fallback
                if ground_truth is not None:
                    final_prediction = 1 - ground_truth  # Opposite of ground truth
                    print(f"[FALLBACK] No prediction from any agent, predicting opposite of ground truth: {final_prediction} (GT={ground_truth})")
                    logger.warning(f"Final fallback - no predictions, predicting opposite of ground_truth={ground_truth}, prediction={final_prediction}")
                else:
                    print("Warning: No prediction from any agent and no ground truth, final answer is None")
                    final_prediction = None
        
        print(f"\n{'='*80}")
        print(f"DEBATE COMPLETED - Final Prediction: {final_prediction}")
        print(f"{'='*80}")
        
        # Log final result and close logging
        logger.info(f"DEBATE COMPLETED - Final Prediction: {final_prediction}")
        logger.info(f"Debate history: {len(debate_history)} rounds completed")
        
        # Clean up handlers - check if file_handler exists
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
                break
        
        return {
            'final_prediction': final_prediction,
            'debate_history': debate_history,
            'rounds_completed': 3,  # Always 3 rounds in structured flow
            'total_generation_time': sum(r.get('generation_time', 0) for r in debate_history),
            'integrator_prediction': integrator_response.get('prediction'),
            'target_prediction': target_response.get('prediction')
        }

# Test the debate system
if __name__ == "__main__":
    try:
        # Initialize debate system
        debate_system = MortalityDebateSystem()
        
        # Test with sample data
        test_patient_context = """
Patient ID: TEST001
Visit ID: TEST_VISIT_001

## Medical Conditions ##
1. Acute myocardial infarction
2. Severe heart failure
3. Septicemia
4. Acute kidney injury

## Medical Procedures ##
1. Mechanical ventilation
2. Hemodialysis
3. Central venous catheter insertion

## Medications ##
1. Norepinephrine
2. Furosemide
3. Heparin
4. Morphine sulfate
"""
        
        test_similar_patients = """
Similar Patient 1 (Mortality=1): Advanced cancer patient with sepsis, required mechanical ventilation and vasopressors, died within 48 hours.
Similar Patient 2 (Survival=0): Heart failure patient with acute exacerbation, responded well to diuretics and supportive care, discharged home.
"""
        
        test_medical_knowledge = """
Acute myocardial infarction combined with septicemia carries extremely high mortality risk. 
Mechanical ventilation in the setting of multiple organ failure is associated with poor prognosis.
"""
        
        # Run debate
        test_negative_similars = "Similar Patient (Survival=0): Heart failure patient with acute exacerbation, responded well to diuretics and supportive care, discharged home."
        result = debate_system.debate_mortality_prediction(
            patient_context=test_patient_context,
            positive_similars=test_similar_patients,
            negative_similars=test_negative_similars,
            medical_knowledge=test_medical_knowledge,
            patient_id="TEST001",
            model_name="Qwen/Qwen3-4B-Instruct-2507"
        )
        
        print(f"\\n{'='*80}")
        print("DEBATE RESULTS")
        print(f"{'='*80}")
        print(f"Final Prediction: {result['final_prediction']}")
        print(f"Rounds Completed: {result['rounds_completed']}")
        print(f"Total Generation Time: {result['total_generation_time']:.2f}s")
        
    except Exception as e:
        print(f"Error testing debate system: {e}")
        import traceback
        traceback.print_exc()