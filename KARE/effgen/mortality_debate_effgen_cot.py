#!/usr/bin/env python3
"""
Enhanced Multi-Agent Debate System for KARE Mortality Prediction using effGen
CoT Mode (Chain-of-Thought without retrieval)
"""

import os
import sys
import json
import re
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
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

# Import preprocessing utilities
try:
    from kare_contrastive_preprocessing import (
        preprocess_for_debate,
        format_integrator_history_with_labels
    )
except ImportError:
    print("Warning: kare_contrastive_preprocessing not found, using fallback")
    preprocess_for_debate = None
    format_integrator_history_with_labels = None


class MortalityDebateSystemEffGen:
    """
    Enhanced debate system using effGen framework for KARE mortality prediction.
    CoT Mode: Pure chain-of-thought reasoning without external retrieval.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 gpu_ids: str = "0",
                 model_cache_dir: str = "/data/wang/junh/.cache/huggingface"):
        """
        Initialize the mortality debate system with effGen.
        
        Args:
            model_name: HuggingFace model name or path
            gpu_ids: GPU IDs to use (comma-separated)
            model_cache_dir: Cache directory for models
        """
        print(f"[INIT] Initializing effGen Debate System (CoT Mode)")
        print(f"[INIT] Model: {model_name}")
        print(f"[INIT] GPU IDs: {gpu_ids}")
        
        # Store configurations
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.model_cache_dir = model_cache_dir
        
        # Set CUDA_VISIBLE_DEVICES once (following MEDRAG_GPU_SETUP_FIX.md)
        print(f"[CUDA] Setting CUDA_VISIBLE_DEVICES={gpu_ids}")
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        
        # Load model using effgen (full precision, no quantization)
        print(f"[MODEL] Loading model from effgen...")
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
            traceback.print_exc()
            raise
        
        # Agent configuration
        self.agent_roles = [
            "mortality_risk_assessor", 
            "protective_factor_analyst",
            "balanced_clinical_integrator"
        ]
        self.max_rounds = 2
        
        # System prompts for each agent (identical to VLLM version)
        self.agent_prompts = self._initialize_agent_prompts()
        
        # Create agent instances
        self._initialize_agents()
        
    def _initialize_agent_prompts(self) -> Dict[str, str]:
        """Initialize specialized prompts for each agent role."""
        
        return {
            "mortality_risk_assessor": """You are a medical AI that analyzes clinical patterns between patients.

Task:
Given (1) Target patient and (2) One Similar patient, produce a contrastive comparison that is grounded in the provided codes.

**CLINICAL PATTERN ANALYSIS:**

1. **Shared Clinical Features:**
   - What conditions, procedures, and medications appear in BOTH patients?
   - What is the clinical significance of these commonalities?

2. **Similar-Specific Features:**
   - What is unique to the similar patient?
   - What does this tell us about different clinical paths?

**TEMPORAL PROGRESSION:**
Analyze how shared and unique patterns evolve across visits.

**IMPORTANT:** Do NOT speculate about outcomes or mortality. Focus solely on clinical pattern analysis.""",

            "protective_factor_analyst": """You are a medical AI that analyzes clinical patterns between patients.

Task:
Given (1) Target patient and (2) One Similar patient, produce a contrastive comparison that is grounded in the provided codes.

**CLINICAL PATTERN ANALYSIS:**

1. **Shared Clinical Features:**
   - What conditions, procedures, and medications appear in BOTH patients?
   - What is the clinical significance of these commonalities?

2. **Similar-Specific Features:**
   - What is unique to the similar patient?
   - What does this tell us about different clinical paths?

**TEMPORAL PROGRESSION:**
Analyze how shared and unique patterns evolve across visits.

**IMPORTANT:** Do NOT speculate about outcomes or mortality. Focus solely on clinical pattern analysis.""",

            "balanced_clinical_integrator": """You are a medical AI Clinical Assistant analyzing mortality and survival probabilities for the NEXT hospital visit.

IMPORTANT: Mortality is rare - only predict mortality probability > 0.5 if evidence STRONGLY supports it. When uncertain, predict survival probability > 0.5.

Workflow:
1) Review the two clinical pattern analyses from previous agents
2) Identify 3-4 key factors that contribute to the target patient's next visit outcome
3) Analyze BOTH risky factors AND survival factors
4) Be conservative: mortality is rare, so strong evidence is needed for high mortality probability
5) Provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
        }
    
    def _initialize_agents(self):
        """Initialize effGen agents with proper configurations."""
        print(f"[AGENTS] Initializing 3 specialized agents...")
        
        self.agents = {}
        
        # Agent 1: Mortality Risk Assessor
        self.agents["mortality_risk_assessor"] = Agent(
            config=AgentConfig(
                name="mortality_risk_assessor",
                model=self.model,
                tools=[],  # No tools for CoT mode
                system_prompt=self.agent_prompts["mortality_risk_assessor"],
                max_iterations=1,  # Single-turn execution
                temperature=0.3,
            )
        )
        print(f"[AGENTS] Created mortality_risk_assessor")
        
        # Agent 2: Protective Factor Analyst
        self.agents["protective_factor_analyst"] = Agent(
            config=AgentConfig(
                name="protective_factor_analyst",
                model=self.model,
                tools=[],  # No tools for CoT mode
                system_prompt=self.agent_prompts["protective_factor_analyst"],
                max_iterations=1,  # Single-turn execution
                temperature=0.3,
            )
        )
        print(f"[AGENTS] Created protective_factor_analyst")
        
        # Agent 3: Balanced Clinical Integrator
        self.agents["balanced_clinical_integrator"] = Agent(
            config=AgentConfig(
                name="balanced_clinical_integrator",
                model=self.model,
                tools=[],  # No tools for CoT mode
                system_prompt=self.agent_prompts["balanced_clinical_integrator"],
                max_iterations=1,  # Single-turn execution
                temperature=0.5,
            )
        )
        print(f"[AGENTS] Created balanced_clinical_integrator")
        
        print(f"[AGENTS] All agents initialized successfully")
    
    def _extract_prediction_and_probabilities(self, response: str, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract prediction and probability scores from agent response.
        
        Args:
            response: Agent response text
            ground_truth: Ground truth label for fallback logic
            
        Returns:
            Dictionary with prediction, mortality_prob, survival_prob, is_fallback
        """
        result = {
            'prediction': None,
            'mortality_probability': None,
            'survival_probability': None,
            'confidence': None,
            'is_fallback': False
        }
        
        # Extract mortality probability
        mortality_patterns = [
            r'\*\*MORTALITY PROBABILITY:?\*\*[\s:]*([0-9]*\.?[0-9]+)',
            r'\*\*Mortality Probability:?\*\*[\s:]*([0-9]*\.?[0-9]+)',
            r'MORTALITY PROBABILITY:[\s]*([0-9]*\.?[0-9]+)',
            r'Mortality Probability:[\s]*([0-9]*\.?[0-9]+)',
            r'mortality probability[:\s]+([0-9]*\.?[0-9]+)',
        ]
        for pattern in mortality_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['mortality_probability'] = float(match.group(1))
                break
        
        # Extract survival probability
        survival_patterns = [
            r'\*\*SURVIVAL PROBABILITY:?\*\*[\s:]*([0-9]*\.?[0-9]+)',
            r'\*\*Survival Probability:?\*\*[\s:]*([0-9]*\.?[0-9]+)',
            r'SURVIVAL PROBABILITY:[\s]*([0-9]*\.?[0-9]+)',
            r'Survival Probability:[\s]*([0-9]*\.?[0-9]+)',
            r'survival probability[:\s]+([0-9]*\.?[0-9]+)',
        ]
        for pattern in survival_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['survival_probability'] = float(match.group(1))
                break
        
        # Determine prediction with VLLM-matching fallback logic
        mort_prob = result['mortality_probability']
        surv_prob = result['survival_probability']
        
        if mort_prob is not None and surv_prob is not None:
            # Both probabilities available - normal case
            result['prediction'] = 1 if mort_prob > surv_prob else 0
            result['is_fallback'] = False
            
        elif mort_prob is None and surv_prob is None:
            # Both None - fallback to opposite of ground truth
            if ground_truth is not None:
                result['prediction'] = 1 - ground_truth
            else:
                result['prediction'] = 0  # Default fallback
            result['is_fallback'] = True
                
        elif mort_prob is not None and surv_prob is None:
            # Only mortality prob available
            if mort_prob > 0.5:
                result['prediction'] = 1
                result['is_fallback'] = False
            else:
                # mort_prob <= 0.5, fallback to opposite of ground truth
                if ground_truth is not None:
                    result['prediction'] = 1 - ground_truth
                else:
                    result['prediction'] = 0
                result['is_fallback'] = True
                    
        else:  # surv_prob is not None and mort_prob is None
            # Only survival prob available
            if surv_prob > 0.5:
                result['prediction'] = 0
                result['is_fallback'] = False
            else:
                # surv_prob <= 0.5, fallback to opposite of ground truth
                if ground_truth is not None:
                    result['prediction'] = 1 - ground_truth
                else:
                    result['prediction'] = 1
                result['is_fallback'] = True
        
        return result
    
    def _prepare_integrator_history(self, debate_history: List[Dict[str, Any]]) -> str:
        """
        Prepare history for integrator.
        
        Args:
            debate_history: List of previous debate responses
            
        Returns:
            Combined history text
        """
        combined_history = "\n## Previous Analysis:\n"
        
        for entry in debate_history:
            role = entry.get('role', 'unknown')
            message = entry.get('message', '')
            
            # Add labels for integrator
            if 'risk' in role.lower() or 'mortality' in role.lower():
                label_header = "### Similar Case with Mortality=1 (positive class) Analysis:\n"
            elif 'protective' in role.lower():
                label_header = "### Similar Case with Survival=0 (negative class) Analysis:\n"
            else:
                label_header = f"### {role.upper().replace('_', ' ')}:\n"
            
            combined_history += f"\n{label_header}{message}\n"
        
        combined_history += "\n**Note:** The above analyses were conducted without knowledge of outcomes. Use these pattern comparisons to inform your assessment.\n"
        
        return combined_history
    
    def _agent_turn(self,
                   role: str,
                   patient_context: str,
                   similar_patients: Dict[str, str],
                   debate_history: List[Dict[str, Any]] = None,
                   logger = None,
                   ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a single agent turn using effGen.
        
        Args:
            role: Agent role identifier
            patient_context: Target patient's EHR context
            similar_patients: Similar patient contexts
            debate_history: Previous debate messages
            logger: Logger instance
            
        Returns:
            Agent response dictionary
        """
        print(f"\n--- {role.upper()} TURN ---")
        
        # Build prompt based on role
        if role == "mortality_risk_assessor":
            context = f"## Target Patient ##\n{patient_context}\n\n## Similar Patient ##\n{similar_patients.get('positive', 'No similar patient available.')}"
        elif role == "protective_factor_analyst":
            context = f"## Target Patient ##\n{patient_context}\n\n## Similar Patient ##\n{similar_patients.get('negative', 'No similar patient available.')}"
        elif role == "balanced_clinical_integrator":
            context = f"## Target Patient EHR Context ##\n{patient_context}"
            if debate_history:
                context += "\n" + self._prepare_integrator_history(debate_history)
        else:
            context = patient_context
        
        # Add task instruction
        prompt = f"{context}\n\nProvide your clinical analysis and mortality risk assessment:"
        
        # Execute agent
        try:
            start_time = time.time()
            
            # Get agent
            agent = self.agents.get(role)
            if not agent:
                raise ValueError(f"Agent {role} not found")
            
            # Run agent with effgen
            print(f"[{role}] Running agent with effgen...")
            # Import AgentMode to explicitly specify single agent execution
            from effgen.core.agent import AgentMode
            
            # Set agent-specific max_tokens (match original vllm)
            agent_max_tokens = {
                "mortality_risk_assessor": 2048,
                "protective_factor_analyst": 2048,
                "balanced_clinical_integrator": 4096
            }
            max_tokens = agent_max_tokens.get(role, 2048)
            
            result = agent.run(
                prompt,
                mode=AgentMode.SINGLE,
                top_p=0.9,  # Match original vllm
                max_tokens=max_tokens,
                repetition_penalty=1.2,  # Match original vllm
                stop_sequences=["<|im_end|>", "</s>"]  # Match original vllm
            )
            
            # Extract response text
            if hasattr(result, 'output'):
                response = result.output
            elif isinstance(result, dict):
                response = result.get('output', str(result))
            else:
                response = str(result)
            
            generation_time = time.time() - start_time
            
            # Log raw response
            if logger:
                log_message = f"\n{'='*50}\nRAW RESPONSE from {role.upper()}\n{'='*50}\nResponse length: {len(response)}\nFull response: {response}\n{'='*50}"
                logger.info(log_message)
            
            # Extract prediction and probabilities
            extraction_result = self._extract_prediction_and_probabilities(response, ground_truth=ground_truth)
            
            if logger:
                logger.info(f"EXTRACTED PREDICTION: {extraction_result['prediction']}")
                logger.info(f"MORTALITY PROBABILITY: {extraction_result['mortality_probability']}")
                logger.info(f"SURVIVAL PROBABILITY: {extraction_result['survival_probability']}")
                logger.info(f"IS FALLBACK: {extraction_result['is_fallback']}")
            
            return {
                'role': role,
                'message': response,
                'prediction': extraction_result['prediction'] if role == "balanced_clinical_integrator" else None,
                'mortality_probability': extraction_result.get('mortality_probability'),
                'survival_probability': extraction_result.get('survival_probability'),
                'confidence': extraction_result.get('confidence'),
                'generation_time': generation_time,
                'prompt_length': len(prompt),
                'response_length': len(response)
            }
            
        except Exception as e:
            print(f"[ERROR] Error in {role} turn: {e}")
            traceback.print_exc()
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
        Conduct structured multi-agent debate for mortality prediction.
        
        Args:
            patient_context: Target patient's EHR context
            positive_similars: Positive similar patient contexts (mortality=1)
            negative_similars: Negative similar patient contexts (mortality=0)
            medical_knowledge: Retrieved medical knowledge (optional, unused in CoT)
            patient_id: Patient identifier
            model_name: Model name (for logging)
            output_dir: Output directory path
            ground_truth: Ground truth label
            
        Returns:
            Debate results dictionary
        """
        print(f"\n{'='*80}")
        print("STARTING MORTALITY PREDICTION DEBATE (effGen CoT Mode)")
        print(f"{'='*80}")
        
        # Setup logging
        if model_name is None:
            model_name = self.model_name
        
        clean_model_name = model_name.replace('/', '_').replace('-', '_')
        
        # Create log directory
        if output_dir:
            output_path = Path(output_dir)
            if output_path.is_file() or output_path.suffix in ['.json', '.log', '.txt']:
                log_dir = output_path.parent / "logs"
            else:
                log_dir = output_path / "logs"
        else:
            script_dir = Path(__file__).parent
            log_dir = script_dir / "results" / f"effgen_cot_{clean_model_name}" / "logs"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create patient-specific log file
        log_filename = log_dir / f"debate_responses_{patient_id}.log"
        logger = logging.getLogger(f'debate_effgen_{patient_id}')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        try:
            file_handler = logging.FileHandler(log_filename, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Starting debate for patient {patient_id}")
        except Exception as e:
            print(f"Warning: Failed to create log file: {e}")
        
        # Initialize debate
        debate_history = []
        similar_patients_dict = {
            'positive': positive_similars,
            'negative': negative_similars
        }
        
        # Round 1: Similar Patient Comparisons
        print(f"\n--- ROUND 1: SIMILAR PATIENT COMPARISONS ---")
        
        # Agent 1: Mortality Risk Assessor
        risk_response = self._agent_turn(
            role="mortality_risk_assessor",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            debate_history=[],
            logger=logger,
            ground_truth=ground_truth
        )
        debate_history.append(risk_response)
        print(f"Risk Assessor: {risk_response.get('message', '')[:200]}...")
        
        # Agent 2: Protective Factor Analyst
        protective_response = self._agent_turn(
            role="protective_factor_analyst",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            debate_history=debate_history,
            logger=logger,
            ground_truth=ground_truth
        )
        debate_history.append(protective_response)
        print(f"Protective Analyst: {protective_response.get('message', '')[:200]}...")
        
        # Round 2: Integration and Consensus
        print(f"\n--- ROUND 2: INTEGRATION AND CONSENSUS ---")
        integrator_response = self._agent_turn(
            role="balanced_clinical_integrator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            debate_history=debate_history,
            logger=logger,
            ground_truth=ground_truth
        )
        debate_history.append(integrator_response)
        print(f"Integrator: {integrator_response.get('message', '')[:200]}...")
        print(f"Final Prediction: {integrator_response.get('prediction')}")
        
        # Extract final results
        final_prediction = integrator_response.get('prediction')
        final_mortality_prob = integrator_response.get('mortality_probability')
        final_survival_prob = integrator_response.get('survival_probability')
        final_confidence = integrator_response.get('confidence')
        
        # Fallback if no probabilities
        if final_mortality_prob is None and final_survival_prob is None:
            print("[WARNING] No probabilities extracted from integrator")
            if ground_truth is not None:
                final_prediction = 1 - ground_truth  # Conservative fallback
        
        print(f"\n{'='*80}")
        print(f"DEBATE COMPLETED - Final Prediction: {final_prediction}")
        if final_mortality_prob is not None:
            print(f"Final Mortality Probability: {final_mortality_prob}")
        if final_survival_prob is not None:
            print(f"Final Survival Probability: {final_survival_prob}")
        print(f"{'='*80}")
        
        # Log final result
        logger.info(f"DEBATE COMPLETED - Final Prediction: {final_prediction}")
        if final_mortality_prob is not None:
            logger.info(f"Final Mortality Probability: {final_mortality_prob}")
        if final_survival_prob is not None:
            logger.info(f"Final Survival Probability: {final_survival_prob}")
        
        # Clean up logger
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)
        
        return {
            'final_prediction': final_prediction,
            'final_mortality_probability': final_mortality_prob,
            'final_survival_probability': final_survival_prob,
            'final_confidence': final_confidence,
            'debate_history': debate_history,
            'rounds_completed': 2,
            'total_generation_time': sum(r.get('generation_time', 0) for r in debate_history),
            'integrator_prediction': integrator_response.get('prediction'),
            'integrator_mortality_probability': integrator_response.get('mortality_probability'),
            'integrator_survival_probability': integrator_response.get('survival_probability'),
            'integrator_confidence': integrator_response.get('confidence')
        }


# Test the debate system
if __name__ == "__main__":
    print("Testing effGen CoT Debate System...")
    
    try:
        # Initialize debate system
        debate_system = MortalityDebateSystemEffGen(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            gpu_ids="0"
        )
        
        # Test with sample data
        test_patient_context = """
Patient ID: TEST001

Visit 0:
Conditions:
1. Acute myocardial infarction
2. Severe heart failure
3. Septicemia

Procedures:
1. Mechanical ventilation
2. Central venous catheter insertion

Medications:
1. Norepinephrine
2. Furosemide
3. Heparin
"""
        
        test_positive = "Similar Patient (mortality=1): Advanced heart disease, sepsis, died."
        test_negative = "Similar Patient (survival=0): Heart failure, recovered with treatment."
        
        # Run debate
        result = debate_system.debate_mortality_prediction(
            patient_context=test_patient_context,
            positive_similars=test_positive,
            negative_similars=test_negative,
            patient_id="TEST001"
        )
        
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        print(f"Final Prediction: {result['final_prediction']}")
        print(f"Mortality Probability: {result.get('final_mortality_probability')}")
        print(f"Survival Probability: {result.get('final_survival_probability')}")
        print(f"Total Time: {result['total_generation_time']:.2f}s")
        
    except Exception as e:
        print(f"Error testing debate system: {e}")
        traceback.print_exc()
