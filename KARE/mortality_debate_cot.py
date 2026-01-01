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
import logging
import traceback
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from openai import OpenAI
    FIREWORKS_AVAILABLE = True
except ImportError:
    FIREWORKS_AVAILABLE = False
    print("Warning: OpenAI library not available. Fireworks API support disabled.")

# Use VLLM directly for COT mode (no MedRAG wrapper needed)
from vllm import LLM, SamplingParams

class MortalityDebateSystem:
    """
    Enhanced debate system specifically designed for KARE mortality prediction task.
    Uses three specialized agents with distinct analytical perspectives.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507", 
                 gpu_ids: str = "6,7",
                 integrator_model_name: str = None,
                 integrator_gpu: str = None,
                 use_fireworks: bool = False,
                 fireworks_api_key: str = None):
        """
        Initialize the mortality debate system.
        
        Args:
            model_name: HuggingFace model name for VLLM (agents 1-3) or Fireworks model name if use_fireworks=True
            gpu_ids: GPU IDs to use for main model (comma-separated)
            integrator_model_name: Model name for integrator agent. If None, uses model_name
            integrator_gpu: GPU ID for integrator model. If None, uses second GPU from gpu_ids
            use_fireworks: If True, use Fireworks API for agents 1-3 instead of VLLM
            fireworks_api_key: Fireworks API key. If None, reads from FIREWORKS_API_KEY env var
        """
        # Store model configurations
        self.model_name = model_name
        self.integrator_model_name = integrator_model_name or model_name
        self.gpu_ids = gpu_ids
        self.use_fireworks = use_fireworks
        
        # Check environment before any changes
        print(f"[PRE-INIT] Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        print(f"[PRE-INIT] Current working directory: {os.getcwd()}")
        print(f"[PRE-INIT] Python path: {sys.executable}")
        
        # Check for any CUDA/GPU related env vars
        cuda_env_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k or 'GPU' in k}
        if cuda_env_vars:
            print(f"[PRE-INIT] Existing CUDA/GPU env vars: {cuda_env_vars}")
        
        # Check GPU processes
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print(f"[PRE-INIT] GPU processes running:")
                print(result.stdout)
            else:
                print("[PRE-INIT] No GPU processes detected")
        except Exception as e:
            print(f"[PRE-INIT] Could not check GPU processes: {e}")
        
        # Set CUDA_VISIBLE_DEVICES to all GPUs (same as single agent code)
        print(f"[STEP 1] Setting CUDA_VISIBLE_DEVICES to: {gpu_ids}")
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"[STEP 2] CUDA_VISIBLE_DEVICES set successfully: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # Count available GPUs for tensor parallelism
        gpu_list = gpu_ids.split(',')
        num_gpus = len(gpu_list)
        print(f"[STEP 3] Parsed GPU list: {gpu_list}, count: {num_gpus}")
        
        # Import torch to check GPU availability
        try:
            import torch
            print(f"[STEP 4] Torch CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[STEP 5] Torch visible device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"[STEP 5.{i}] GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"[STEP 5.{i}] GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
                    # Check memory usage on each GPU
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    print(f"[STEP 5.{i}] GPU {i} allocated: {allocated:.4f} GB, reserved: {reserved:.4f} GB")
                    # Reset memory stats to get clean state
                    torch.cuda.reset_peak_memory_stats(i)
                    torch.cuda.empty_cache()
                    print(f"[STEP 5.{i}] GPU {i} memory cleared")
            else:
                print("[ERROR] CUDA not available - cannot proceed!")
                raise RuntimeError("CUDA not available")
        except Exception as e:
            print(f"[ERROR] Failed to query torch CUDA: {e}")
            raise
        
        print(f"[STEP 6] Will use tensor_parallel_size={num_gpus}")
        
        # Initialize main model (Fireworks API or VLLM) for all agents
        if use_fireworks:
            if not FIREWORKS_AVAILABLE:
                raise ImportError("OpenAI library required for Fireworks API. Install with: pip install openai")
            
            # Get API key from parameter or environment
            api_key = fireworks_api_key or os.environ.get('FIREWORKS_API_KEY')
            if not api_key:
                raise ValueError("Fireworks API key required. Set FIREWORKS_API_KEY env var or pass fireworks_api_key parameter")
            
            print(f"Main model ({model_name}) will use Fireworks API")
            self.fireworks_client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=api_key
            )
            self.llm = None  # Not using VLLM for main agents
        else:
            # Initialize VLLM directly (like single agent COT) with tensor parallelism across all GPUs
            print(f"[STEP 7] Initializing VLLM LLM for {model_name} with tensor_parallel_size={num_gpus}")
            gpu_util = 0.85 if num_gpus == 1 else 0.6  # Higher for single GPU, lower for TP
            print(f"[STEP 8] GPU memory utilization: {gpu_util}")
            
            print(f"[STEP 9] Creating LLM instance with parameters:")
            print(f"  - model: {model_name}")
            print(f"  - tensor_parallel_size: {num_gpus}")
            print(f"  - trust_remote_code: True")
            print(f"  - gpu_memory_utilization: {gpu_util}")
            print(f"  - enforce_eager: True")
            
            # Check VLLM version and environment
            import vllm
            print(f"[STEP 9.1] VLLM version: {vllm.__version__}")
            
            # Check for distributed environment variables
            dist_env_vars = {k: v for k, v in os.environ.items() if any(x in k for x in ['RANK', 'WORLD', 'MASTER', 'LOCAL_RANK', 'NCCL', 'GLOO'])}
            if dist_env_vars:
                print(f"[STEP 9.2] Distributed env vars found: {dist_env_vars}")
            else:
                print("[STEP 9.2] No distributed env vars detected")
            
            try:
                print(f"[STEP 10] Calling LLM() constructor...")
                self.llm = LLM(
                    model=model_name,
                    tensor_parallel_size=num_gpus,
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_util,
                    enforce_eager=True
                )
                print(f"[STEP 11] LLM initialized successfully!")
            except Exception as e:
                print(f"[FATAL ERROR] LLM initialization failed: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            print(f"[STEP 12] VLLM initialized successfully with TP={num_gpus}, gpu_util={gpu_util}")
            self.fireworks_client = None
        
        # All agents share same model instance
        print("All agents (including integrator) share the same model instance")
        self.integrator_llm = self.llm
        
        # Agent configuration - Three specialized agents (aligned with RAG mode)
        self.agent_roles = [
            "mortality_risk_assessor", 
            "protective_factor_analyst",
            "balanced_clinical_integrator"
        ]
        self.max_rounds = 2
        
        # System prompts for each agent
        self.agent_prompts = self._initialize_agent_prompts()
        
    def _call_fireworks_api(self, prompt: str, max_tokens: int = 8192, temperature: float = 0.7) -> str:
        """
        Call Fireworks API for text generation.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            response = self.fireworks_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Fireworks API: {e}")
            raise
    
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
    
    def _extract_prediction_and_probabilities(self, response) -> Dict[str, Any]:
        """
        Extract prediction and probability scores from agent response.
        
        Args:
            response: Agent response (string or list format)
            
        Returns:
            Dictionary with prediction, mortality_prob, survival_prob
        """
        # Ensure response is a string
        if isinstance(response, list):
            if len(response) > 0 and isinstance(response[0], dict):
                response = response[0].get('generated_text', '')
            else:
                response = str(response)
        elif not isinstance(response, str):
            response = str(response)
        
        result = {
            'prediction': None,
            'mortality_probability': None,
            'survival_probability': None,
            'confidence': None
        }
        
        # Extract mortality probability - comprehensive patterns
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
        
        # Extract survival probability - comprehensive patterns
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
        
        # Extract confidence level
        confidence_patterns = [
            r'Confidence Level:\s*(Very High|High|Moderate|Low)',
            r'confidence\s*:?\s*(Very High|High|Moderate|Low)',
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['confidence'] = match.group(1)
                break
        
        # Store ground_truth from function parameter (passed through)
        ground_truth = result.get('ground_truth', None)
        
        # Determine prediction with improved fallback logic
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
                # mort_prob < 0.5, fallback to opposite of ground truth
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
                # surv_prob < 0.5, fallback to opposite of ground truth
                if ground_truth is not None:
                    result['prediction'] = 1 - ground_truth
                else:
                    result['prediction'] = 1
                result['is_fallback'] = True
        
        return result
    
    def _summarize_round_response(self, round_text: str, round_name: str, target_tokens: int = 4000) -> str:
        """
        Summarize individual round response if it exceeds token limits.
        
        Args:
            round_text: Single round response text
            round_name: Name of the round (e.g., "risk_assessment", "protective_analysis")
            target_tokens: Target number of tokens for summary (default 4000 per round)
            
        Returns:
            Summarized round text or original if under limit
        """
        # Rough estimate: 1 token ≈ 4 chars, so 6000 tokens ≈ 24000 chars
        token_limit_chars = 6000 * 4  # 24000 chars
        
        if len(round_text) <= token_limit_chars:
            return round_text
            
        print(f"[ROUND] Summarizing {round_name} from {len(round_text)} chars to ~{target_tokens} tokens")
        
        # Create round-specific summary prompt
        if "risk" in round_name.lower() or "mortality" in round_name.lower():
            focus_areas = """- Key mortality risk factors
- Critical medical conditions
- High-risk procedures and interventions
- Warning signs and complications"""
        else:  # protective/survival analysis
            focus_areas = """- Key protective factors
- Stabilizing treatments
- Positive prognostic indicators
- Recovery trajectory patterns"""
        
        summary_prompt = f"""Create a CONCISE medical summary in EXACTLY 6000 tokens or less.

STRICT REQUIREMENTS:
- Maximum 6000 tokens
- Use bullet points for key information
- No repetition or redundancy
- Focus ONLY on essential medical facts

Focus areas:
{focus_areas}

Original text ({len(round_text)} chars): {round_text}

CONCISE SUMMARY (6000 tokens max):
•"""

        try:
            # Use integrator model for summarization
            if self.use_fireworks:
                summary = self._call_fireworks_api(summary_prompt, max_tokens=6000, temperature=0.3)
            else:
                llm_to_use = self.integrator_llm if hasattr(self, 'integrator_llm') and self.integrator_llm else self.llm
                # Use raw VLLM generate
                sampling_params = SamplingParams(
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=6000,
                    stop=["<|im_end|>", "</s>"],
                    repetition_penalty=1.1
                )
                outputs = llm_to_use.generate([summary_prompt], sampling_params)
                summary = outputs[0].outputs[0].text
            
            print(f"[ROUND] Summary generated: {len(summary)} chars")
            return summary
            
        except Exception as e:
            print(f"[ERROR] Summarization failed: {e}, using truncated original")
            # Fallback: truncate to approximate token limit
            return round_text[:token_limit_chars]
    
    def _prepare_integrator_history(self, debate_history: List[Dict[str, Any]]) -> str:
        """
        Prepare history for integrator by summarizing individual rounds if needed and combining them.
        
        Args:
            debate_history: List of all previous debate responses
            
        Returns:
            Combined history text with individual rounds summarized if needed
        """
        combined_history = "\n## Previous Analysis:\n"
        
        # Process each round individually
        for entry in debate_history:
            role = entry.get('role', 'unknown')
            message = entry.get('message', '')
            
            # Determine round name for summarization
            if 'risk' in role.lower() or 'mortality' in role.lower():
                round_name = "mortality_risk_assessment"
            elif 'protective' in role.lower() or 'survival' in role.lower():
                round_name = "protective_factor_analysis"
            else:
                round_name = role
            
            # Summarize if needed
            summarized_message = self._summarize_round_response(message, round_name, target_tokens=4000)
            
            # Add to combined history
            combined_history += f"\n### {role.upper().replace('_', ' ')}:\n{summarized_message}\n"
        
        print(f"[INTEGRATOR] Prepared combined history: {len(combined_history)} chars")
        return combined_history
    
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
        system_prompt = self.agent_prompts.get(role, self.agent_prompts["mortality_risk_assessor"])
        
        # Format debate history for context (only for integrator)
        history_text = ""
        if debate_history and role == "balanced_clinical_integrator":
            history_text = self._prepare_integrator_history(debate_history)
        
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
        
        # Build context based on agent role - LABEL-BLIND for analysts
        if role == "mortality_risk_assessor":
            # Label-blind: only show one similar patient without mortality label
            primary_context = f"## Target Patient ##\n{patient_context}"
            secondary_context = f"\n## Similar Patient ##\n{similar_patients.get('positive', 'No similar patient available.')}"
        elif role == "protective_factor_analyst":
            # Label-blind: only show one similar patient without mortality label
            primary_context = f"## Target Patient ##\n{patient_context}"
            secondary_context = f"\n## Similar Patient ##\n{similar_patients.get('negative', 'No similar patient available.')}"
        elif role == "balanced_clinical_integrator":
            primary_context = f"## Target Patient EHR Context ##\n{patient_context}"
            secondary_context = ""
        
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
                "mortality_risk_assessor": 0.3,
                "protective_factor_analyst": 0.3,
                "balanced_clinical_integrator": 0.5
            }
            temperature = agent_temps.get(role, 0.5)
            
            # Set token limits based on debate round and role
            if role in ["mortality_risk_assessor", "protective_factor_analyst"]:
                max_tokens = 2048
            elif role == "balanced_clinical_integrator":
                max_tokens = 4096
            else:
                max_tokens = 2048
            
            # Select appropriate model/API based on agent role
            if role == "balanced_clinical_integrator":
                # Use integrator model if available
                if self.use_fireworks:
                    response = self._call_fireworks_api(prompt, max_tokens=max_tokens, temperature=temperature)
                else:
                    # Use raw VLLM generate (not VLLMWrapper)
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=0.9,
                        max_tokens=max_tokens,
                        stop=["<|im_end|>", "</s>"],
                        repetition_penalty=1.2
                    )
                    llm_to_use = self.integrator_llm if hasattr(self, 'integrator_llm') and self.integrator_llm else self.llm
                    outputs = llm_to_use.generate([prompt], sampling_params)
                    response = outputs[0].outputs[0].text
            elif self.use_fireworks:
                # Use Fireworks API for non-integrator agents
                response = self._call_fireworks_api(prompt, max_tokens=max_tokens, temperature=temperature)
            else:
                # Use raw VLLM generate for non-integrator agents
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=max_tokens,
                    stop=["<|im_end|>", "</s>"],
                    repetition_penalty=1.2
                )
                outputs = self.llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text
            
            generation_time = time.time() - start_time
            
            # Log raw response for debugging
            log_message = f"\n{'='*50}\nRAW RESPONSE from {role.upper()}\n{'='*50}\nResponse type: {type(response)}\nResponse length: {len(response) if response else 0}\nFull response: {response}\n{'='*50}"
            if logger:
                logger.info(log_message)
            
            # Extract prediction and probabilities
            extraction_result = self._extract_prediction_and_probabilities(response)
            if logger:
                logger.info(f"EXTRACTED PREDICTION: {extraction_result['prediction']}")
                logger.info(f"MORTALITY PROBABILITY: {extraction_result['mortality_probability']}")
                logger.info(f"SURVIVAL PROBABILITY: {extraction_result['survival_probability']}")
            
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
        
        # Round 1: Similar Patient Comparisons (Label-Blind)
        print(f"\n--- ROUND 1: SIMILAR PATIENT COMPARISONS (LABEL-BLIND) ---")
        
        # Mortality risk assessor (analyzes one similar patient - label-blind)
        risk_response = self._agent_turn(
            role="mortality_risk_assessor",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=[],
            logger=logger
        )
        debate_history.append(risk_response)
        print(f"Risk Assessor: {risk_response.get('message', 'No response')[:200]}...")
        
        # Protective factor analyst (analyzes another similar patient - label-blind)
        protective_response = self._agent_turn(
            role="protective_factor_analyst",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger
        )
        debate_history.append(protective_response)
        print(f"Protective Factor Analyst: {protective_response.get('message', 'No response')[:200]}...")
        
        # Round 2: Integration and Final Consensus (Single-Step)
        print(f"\n--- ROUND 2: INTEGRATION AND CONSENSUS ---")
        integrator_response = self._agent_turn(
            role="balanced_clinical_integrator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger
        )
        debate_history.append(integrator_response)
        print(f"Clinical Integrator: {integrator_response.get('message', 'No response')[:200]}...")
        print(f"Final Prediction: {integrator_response.get('prediction')}")
        
        # Use integrator's prediction as the final result with improved fallback logic
        final_prediction = integrator_response.get('prediction')
        final_mortality_prob = integrator_response.get('mortality_probability')
        final_survival_prob = integrator_response.get('survival_probability')
        final_confidence = integrator_response.get('confidence')
        
        # Improved fallback: if integrator fails to output probabilities, re-run Round 1 agents
        if final_mortality_prob is None and final_survival_prob is None:
            print("[WARNING] Integrator failed to produce probabilities, re-running Round 1 agents...")
            if logger:
                logger.warning("Integrator produced no probabilities - re-running Round 1 agents")
            
            # Re-run Round 1 agents
            retry_risk = self._agent_turn(
                role="mortality_risk_assessor",
                patient_context=patient_context,
                similar_patients=similar_patients_dict,
                medical_knowledge=medical_knowledge,
                debate_history=[],
                logger=logger
            )
            
            retry_protective = self._agent_turn(
                role="protective_factor_analyst",
                patient_context=patient_context,
                similar_patients=similar_patients_dict,
                medical_knowledge=medical_knowledge,
                debate_history=[retry_risk],
                logger=logger
            )
            
            # Check if either agent produced probabilities
            retry_mort_prob = retry_risk.get('mortality_probability') or retry_protective.get('mortality_probability')
            retry_surv_prob = retry_risk.get('survival_probability') or retry_protective.get('survival_probability')
            
            if retry_mort_prob is not None and retry_mort_prob > 0.5:
                final_prediction = 1
                final_mortality_prob = retry_mort_prob
                print(f"[FALLBACK] Using mortality probability {retry_mort_prob:.3f} -> prediction: 1")
                if logger:
                    logger.info(f"Fallback: mortality_prob={retry_mort_prob:.3f} > 0.5, prediction=1")
            elif retry_surv_prob is not None and retry_surv_prob > 0.5:
                final_prediction = 0
                final_survival_prob = retry_surv_prob
                print(f"[FALLBACK] Using survival probability {retry_surv_prob:.3f} -> prediction: 0")
                if logger:
                    logger.info(f"Fallback: survival_prob={retry_surv_prob:.3f} > 0.5, prediction=0")
            elif retry_mort_prob is not None:
                final_prediction = 0
                final_mortality_prob = retry_mort_prob
                print(f"[FALLBACK] Mortality probability {retry_mort_prob:.3f} <= 0.5 -> prediction: 0")
                if logger:
                    logger.info(f"Fallback: mortality_prob={retry_mort_prob:.3f} <= 0.5, prediction=0")
            elif retry_surv_prob is not None:
                final_prediction = 1
                final_survival_prob = retry_surv_prob
                print(f"[FALLBACK] Survival probability {retry_surv_prob:.3f} <= 0.5 -> prediction: 1")
                if logger:
                    logger.info(f"Fallback: survival_prob={retry_surv_prob:.3f} <= 0.5, prediction=1")
            else:
                # Still no probabilities, predict opposite of ground truth as final fallback
                if ground_truth is not None:
                    final_prediction = 1 - ground_truth
                    print(f"[FALLBACK] Still no probabilities from retry, predicting opposite of ground truth: {final_prediction} (GT={ground_truth})")
                    if logger:
                        logger.warning(f"Double fallback - both probabilities None after retry, predicting opposite of ground_truth={ground_truth}, prediction={final_prediction}")
                else:
                    final_prediction = None
        
        print(f"\n{'='*80}")
        print(f"DEBATE COMPLETED - Final Prediction: {final_prediction}")
        if final_mortality_prob is not None:
            print(f"Final Mortality Probability: {final_mortality_prob}")
        if final_survival_prob is not None:
            print(f"Final Survival Probability: {final_survival_prob}")
        if final_confidence is not None:
            print(f"Final Confidence Level: {final_confidence}")
        print(f"{'='*80}")
        
        # Log final result and probabilities
        logger.info(f"DEBATE COMPLETED - Final Prediction: {final_prediction}")
        if final_mortality_prob is not None:
            logger.info(f"Final Mortality Probability: {final_mortality_prob}")
        if final_survival_prob is not None:
            logger.info(f"Final Survival Probability: {final_survival_prob}")
        if final_confidence is not None:
            logger.info(f"Final Confidence Level: {final_confidence}")
        logger.info(f"Debate history: {len(debate_history)} rounds completed")
        
        # Clean up handlers - properly close file handlers
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
            'rounds_completed': 2,  # Now 2 rounds after alignment with RAG mode
            'total_generation_time': sum(r.get('generation_time', 0) for r in debate_history),
            'integrator_prediction': integrator_response.get('prediction'),
            'integrator_mortality_probability': integrator_response.get('mortality_probability'),
            'integrator_survival_probability': integrator_response.get('survival_probability'),
            'integrator_confidence': integrator_response.get('confidence')
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