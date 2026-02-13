#!/usr/bin/env python3
"""
Enhanced Multi-Agent Debate System for KARE Mortality Prediction using effGen
RAG Mode (with MedRAG retrieval)
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
    from effgen_medrag_tool import MedRAGRetrievalTool, DualQueryMedRAGTool
except ImportError:
    print("Error: effgen_medrag_tool.py not found. Make sure it's in the same directory.")
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


class MortalityDebateSystemEffGenRAG:
    """
    Enhanced debate system using effGen framework with MedRAG retrieval.
    RAG Mode: Integrator can retrieve medical evidence during reasoning.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 gpu_ids: str = "0",
                 model_cache_dir: str = "/data/wang/junh/.cache/huggingface",
                 corpus_name: str = "MedCorp2",
                 retriever_name: str = "MedCPT",
                 db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"):
        """
        Initialize the mortality debate system with effGen and MedRAG.
        
        Args:
            model_name: HuggingFace model name or path
            gpu_ids: GPU IDs to use (comma-separated)
            model_cache_dir: Cache directory for models
            corpus_name: MedRAG corpus name
            retriever_name: MedRAG retriever name
            db_dir: MedRAG database directory
        """
        print(f"[INIT] Initializing effGen Debate System (RAG Mode)")
        print(f"[INIT] Model: {model_name}")
        print(f"[INIT] GPU IDs: {gpu_ids}")
        print(f"[INIT] MedRAG Corpus: {corpus_name}, Retriever: {retriever_name}")
        
        # Store configurations
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.model_cache_dir = model_cache_dir
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.db_dir = db_dir
        
        # Set CUDA_VISIBLE_DEVICES once (following MEDRAG_GPU_SETUP_FIX.md)
        print(f"[CUDA] Setting CUDA_VISIBLE_DEVICES={gpu_ids}")
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        
        # Get main GPU for retriever
        gpu_list = gpu_ids.split(',')
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
            traceback.print_exc()
            raise
        
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
        
        # System prompts for each agent (identical to VLLM RAG version)
        self.agent_prompts = self._initialize_agent_prompts()
        
        # Create MedRAG retrieval tools
        self.retrieval_tool = None  # Will be set per debate with log_dir
        
        # Create agent instances (without tools initially)
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

            "balanced_clinical_integrator": """You are a medical AI Clinical Assistant with access to medical evidence retrieval.

CRITICAL: You MUST use the retrieve_medical_evidence tool to search for clinical evidence BEFORE making your final prediction. Never make final assessments without first retrieving evidence.

Workflow:
1) Review the Target patient and the two analyses from other agents
2) FIRST: Use retrieve_medical_evidence tool to search for relevant medical evidence about key risk factors
3) THEN: After receiving retrieved evidence, analyze BOTH mortality risks AND survival factors
4) FINALLY: Provide your assessment with probabilities:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00
IMPORTANT: Mortality is rare - only assign high mortality probability when evidence STRONGLY supports it."""
        }
    
    def _initialize_agents(self):
        """Initialize effGen agents with proper configurations."""
        print(f"[AGENTS] Initializing 3 specialized agents...")
        
        self.agents = {}
        
        # Agent 1: Mortality Risk Assessor (NO tools)
        self.agents["mortality_risk_assessor"] = Agent(
            config=AgentConfig(
                name="mortality_risk_assessor",
                model=self.model,
                tools=[],  # No retrieval for Round 1 agents
                system_prompt=self.agent_prompts["mortality_risk_assessor"],
                max_iterations=1,
                temperature=0.3,
            )
        )
        print(f"[AGENTS] Created mortality_risk_assessor")
        
        # Agent 2: Protective Factor Analyst (NO tools)
        self.agents["protective_factor_analyst"] = Agent(
            config=AgentConfig(
                name="protective_factor_analyst",
                model=self.model,
                tools=[],  # No retrieval for Round 1 agents
                system_prompt=self.agent_prompts["protective_factor_analyst"],
                max_iterations=1,
                temperature=0.3,
            )
        )
        print(f"[AGENTS] Created protective_factor_analyst")
        
        # Agent 3: Balanced Clinical Integrator (WITH retrieval tool)
        # Note: Tool will be added dynamically per debate (needs log_dir)
        self.agents["balanced_clinical_integrator"] = None  # Created dynamically
        
        print(f"[AGENTS] Agents initialized (integrator created per-debate)")
    
    def _create_integrator_agent(self, log_dir: str):
        """Create integrator agent with retrieval tool."""
        print(f"[INTEGRATOR] Creating integrator with retrieval tool (log_dir={log_dir})")
        
        # Create retrieval tool with log directory
        retrieval_tool = MedRAGRetrievalTool(
            medrag_instance=self.medrag,
            k=8,
            max_query_tokens=2048,
            log_dir=log_dir
        )
        
        # Create integrator agent with tool
        integrator = Agent(
            config=AgentConfig(
                name="balanced_clinical_integrator",
                model=self.model,
                tools=[retrieval_tool],
                system_prompt=self.agent_prompts["balanced_clinical_integrator"],
                max_iterations=5,  # Allow for tool use + reasoning (match effgen example)
                temperature=0.5,
                enable_sub_agents=False,  # Disable sub-agents
                enable_memory=False  # Disable memory for reproducibility
            )
        )
        
        print(f"[INTEGRATOR] Integrator agent created with retrieval tool")
        return integrator
    
    def _extract_prediction_and_probabilities(self, response: str, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """Extract prediction and probability scores from agent response."""
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
        """Prepare history for integrator with labels."""
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
                   integrator_agent = None,
                   ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """Execute a single agent turn using effGen."""
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
            
            # Get agent (use provided integrator for integrator role)
            if role == "balanced_clinical_integrator" and integrator_agent:
                agent = integrator_agent
            else:
                agent = self.agents.get(role)
            
            if not agent:
                raise ValueError(f"Agent {role} not found")
            
            # Run agent with effgen
            print(f"[{role}] Running agent with effgen...")
            # For integrator with tools, use default mode to enable ReAct
            # For other agents without tools, use SINGLE mode
            if role == "balanced_clinical_integrator":
                # Integrator has tools - use default mode for ReAct loop
                result = agent.run(prompt)
            else:
                # Other agents have no tools - use SINGLE mode
                from effgen.core.agent import AgentMode
                result = agent.run(prompt, mode=AgentMode.SINGLE)
            
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
                logger.info(f"IS FALLBACK: {extraction_result['is_fallback']}")
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
        """Conduct structured multi-agent debate with RAG."""
        print(f"\n{'='*80}")
        print("STARTING MORTALITY PREDICTION DEBATE (effGen RAG Mode)")
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
            log_dir = script_dir / "results" / f"effgen_rag_{clean_model_name}" / "logs"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create integrator agent with retrieval tool (now that we have log_dir)
        integrator_agent = self._create_integrator_agent(str(log_dir))
        
        # Create patient-specific log file
        log_filename = log_dir / f"debate_responses_{patient_id}.log"
        logger = logging.getLogger(f'debate_effgen_rag_{patient_id}')
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
        
        # Round 2: Integration and Consensus (with retrieval)
        print(f"\n--- ROUND 2: INTEGRATION AND CONSENSUS (WITH RETRIEVAL) ---")
        integrator_response = self._agent_turn(
            role="balanced_clinical_integrator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            debate_history=debate_history,
            logger=logger,
            integrator_agent=integrator_agent,
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
    print("Testing effGen RAG Debate System...")
    
    try:
        # Initialize debate system
        debate_system = MortalityDebateSystemEffGenRAG(
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
