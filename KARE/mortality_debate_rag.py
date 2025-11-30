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

# Add MedRAG paths for VLLM wrapper and retrieval
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"

sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))
sys.path.insert(0, mirage_src)

from run_medrag_vllm import VLLMWrapper, patch_medrag_for_vllm
from medrag import MedRAG

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
                 rag_enabled: bool = True,
                 corpus_name: str = "MedCorp2",
                 retriever_name: str = "MedCPT",
                 db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"):
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
        
        # Store RAG configuration
        self.rag_enabled = rag_enabled
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.db_dir = db_dir
        
        # Initialize MedRAG if enabled
        if self.rag_enabled:
            print(f"Initializing MedRAG with {corpus_name} corpus and {retriever_name} retriever...")
            patch_medrag_for_vllm()
            self.medrag = MedRAG(
                llm_name=model_name,
                rag=True,
                retriever_name=retriever_name,
                corpus_name=corpus_name,
                db_dir=db_dir,
                corpus_cache=True,
                HNSW=True
            )
            # Create retrieval tools for each round
            self.retrieval_tools = {
                "round1": self._create_retrieval_tool(k=8),
                "round2": self._create_retrieval_tool(k=8), 
                "round3": self._create_retrieval_tool(k=16)
            }
            print("MedRAG initialization complete.")
        else:
            self.medrag = None
            self.retrieval_tools = {}
            print("RAG disabled - using debate without medical retrieval.")
        
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
            "mortality_risk_assessor", 
            "protective_factor_analyst",
            "balanced_clinical_integrator"
        ]
        self.max_rounds = 3
        
        # System prompts for each agent
        self.agent_prompts = self._initialize_agent_prompts()
        
    def _initialize_agent_prompts(self) -> Dict[str, str]:
        """Initialize specialized prompts for each agent role."""
        
        return {
            "target_patient_analyst": """You are a medical AI that provides balanced clinical assessment for mortality prediction in the NEXT hospital visit.

You have access to:
1) The target patient's temporal EHR context
2) Retrieved medical evidence documents (if available)

IMPORTANT CONTEXT: Mortality is relatively rare. Only patients with extremely very high risk of mortality (definitely die) should be predicted as 1.

Your job:
1) Read all visits in order and summarize the main clinical story (conditions, procedures, medications).
2) Review the retrieved medical evidence for relevant mortality/survival factors.
3) **BALANCED ASSESSMENT**: List 3 main RISK factors that increase mortality risk AND 3 main PROTECTIVE factors that support survival.
4) Provide a brief explanation (2-3 sentences) integrating patient data with medical evidence.
5) Estimate probability scores and make prediction.

CONSERVATIVE PREDICTION GUIDELINES:
- Only predict mortality (1) if evidence STRONGLY indicates death is very likely
- When uncertain, err toward survival prediction (0)
- Consider both risk factors AND protective factors in your assessment

CRITICAL FORMAT REQUIREMENT: You MUST end your response with exactly these lines:
Mortality Probability: X.XX (0.00 to 1.00)
Survival Probability: X.XX (0.00 to 1.00)
\\boxed{0} - if you predict SURVIVAL (patient will NOT die in next visit)
\\boxed{1} - if you predict MORTALITY (patient WILL die in next visit)

Do not include anything after the \\boxed{} prediction.""",

            "mortality_risk_assessor": """You are a medical AI Risk Assessor that identifies factors that increase mortality risk.

You have access to:
1) Similar patients with mortality outcomes (who died)
2) Retrieved medical evidence documents (if available)

IMPORTANT CONTEXT: You are providing risk assessment evidence. Remember that mortality is relatively rare, so focus on truly significant risk factors.

Your job (EVIDENCE ANALYSIS ONLY, NO FINAL PREDICTION):
1) Analyze common patterns in fatal cases (diseases, procedures, medications, clinical trajectories).
2) Review retrieved medical evidence for established mortality risk factors.
3) Identify 3-5 key factors that clearly and significantly increase mortality risk.
4) **CRITICAL ASSESSMENT**: Rate the strength of each risk factor (Weak/Moderate/Strong evidence).
5) Explain (2-3 sentences) why these factors indicate increased mortality risk, supported by medical evidence.

IMPORTANT:
- DO NOT make a final prediction.
- DO NOT output \\boxed{0} or \\boxed{1}.
- Your job is only to provide EVIDENCE that supports MORTALITY (death in the next visit).""",

            "protective_factor_analyst": """You are a medical AI Protective Factor Analyst that identifies factors that decrease mortality risk and support survival.

You have access to:
1) Similar patients with survival outcomes (who survived)
2) Retrieved medical evidence documents (if available)

IMPORTANT CONTEXT: Most patients in this population survive. Your role is to identify what factors support survival and recovery.

Your job (EVIDENCE ANALYSIS ONLY, NO FINAL PREDICTION):
1) Analyze common patterns in survival cases (effective treatments, recovery trajectories, patient characteristics).
2) Review retrieved medical evidence for established protective and survival factors.
3) Identify 3-5 key factors that clearly support survival and decrease mortality risk.
4) **CRITICAL ASSESSMENT**: Rate the strength of each protective factor (Weak/Moderate/Strong evidence).
5) Explain (2-3 sentences) why these factors indicate decreased mortality risk and support survival, with medical evidence.

IMPORTANT:
- DO NOT make a final prediction.
- DO NOT output \\boxed{0} or \\boxed{1}.
- Your job is only to provide EVIDENCE that supports SURVIVAL (no death in the next visit).""",

            "balanced_clinical_integrator": """You are a medical AI Clinical Integrator that makes the FINAL balanced decision about mortality outcome in the NEXT hospital visit.

You have access to:
- Target patient's balanced clinical assessment with risk/protective factors (Round 1)
- Mortality risk factors from Risk Assessor
- Protective/survival factors from Protective Factor Analyst
- Retrieved comprehensive medical evidence documents (if available)

IMPORTANT CONTEXT: Mortality is relatively rare. Only patients with extremely very high risk of mortality (definitely die) should be predicted as 1

Your job:
1) **BALANCE ASSESSMENT**: Weigh mortality risks against survival/protective factors systematically.
2) Review all medical evidence for supporting or contradicting information.
3) **CONSERVATIVE DECISION MAKING**: Only predict mortality if evidence STRONGLY indicates death is very likely.
4) Consider the base rate: most patients survive, so strong evidence is needed to predict mortality.
5) Provide final probability estimates and reasoning.

Decision Framework:
- Strong evidence for mortality + weak protective factors → Consider mortality (1)
- Moderate evidence for mortality + strong protective factors → Lean survival (0)  
- Any uncertainty or balanced evidence → Default to survival (0)

CRITICAL FORMAT REQUIREMENT: You MUST end your response with exactly these lines:
Final Mortality Probability: X.XX (0.00 to 1.00)
Final Survival Probability: X.XX (0.00 to 1.00)
Confidence Level: [Very High/High/Moderate/Low]
\\boxed{0} - if your FINAL prediction is SURVIVAL (patient will NOT die in next visit)
\\boxed{1} - if your FINAL prediction is MORTALITY (patient WILL die in next visit)

Do not include anything after the \\boxed{} prediction."""
        }
    
    def _create_retrieval_tool(self, k=8):
        """Create a retrieval tool for medical evidence - following run_debate_medrag_rag.py pattern"""
        if not self.rag_enabled or not self.medrag:
            return None
            
        def retrieve_tool(query, qid=None, log_dir=None):
            """Retrieve medical documents using MedRAG system"""
            try:
                print(f"[RETRIEVE] Query length: {len(query)} chars")
                
                # Use MedRAG's source-specific retrieval method
                if hasattr(self.medrag, 'medrag_answer_by_source'):
                    _, retrieved_snippets, scores = self.medrag.medrag_answer_by_source(
                        question=query,
                        options=None,
                        k=k,
                        rrf_k=100,
                        save_dir=log_dir
                    )
                else:
                    # Fallback to standard method
                    _, retrieved_snippets, scores = self.medrag.medrag_answer(
                        question=query,
                        options=None, 
                        k=k,
                        save_dir=log_dir
                    )
                
                # Format documents for agent context
                formatted_docs = []
                for i, doc in enumerate(retrieved_snippets):
                    formatted_docs.append({
                        'id': doc.get('id', f'doc_{i+1}'),
                        'title': doc.get('title', 'Medical Document'),
                        'content': doc.get('content', ''),
                        'score': scores[i] if i < len(scores) else 0.0,
                        'source_type': doc.get('source_type', 'unknown'),
                        'query_used': query[:200] + "..." if len(query) > 200 else query
                    })
                
                # Save retrieved documents to log directory if provided
                if log_dir and qid:
                    retrieval_file = Path(log_dir) / f"retrieve_{qid}.json"
                    retrieval_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    retrieval_data = {
                        'patient_id': qid,
                        'query': query,
                        'query_length': len(query),
                        'num_retrieved': len(formatted_docs),
                        'k_requested': k,
                        'retrieved_documents': formatted_docs,
                        'timestamp': time.time()
                    }
                    
                    with open(retrieval_file, 'w', encoding='utf-8') as f:
                        json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"[RETRIEVE] Saved {len(formatted_docs)} documents to {retrieval_file}")
                
                print(f"[RETRIEVE] Retrieved {len(formatted_docs)} documents")
                return formatted_docs
                
            except Exception as e:
                print(f"[ERROR] Retrieval failed: {e}")
                return []
        
        return {"name": "retrieve", "func": retrieve_tool}
    
    def _convert_labels_in_text(self, text: str) -> str:
        """Convert mortality labels in text: 0->survive, 1->mortality"""
        # Replace various label formats
        text = text.replace("label: 0", "label: survive")
        text = text.replace("label:0", "label: survive") 
        text = text.replace("labels: 0", "labels: survive")
        text = text.replace("mortality = 0", "mortality = survive")
        text = text.replace("mortality=0", "mortality=survive")
        
        text = text.replace("label: 1", "label: mortality")
        text = text.replace("label:1", "label: mortality")
        text = text.replace("labels: 1", "labels: mortality") 
        text = text.replace("mortality = 1", "mortality = mortality")
        text = text.replace("mortality=1", "mortality=mortality")
        
        return text
    
    def _extract_prediction_and_probabilities(self, response) -> Dict[str, Any]:
        """
        Extract prediction, probability scores, and confidence from agent response.
        
        Args:
            response: Agent response (string or list format)
            
        Returns:
            Dictionary with prediction, mortality_prob, survival_prob, confidence
        """
        # Ensure response is a string
        if isinstance(response, list):
            if len(response) > 0 and isinstance(response[0], dict):
                response = response[0].get('generated_text', '')
            else:
                response = str(response)
        elif not isinstance(response, str):
            response = str(response)
        
        print(f"Debug: Extracting prediction and probabilities from response length: {len(response)}")
        print(f"Debug: Response tail (last 200 chars): ...{response[-200:] if len(response) > 200 else response}")
        
        result = {
            'prediction': None,
            'mortality_probability': None,
            'survival_probability': None,
            'confidence': None
        }
        
        # Extract mortality probability
        mortality_patterns = [
            r'Mortality Probability:\s*([0-9]*\.?[0-9]+)',
            r'Final Mortality Probability:\s*([0-9]*\.?[0-9]+)',
            r'mortality\s*probability\s*:?\s*([0-9]*\.?[0-9]+)',
        ]
        for pattern in mortality_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['mortality_probability'] = float(match.group(1))
                print(f"Debug: Found mortality probability {result['mortality_probability']}")
                break
                
        # Extract survival probability  
        survival_patterns = [
            r'Survival Probability:\s*([0-9]*\.?[0-9]+)',
            r'Final Survival Probability:\s*([0-9]*\.?[0-9]+)', 
            r'survival\s*probability\s*:?\s*([0-9]*\.?[0-9]+)',
        ]
        for pattern in survival_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['survival_probability'] = float(match.group(1))
                print(f"Debug: Found survival probability {result['survival_probability']}")
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
                print(f"Debug: Found confidence {result['confidence']}")
                break
        
        # Look for \\boxed{0} or \\boxed{1} pattern - EXACTLY 0 or 1, not other characters
        boxed_pattern = r'\\boxed\{([01])\}'
        match = re.search(boxed_pattern, response)
        
        if match:
            result['prediction'] = int(match.group(1))
            print(f"Debug: Found boxed prediction: {result['prediction']}")
            return result
        
        # Also try without double backslash (in case it's single backslash) - EXACTLY 0 or 1
        simple_pattern = r'boxed\{([01])\}'
        match = re.search(simple_pattern, response)
        
        if match:
            result['prediction'] = int(match.group(1))
            print(f"Debug: Found simple boxed prediction: {result['prediction']}")
            return result
        
        # Try more flexible patterns
        flexible_patterns = [
            r'boxed\s*\{\s*([01])\s*\}',  # Allow whitespace
            r'final\s+prediction\s*:?\s*([01])',  # "final prediction: 0"
            r'prediction\s*:?\s*([01])',  # "prediction: 0"
            r'answer\s*:?\s*([01])',  # "answer: 0"
            r'\b([01])\s*(?:for|is)\s+(?:the\s+)?(?:final\s+)?(?:mortality\s+)?prediction',  # "0 for the final prediction"
        ]
        
        for pattern in flexible_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['prediction'] = int(match.group(1))
                print(f"Debug: Found flexible prediction with pattern '{pattern}': {result['prediction']}")
                return result
        
        print(f"Debug: No prediction pattern found in response")
        print(f"Debug: Looking for 'boxed', 'prediction', 'answer' in response:")
        for keyword in ['boxed', 'prediction', 'answer', 'final']:
            if keyword.lower() in response.lower():
                # Find the context around the keyword
                idx = response.lower().find(keyword.lower())
                start = max(0, idx - 50)
                end = min(len(response), idx + 100)
                context = response[start:end]
                print(f"Debug: Found '{keyword}' at position {idx}: ...{context}...")
        
        return result
    
    def _agent_turn(self, 
                   role: str, 
                   patient_context: str, 
                   similar_patients: Dict[str, str], 
                   medical_knowledge: str = "",
                   debate_history: List[Dict[str, Any]] = None,
                   logger = None,
                   patient_id: str = "unknown",
                   log_dir: str = None) -> Dict[str, Any]:
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
        if debate_history and role not in ["mortality_risk_assessor", "protective_factor_analyst"]:
            history_text = "\n## Previous Analysis:\n"
            for entry in debate_history[-4:]:  # Last 4 entries for context
                agent_role = entry.get('role', 'Unknown')
                message = entry.get('message', '')
                prediction = entry.get('prediction', 'Unknown')
                history_text += f"{agent_role}: {message} [Prediction: {prediction}]\n"
            history_text += "\n"
        
        # Perform retrieval if RAG is enabled
        retrieved_docs = []
        retrieval_context = ""
        if self.rag_enabled and role in ["target_patient_analyst", "mortality_risk_assessor", "protective_factor_analyst", "balanced_clinical_integrator"]:
            # Determine which retrieval tool to use based on role
            if role == "medical_knowledge_integrator":
                retrieval_tool = self.retrieval_tools.get("round3")
            else:
                retrieval_tool = self.retrieval_tools.get("round1" if role == "target_patient_analyst" else "round2")
            
            if retrieval_tool:
                # Generate query based on role and available context
                if role == "target_patient_analyst":
                    query = self._convert_labels_in_text(patient_context)
                elif role == "positive_similar_comparator":
                    query = self._convert_labels_in_text(f"{patient_context}\n\nSimilar patients with mortality outcomes:\n{similar_patients.get('positive', '')}")
                elif role == "negative_similar_comparator":
                    query = self._convert_labels_in_text(f"{patient_context}\n\nSimilar patients with survival outcomes:\n{similar_patients.get('negative', '')}")
                else:  # medical_knowledge_integrator
                    query = self._convert_labels_in_text(f"{patient_context}\n\nDebate Summary:\n{history_text}")
                
                # Retrieve documents with patient ID and log directory for saving
                retrieval_qid = f"{role}_{patient_id}"
                retrieved_docs = retrieval_tool["func"](query[:4000], qid=retrieval_qid, log_dir=log_dir)  # Limit query length
                
                # Format retrieval context
                if retrieved_docs:
                    retrieval_context = "\n\n## Retrieved Medical Evidence ##\n"
                    for doc in retrieved_docs:
                        doc_text = f"[{doc['id']}] {doc['title']}\n{doc['content'][:1000]}..."
                        retrieval_context += f"{doc_text}\n\n"
        
        # Debug: Check similar patient data
        print(f"\n--- DEBUG: Data for {role.upper()} ---")
        print(f"Target context length: {len(patient_context) if patient_context else 0}")
        print(f"Similar patients keys: {list(similar_patients.keys()) if similar_patients else 'None'}")
        print(f"Retrieved docs: {len(retrieved_docs)}")
        if similar_patients:
            print(f"Positive similars length: {len(similar_patients.get('positive', ''))}")
            print(f"Negative similars length: {len(similar_patients.get('negative', ''))}")
            print(f"Length of history text: {len(history_text)}")
            print(f"Tail of history text: {history_text[-200:]}...")
        print(f"--- END DEBUG ---\n")
        
        # Build context based on agent role (Phase 1 restructured)
        if role == "target_patient_analyst":
            primary_context = f"## Target Patient EHR Context ##\n{patient_context}"
            secondary_context = ""
        elif role == "mortality_risk_assessor":
            primary_context = f"## Similar Patients with Mortality Outcomes (Died) ##\n{similar_patients.get('positive', 'No mortality cases available for analysis.')}"
            secondary_context = ""
        elif role == "protective_factor_analyst":
            primary_context = f"## Similar Patients with Survival Outcomes (Survived) ##\n{similar_patients.get('negative', 'No survival cases available for analysis.')}"
            secondary_context = ""
        else:  # balanced_clinical_integrator
            primary_context = f"## Target Patient EHR Context ##\n{patient_context}"
            secondary_context = f"\n## Mortality Risk Cases ##\n{similar_patients.get('positive', 'None')}"
            secondary_context += f"\n## Survival Cases ##\n{similar_patients.get('negative', 'None')}"
            if medical_knowledge:
                secondary_context += f"\n## Medical Knowledge ##\n{medical_knowledge}"
        
        # Build full prompt with retrieval context
        prompt = f"""{system_prompt}

{primary_context}{secondary_context}

{retrieval_context}{history_text}

Provide your clinical analysis and mortality risk assessment:"""
        
        # Generate response
        try:
            start_time = time.time()
            
            # Use different temperatures for each agent to promote diversity
            agent_temps = {
                "target_patient_analyst": 0.7,
                "mortality_risk_assessor": 0.3,
                "protective_factor_analyst": 0.3,
                "balanced_clinical_integrator": 0.5
            }
            temperature = agent_temps.get(role, 0.5)
            
            # Set token limits based on debate round and role
            if role == "target_patient_analyst":
                max_tokens = 8192  # Round 1: Comprehensive target analysis with full context
            elif role in ["mortality_risk_assessor", "protective_factor_analyst"]:
                max_tokens = 8192  # Round 2: Detailed comparison analysis with full similar patient context
            else:  # balanced_clinical_integrator
                max_tokens = 32768  # Round 3: Maximum available tokens for comprehensive integration
            
            # Select appropriate model based on agent role
            if role == "balanced_clinical_integrator":
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
            
            # Extract prediction and probabilities
            extraction_result = self._extract_prediction_and_probabilities(response)
            prediction = extraction_result.get('prediction') if isinstance(extraction_result, dict) else None
            mortality_prob = extraction_result.get('mortality_probability') if isinstance(extraction_result, dict) else None
            survival_prob = extraction_result.get('survival_probability') if isinstance(extraction_result, dict) else None
            confidence = extraction_result.get('confidence') if isinstance(extraction_result, dict) else None
            
            # Log prediction and probabilities
            if logger:
                logger.info(f"EXTRACTED PREDICTION: {prediction}")
                if mortality_prob is not None:
                    logger.info(f"EXTRACTED MORTALITY PROBABILITY: {mortality_prob}")
                if survival_prob is not None:
                    logger.info(f"EXTRACTED SURVIVAL PROBABILITY: {survival_prob}")
                if confidence is not None:
                    logger.info(f"EXTRACTED CONFIDENCE LEVEL: {confidence}")
            
            return {
                'role': role,
                'message': response,
                'prediction': prediction if role not in ["mortality_risk_assessor", "protective_factor_analyst"] else None,
                'mortality_probability': mortality_prob,
                'survival_probability': survival_prob,
                'confidence': confidence,
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
                                  output_dir: str = None) -> Dict[str, Any]:
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
            log_dir = script_dir / "results" / f"rag_mor_{clean_model_name}" / "debate_logs"
        
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
        
        print(f"RAG Debug: Output dir parameter: {output_dir}")
        
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
            print(f"RAG Debug: Successfully created log file: {log_filename}")
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
            logger=logger,
            patient_id=patient_id,
            log_dir=str(log_dir)
        )
        debate_history.append(target_response)
        print(f"Target Analysis: {target_response.get('message', 'No response')[:200]}...")
        print(f"Initial Prediction: {target_response.get('prediction')}")
        
        # Round 2: Similar Patient Comparisons
        print(f"\n--- ROUND 2: SIMILAR PATIENT COMPARISONS ---")
        
        # Mortality risk assessor
        positive_response = self._agent_turn(
            role="mortality_risk_assessor",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger,
            patient_id=patient_id,
            log_dir=str(log_dir)
        )
        debate_history.append(positive_response)
        print(f"Risk Assessor: {positive_response.get('message', 'No response')[:200]}...")
        
        # Protective factor analyst  
        negative_response = self._agent_turn(
            role="protective_factor_analyst",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger,
            patient_id=patient_id,
            log_dir=str(log_dir)
        )
        debate_history.append(negative_response)
        print(f"Protective Factor Analyst: {negative_response.get('message', 'No response')[:200]}...")
        
        # Round 3: Integration and Final Consensus
        print(f"\n--- ROUND 3: INTEGRATION AND CONSENSUS ---")
        integrator_response = self._agent_turn(
            role="balanced_clinical_integrator",
            patient_context=patient_context,
            similar_patients=similar_patients_dict,
            medical_knowledge=medical_knowledge,
            debate_history=debate_history,
            logger=logger,
            patient_id=patient_id,
            log_dir=str(log_dir)
        )
        debate_history.append(integrator_response)
        print(f"Clinical Integrator: {integrator_response.get('message', 'No response')[:200]}...")
        print(f"Final Prediction: {integrator_response.get('prediction')}")
        
        # Use integrator's prediction as the final result (no fallback to consensus)
        final_prediction = integrator_response.get('prediction')
        final_mortality_prob = integrator_response.get('mortality_probability')
        final_survival_prob = integrator_response.get('survival_probability')
        final_confidence = integrator_response.get('confidence')
        
        # Only fallback to target if integrator completely fails
        if final_prediction is None:
            print("Warning: No prediction from final integrator, using target prediction as fallback")
            final_prediction = target_response.get('prediction')
            final_mortality_prob = target_response.get('mortality_probability')
            final_survival_prob = target_response.get('survival_probability')
            final_confidence = target_response.get('confidence')
            if final_prediction is None:
                print("Warning: No prediction from any agent, final answer is None")
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
        
        # Clean up handlers - check if file_handler exists
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
                break
        
        return {
            'final_prediction': final_prediction,
            'final_mortality_probability': final_mortality_prob,
            'final_survival_probability': final_survival_prob,
            'final_confidence': final_confidence,
            'debate_history': debate_history,
            'rounds_completed': 3,  # Always 3 rounds in structured flow
            'total_generation_time': sum(r.get('generation_time', 0) for r in debate_history),
            'integrator_prediction': integrator_response.get('prediction'),
            'integrator_mortality_probability': integrator_response.get('mortality_probability'),
            'integrator_survival_probability': integrator_response.get('survival_probability'),
            'integrator_confidence': integrator_response.get('confidence'),
            'target_prediction': target_response.get('prediction'),
            'target_mortality_probability': target_response.get('mortality_probability'),
            'target_survival_probability': target_response.get('survival_probability'),
            'target_confidence': target_response.get('confidence')
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