#!/usr/bin/env python3
"""
Utility functions for GPT-based ablation experiments.
Provides OpenAI API calls and shared functionality across conditions A, B, C.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")

# Add MedRAG paths for retrieval
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
medrag_src = os.path.join(medrag_root, "src")

print(f"DEBUG: Adding MedRAG paths to sys.path...")
print(f"  medrag_root: {medrag_root} (exists: {os.path.exists(medrag_root)})")
print(f"  medrag_src: {medrag_src} (exists: {os.path.exists(medrag_src)})")
print(f"  Checking for medrag.py: {os.path.exists(os.path.join(medrag_src, 'medrag.py'))}")

# IMPORTANT: Only add MedRAG paths, NOT MIRAGE to avoid utils.py conflicts
sys.path.insert(0, medrag_src)
sys.path.insert(0, medrag_root)

try:
    from medrag import MedRAG
    MEDRAG_AVAILABLE = True
    print("✓ MedRAG module imported successfully")
except ImportError as e:
    MEDRAG_AVAILABLE = False
    print(f"✗ MedRAG import failed: {e}")
    print("  Retrieval functionality will be disabled.")
    import traceback
    traceback.print_exc()




class GPTClient:
    """Wrapper for OpenAI GPT API calls."""
    
    # Model-specific max token limits
    MAX_COMPLETION_TOKENS = {
        "gpt-4-turbo-preview": 4096,
        "gpt-4": 8192,
        "gpt-4o": 16384,
        "gpt-4o-mini": 16384,
        "o3-mini": 100000,  # o3-mini has very high limit
        "gpt-3.5-turbo": 4096,
    }
    
    # Models that use max_completion_tokens instead of max_tokens
    NEW_API_MODELS = ["o3-mini", "o1-preview", "o1-mini"]
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize GPT client.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: GPT model name
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        """
        Generate response from GPT.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            # Cap max_tokens to model's limit
            model_limit = self.MAX_COMPLETION_TOKENS.get(self.model, 4096)
            capped_tokens = min(max_tokens, model_limit)
            
            # Use appropriate parameter name based on model
            if self.model in self.NEW_API_MODELS:
                # o3-mini and o1 models use max_completion_tokens
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=capped_tokens,
                    temperature=temperature
                )
            else:
                # Standard models use max_tokens
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=capped_tokens,
                    temperature=temperature
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            return ""


# EXACT SAME PROMPTS AS QWEN SYSTEM (from mortality_debate_rag.py)
AGENT_PROMPTS = {
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

IMPORTANT: Mortality is rare - only predict mortality probability > 0.5 if evidence STRONGLY supports it. When uncertain, predict survival probability > 0.5. The Target patient is the source of truth. Do not treat Similar-only items as present in the Target.

Available tools:
- <search>query</search>: Retrieve medical evidence. Retrieved information will appear in <information>...</information> tags.

Workflow:
1) Compare the Target patient to two similar cases using the two analysis, and write 3-4 key factors contribute to the target patient's next visit.
2) When you need additional knowledge, call <search>your custom query</search> based on the patient's specific conditions (e.g., <search>sepsis mortality prognosis elderly patients</search>)
3) After seeing the <information>retrieved evidence</information>, analyze BOTH risky factors AND survival factors. Be conservative: mortality is rare, so strong evidence is needed for high mortality probability
4) After reviewing all evidence, provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""
}


def initialize_medrag(corpus_name: str = "MedCorp2", retriever_name: str = "MedCPT",
                      db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
                      retriever_device: str = "cuda:0"):
    """
    Initialize MedRAG for retrieval.
    
    Args:
        corpus_name: Corpus name
        retriever_name: Retriever name
        db_dir: Database directory
        retriever_device: Device for retriever (e.g., "cuda:0", "cpu")
        
    Returns:
        MedRAG instance or None
    """
    if not MEDRAG_AVAILABLE:
        print("ERROR: MedRAG module not available")
        print(f"  MedRAG path: {medrag_root}")
        print(f"  Path exists: {os.path.exists(medrag_root)}")
        print(f"  sys.path includes: {[p for p in sys.path if 'medrag' in p.lower()]}")
        return None
    
    try:
        print(f"Attempting to initialize MedRAG...")
        print(f"  corpus_name: {corpus_name}")
        print(f"  retriever_name: {retriever_name}")
        print(f"  db_dir: {db_dir}")
        print(f"  db_dir exists: {os.path.exists(db_dir)}")
        print(f"  retriever_device: {retriever_device}")
        
        medrag = MedRAG(
            llm_name="OpenAI/gpt-3.5-turbo-16k",  # Placeholder, not used for retrieval
            rag=True,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir,
            corpus_cache=True,
            HNSW=True,
            retriever_device=retriever_device
        )
        
        print("✓ MedRAG initialized successfully")
        return medrag
        
    except Exception as e:
        print(f"ERROR initializing MedRAG: {e}")
        import traceback
        traceback.print_exc()
        return None


def retrieve_documents(medrag, query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve documents using MedRAG.
    
    Args:
        medrag: MedRAG instance
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved documents with scores
    """
    if not medrag:
        return []
    
    try:
        # Handle MedCorp2 which uses source_retrievers instead of retrieval_system
        if hasattr(medrag, 'source_retrievers') and medrag.corpus_name == "MedCorp2":
            # Retrieve from both sources directly (matches pipeline usage)
            k_medcorp = k // 2 + k % 2  # Give extra to general literature if odd
            k_umls = k // 2
            
            all_retrieved_snippets = []
            all_scores = []
            
            for source, k_source in [("medcorp", k_medcorp), ("umls", k_umls)]:
                if source in medrag.source_retrievers:
                    source_retrieval_system = medrag.source_retrievers[source]
                    snippets, scores = source_retrieval_system.retrieve(query, k=k_source, rrf_k=60)
                    all_retrieved_snippets.extend(snippets)
                    all_scores.extend(scores)
            
            retrieved_snippets = all_retrieved_snippets
            scores = all_scores
            
        elif hasattr(medrag, 'retrieval_system') and medrag.retrieval_system:
            # Use direct retrieval system for other corpora
            retrieved_snippets, scores = medrag.retrieval_system.retrieve(query, k=k, rrf_k=60)
        else:
            print("ERROR: MedRAG has no retrieval system available")
            return []
        
        # Format as list of documents
        retrieved_docs = []
        for i, (snippet, score) in enumerate(zip(retrieved_snippets, scores)):
            retrieved_docs.append({
                'rank': i + 1,
                'content': snippet,
                'score': float(score) if score is not None else 0.0
            })
        
        return retrieved_docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        import traceback
        traceback.print_exc()
        return []


def parse_tool_call(response_text: str) -> Optional[str]:
    """
    Parse <search>query</search> from response text.
    
    Args:
        response_text: Response text from integrator
        
    Returns:
        Search query or None
    """
    import re
    
    # Pattern to match <search>query</search>
    patterns = [
        r'<search>(.*?)</search>',
        r'<SEARCH>(.*?)</SEARCH>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            query = match.group(1).strip()
            return query
    
    return None


def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents as context.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Formatted string
    """
    if not docs:
        return "No documents retrieved."
    
    formatted = []
    for doc in docs:
        rank = doc.get('rank', 0)
        score = doc.get('score', 0.0)
        content = doc.get('content', '')
        formatted.append(f"[Document {rank}, Score: {score:.3f}]\n{content}\n")
    
    return "\n".join(formatted)


def extract_probabilities(response: str) -> Dict[str, Any]:
    """
    Extract mortality and survival probabilities from response.
    EXACT SAME LOGIC as Qwen system (_extract_prediction_and_probabilities).
    
    Args:
        response: Response text
        
    Returns:
        Dictionary with prediction, mortality_prob, survival_prob
    """
    import re
    
    result = {
        'prediction': None,
        'mortality_probability': None,
        'survival_probability': None
    }
    
    # Extract mortality probability
    mortality_patterns = [
        r'\*\*MORTALITY PROBABILITY:?\*\*[\s:]*([0-9]*\.?[0-9]+)',
        r'\*\*Mortality Probability:?\*\*[\s:]*([0-9]*\.?[0-9]+)',
        r'MORTALITY PROBABILITY:[\s:]*\*\*([0-9]*\.?[0-9]+)\*\*',  # Handle: MORTALITY PROBABILITY: **0.45**
        r'Mortality Probability:[\s:]*\*\*([0-9]*\.?[0-9]+)\*\*',
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
        r'SURVIVAL PROBABILITY:[\s:]*\*\*([0-9]*\.?[0-9]+)\*\*',  # Handle: SURVIVAL PROBABILITY: **0.55**
        r'Survival Probability:[\s:]*\*\*([0-9]*\.?[0-9]+)\*\*',
        r'SURVIVAL PROBABILITY:[\s]*([0-9]*\.?[0-9]+)',
        r'Survival Probability:[\s]*([0-9]*\.?[0-9]+)',
        r'survival probability[:\s]+([0-9]*\.?[0-9]+)',
    ]
    for pattern in survival_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result['survival_probability'] = float(match.group(1))
            break
    
    # Determine prediction
    mort_prob = result['mortality_probability']
    surv_prob = result['survival_probability']
    
    if mort_prob is not None and surv_prob is not None:
        result['prediction'] = 1 if mort_prob > surv_prob else 0
    elif mort_prob is not None:
        result['prediction'] = 1 if mort_prob > 0.5 else 0
        result['survival_probability'] = 1.0 - mort_prob
    elif surv_prob is not None:
        result['prediction'] = 0 if surv_prob > 0.5 else 1
        result['mortality_probability'] = 1.0 - surv_prob
    
    return result


def load_qwen_debate_history(patient_id: str, base_log_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load Qwen debate history from existing logs.
    
    Args:
        patient_id: Patient ID (e.g., "10117_0")
        base_log_dir: Base directory containing debate logs
        
    Returns:
        Dictionary with analyst outputs and retrieval info, or None
    """
    log_path = Path(base_log_dir) / f"debate_responses_{patient_id}.log"
    
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the log to extract analyst outputs
        result = {
            'analyst1_output': None,
            'analyst2_output': None,
            'retrieval_query': None,
            'retrieval_docs': None,
            'called_retriever': False
        }
        
        # Extract analyst outputs from actual log format
        import re
        
        # Pattern for analyst 1 (mortality_risk_assessor)
        # Format: "BATCH RESPONSE from MORTALITY_RISK_ASSESSOR\n==..==\nResponse length: ...\nFull response: <text>"
        analyst1_match = re.search(
            r'BATCH RESPONSE from MORTALITY_RISK_ASSESSOR.*?Full response:\s*(.*?)(?=BATCH RESPONSE|INTEGRATOR|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        if analyst1_match:
            result['analyst1_output'] = analyst1_match.group(1).strip()
        
        # Pattern for analyst 2 (protective_factor_analyst)
        analyst2_match = re.search(
            r'BATCH RESPONSE from PROTECTIVE_FACTOR_ANALYST.*?Full response:\s*(.*?)(?=BATCH RESPONSE|INTEGRATOR|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        if analyst2_match:
            result['analyst2_output'] = analyst2_match.group(1).strip()
        
        # Check for retrieval
        if '<search>' in content and '<information>' in content:
            result['called_retriever'] = True
            
            # Extract search query
            search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
            if search_match:
                result['retrieval_query'] = search_match.group(1).strip()
            
            # Extract retrieved information
            info_match = re.search(r'<information>(.*?)</information>', content, re.DOTALL)
            if info_match:
                result['retrieval_docs'] = info_match.group(1).strip()
        
        return result
        
    except Exception as e:
        print(f"Error loading Qwen debate history for {patient_id}: {e}")
        return None
