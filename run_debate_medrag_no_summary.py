#!/usr/bin/env python3
"""
Multi-agent debate system for MedQA using MedRAG retrieval.
Two agents (Analyst & Skeptic) retrieve role-specific evidence, debate, and a Judge decides.
"""
# 8 mins for a question in mmlu
import os
import sys
import json
import re
import time
from pathlib import Path

# Add MedRAG paths - use optimized version
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"

sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))

from run_medrag_vllm import patch_medrag_for_vllm, VLLMWrapper, parse_response_standard
from medrag import MedRAG

# Import from MIRAGE (add after MedRAG imports to avoid conflicts)
sys.path.insert(0, mirage_src)
import importlib.util
spec = importlib.util.spec_from_file_location("mirage_utils", os.path.join(mirage_src, "utils.py"))
mirage_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirage_utils)
QADataset = mirage_utils.QADataset
locate_answer = mirage_utils.locate_answer

# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU assignment for debate: GPU 0 for FAISS+embedding, GPU 1 for VLLM
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# Model configuration - using optimized setup
USE_VLLM_ENDPOINT = False
HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# FAISS GPU configuration
FAISS_GPU_ID = 0  # First visible GPU (GPU 0) for FAISS index
VLLM_GPU_ID = 1   # Second visible GPU (GPU 1) for VLLM

MEDCORP_DIR = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"

# Retrieval parameters
DEFAULT_K = 32
RRF_K = 60
MAX_ROUNDS = 2

# ============================================================================
# SYSTEM PROMPTS 
# ============================================================================

# RAG mode: Role-based prompts with tool-calling
ANALYST_SYS = """You are a medical analyst with access to evidence retrieval. Your role is to gather evidence and provide initial clinical assessment.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Process:
1) Call retrieve("search query") to gather relevant medical evidence
2) Analyze the retrieved evidence 
3) Provide your clinical reasoning and conclusion
4) State your answer choice clearly

Focus on evidence-based medicine and clinical guidelines."""

SKEPTIC_SYS = """You are a medical reviewer with access to evidence retrieval. Your role is to critically evaluate the analyst's reasoning and provide alternative perspectives.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Process:
1) Call retrieve("search query") to find additional or contradicting evidence
2) Review both your evidence and the analyst's reasoning
3) Provide critical analysis and your conclusion
4) State your answer choice clearly

Be thorough but constructive in your analysis."""

# CoT mode: Simple equal agent prompts (no role distinction)
AGENT_COT_SYS = """You are a medical AI assistant participating in a clinical discussion.

Instructions:
1) Read the question and options.
2) Write at most 2 sentences of clinical reasoning.
3) State your answer choice clearly

Provide your initial medical analysis."""

# CoT mode: Debate round prompts (with previous discussion context)
AGENT_COT_DEBATE_SYS = """You are a medical AI assistant participating in a clinical discussion.

Instructions:
1) Read the question and options.
2) Review the previous conclusion.
3) Write at most 2 sentences: briefly say whether you agree (add one new reason) or disagree (give one counterpoint).
4) State your answer choice clearly

Consider the previous discussion when forming your response."""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_json(data, path):
    """Save data as JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_jsonl(data, path):
    """Save data as JSONL (one JSON object per line)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def parse_answer_from_text(text):
    """Extract answer choice from agent output - improved sequential parser"""
    if text is None:
        return "PARSE_FAILED", None
    if not isinstance(text, str):
        text = str(text)
    
    # Collect all answers from different pattern types
    all_found_answers = []
    
    # PRIORITY 1: Look for \\boxed{} format (including $\boxed{}$ variants) - find ALL occurrences 
    boxed_patterns = [
        r'\$\\boxed\{([ABCD])\}\$',  # $\boxed{D}$ - highest priority
        r'\\boxed\{([ABCD])\}',     # \boxed{D}
    ]
    for pattern in boxed_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_found_answers.append((match.start(), match.group(1).upper(), 'boxed'))
    
    # PRIORITY 2: Look for answer_choice patterns
    answer_choice_patterns = [
        r'\*\*Answer\s+Choice:\*\*\s*([ABCD])\s*\([^)]+\)',  # **Answer Choice:** C (Description)
        r'Answer\s+Choice:\s*\*\*([ABCD])\*\*',             # Answer Choice: **C**
        r'answer_choice["\']?\s*:\s*["\']?([ABCD])',
        r'"answer_choice"\s*:\s*"([ABCD])"',
        r'answer_choice\s*=\s*["\']?([ABCD])',
    ]
    for pattern in answer_choice_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_found_answers.append((match.start(), match.group(1).upper(), 'answer_choice'))
    
    # PRIORITY 2: Look for explicit "correct answer" statements (highest priority after boxed)
    explicit_answer_patterns = [
        r'The correct answer is\s*\*?([ABCD])\*?:?\s*[A-Z]',  # "The correct answer is C: Previous radiation"
        r'The correct answer is\s*\*?([ABCD])\*?\.?\s*$',     # "The correct answer is C."
        r'The correct answer is\s*\*?([ABCD])\*?',           # "The correct answer is C"
        r'Correct answer:\s*\*?([ABCD])\*?',                 # "Correct answer: C"
        r'correct selection should be\s*\*?([ABCD])\*?',     # "correct selection should be C"
        r'Therefore.{0,30}answer.{0,10}is\s*\*?([ABCD])\*?', # "Therefore the answer is C"
    ]
    for pattern in explicit_answer_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_found_answers.append((match.start(), match.group(1).upper(), 'explicit_correct'))
    
    # PRIORITY 3: Look for Final answer patterns
    final_answer_patterns = [
        r'Final answer:\s*\$?\\boxed\{([ABCD])\}\$?',
        r'Final answer:\s*([ABCD])',
        r'my final stance remains unchanged:\s*\n\n\*\*Answer Choice:\*\*\s*([ABCD])',  # New pattern for the specific case
        r'Therefore.{0,50}\$?\\boxed\{([ABCD])\}\$?',  # Therefore... \boxed{D}
        r'Answer:\s*\$?\\boxed\{([ABCD])\}\$?',
        r'Answer:\s*([ABCD])',
        r'\*\*Answer:\s*([ABCD])\*\*',
        r'The answer is\s*\*\*([ABCD])\*\*',
        r'The answer is\s*([ABCD])',
    ]
    for pattern in final_answer_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_found_answers.append((match.start(), match.group(1).upper(), 'final_answer'))
    
    # PRIORITY 4: Other common patterns
    other_patterns = [
        r'\b([ABCD])\s*is\s+(?:the\s+)?(?:correct|best|right)',
        r'option\s+([ABCD])',
        r'Therefore,?\s*(?:the answer is\s*)?([ABCD])',
    ]
    for pattern in other_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_found_answers.append((match.start(), match.group(1).upper(), 'other'))
    
    # If we found any answers, return the LAST one (most likely final decision)
    if all_found_answers:
        # Sort by position in text and take the last occurrence
        all_found_answers.sort(key=lambda x: x[0])
        final_answer = all_found_answers[-1][1]
        final_position = all_found_answers[-1][0]
        print(f"[DEBUG] Found {len(all_found_answers)} answer patterns, selected last: {final_answer}")
        return final_answer, final_position
    
    # Last resort: look for standalone letters near end of text
    last_part = text[-200:] if len(text) > 200 else text
    standalone_match = re.search(r'\b([ABCD])\b', last_part)
    if standalone_match:
        return standalone_match.group(1).upper(), None
    
    return "PARSE_FAILED", None

def parse_into_dict(response, model_name="Qwen/Qwen3-8B", question_id=None, log_dir=None):
    """Parse response string into structured dict using improved sequential parser"""
    if isinstance(response, dict):
        return response
    
    # Ensure response is a string
    if response is None:
        response = ""
    elif not isinstance(response, str):
        response = str(response)
    
    # First try our improved sequential parser - it handles \boxed{} patterns better
    answer, answer_position = parse_answer_from_text(response)
    
    # If our parser succeeds with a valid answer, use it
    if answer in ["A", "B", "C", "D"]:
        # Truncate step_by_step_thinking to content before the answer pattern
        if answer_position is not None and answer_position > 0:
            truncated_thinking = response[:answer_position].strip()
        else:
            truncated_thinking = response
            
        result = {
            "step_by_step_thinking": truncated_thinking,
            "answer_choice": answer
        }
        return result
    # Only fall back to MedRAG's parser if our improved parser fails
    try:
        parsed_result = parse_response_standard(
            response, 
            model_name=model_name, 
            question_id=question_id, 
            log_dir=log_dir
        )
        # Reject MedRAG's fallback results (they just pick first letter found)
        if (isinstance(parsed_result, dict) and 
            "answer_choice" in parsed_result and 
            parsed_result["answer_choice"] in ["A", "B", "C", "D"] and
            not ("Fallback: extracted single letter" in parsed_result.get("step_by_step_thinking", "") or
                 "Error: Could not parse model response" in parsed_result.get("step_by_step_thinking", ""))):
            return parsed_result
    except Exception as e:
        print(f"[DEBUG] parse_response_standard failed: {e}")
    
    # If our parser fails and response is mostly empty/error, use "ERROR"
    if answer == "PARSE_FAILED" and (not response.strip() or len(response.strip()) < 10):
        answer = "ERROR"
        print(f"[DEBUG] Response too short or empty, using ERROR: '{response.strip()}'")
    
    result = {
        "step_by_step_thinking": response if response.strip() else "Error: Empty or invalid response",
        "answer_choice": answer
    }
    return result

def clean_repetitive_response(text, max_length=1000):
    """
    Clean repetitive text patterns that indicate model got stuck in loops
    """
    if not isinstance(text, str) or not text.strip():
        return text
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Track repeating patterns
    seen_patterns = {}
    repeat_threshold = 3  # Consider something repetitive if it appears 3+ times
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for exact line repetition
        if line in seen_patterns:
            seen_patterns[line] += 1
            # Skip if we've seen this line too many times
            if seen_patterns[line] >= repeat_threshold:
                continue
        else:
            seen_patterns[line] = 1
        
        cleaned_lines.append(line)
    
    # Look for phrase-level repetition patterns
    text = '\n'.join(cleaned_lines)
    
    # Remove patterns like "The correct answer is X" repeated multiple times
    answer_pattern = r'(\*\*(?:The )?(?:correct )?(?:answer )?(?:is )?[ABCD]\*\*)\s*'
    text = re.sub(answer_pattern + r'(\1\s*){2,}', r'\1', text, flags=re.IGNORECASE)
    
    # Remove repeated "Answer: X" patterns
    answer_choice_pattern = r'(Answer:\s*[ABCD])\s*'
    text = re.sub(answer_choice_pattern + r'(\1\s*){2,}', r'\1', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    
    return text.strip()

# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

def create_retrieval_tool(medrag, k=32, rrf_k=100):
    """
    Create a retrieval tool function that agents can call.
    
    Args:
        medrag: MedRAG instance
        k: Number of documents to retrieve
        rrf_k: RRF parameter
    
    Returns:
        Dictionary containing tool definition and function
    """
    def retrieve_tool(query, qid=None, log_dir=None):
        """
        Retrieve relevant medical documents for a query.
        
        Args:
            query: Search query string
            qid: Question ID for logging
            log_dir: Directory for saving retrieval logs
            
        Returns:
            List of retrieved documents with id, title, content
        """
        try:
            print(f"[RETRIEVE] Query: {query}")
            
            # Use MedRAG retrieval system
            if hasattr(medrag, 'answer') and medrag.corpus_name == "MedCorp2":
                # Use the optimized source-specific retrieval for MedCorp2
                _, retrieved_snippets, scores = medrag.medrag_answer_by_source(
                    question=query, 
                    options=None, 
                    k=k, 
                    rrf_k=rrf_k,
                    save_dir=log_dir
                )
            else:
                # Use standard MedRAG retrieval
                _, retrieved_snippets, scores = medrag.medrag_answer(
                    question=query,
                    options=None,
                    k=k,
                    rrf_k=rrf_k,
                    save_dir=log_dir
                )
            
            # Format retrieved documents for tool output
            formatted_docs = []
            for i, doc in enumerate(retrieved_snippets):
                formatted_docs.append({
                    'id': i + 1,
                    'title': doc.get('title', 'Unknown'),
                    'content': doc.get('content', ''),
                    'score': scores[i] if i < len(scores) else 0.0,
                    'source': doc.get('source_type', 'unknown')
                })
            
            print(f"[RETRIEVE] Retrieved {len(formatted_docs)} documents")
            
            # Save retrieval logs if directory provided
            if log_dir and qid:
                save_json({
                    'query': query,
                    'retrieved_docs': formatted_docs,
                    'retrieval_time': time.time()
                }, f"{log_dir}/{qid}__retrieval.json")
            
            return formatted_docs
            
        except Exception as e:
            print(f"[ERROR] Retrieval failed for query '{query}': {e}")
            return []
    
    # Tool definition
    tool_def = {
        "name": "retrieve", 
        "func": retrieve_tool,
        "description": "Retrieve medical evidence documents for a given query"
    }
    
    return tool_def

def parse_tool_call(response_text):
    """
    Parse tool call from agent response.
    
    Args:
        response_text: Agent's response text
        
    Returns:
        Tuple of (tool_name, query) or (None, None) if no tool call found
    """
    # Look for tool call patterns
    patterns = [
        r'Tool Call:\s*retrieve\s*\(\s*["\']([^"\']+)["\'\s]*\)',
        r'retrieve\s*\(\s*["\']([^"\']+)["\'\s]*\)',
        r'Tool:\s*retrieve\s*\(\s*["\']([^"\']+)["\'\s]*\)',
        r'RETRIEVE\s*\(\s*["\']([^"\']+)["\'\s]*\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            query = match.group(1).strip()
            return "retrieve", query
    
    return None, None

def execute_tool_call(tool_name, query, tools, qid=None, log_dir=None):
    """
    Execute a tool call.
    
    Args:
        tool_name: Name of the tool to call
        query: Query parameter for the tool
        tools: Dictionary of available tools
        qid: Question ID for logging
        log_dir: Log directory
        
    Returns:
        Tool execution result
    """
    if tool_name in tools:
        tool_func = tools[tool_name]["func"]
        return tool_func(query, qid=qid, log_dir=log_dir)
    else:
        print(f"[ERROR] Unknown tool: {tool_name}")
        return []

def format_retrieved_docs_for_context(retrieved_docs):
    """
    Format retrieved documents for inclusion in agent context.
    
    Args:
        retrieved_docs: List of retrieved document dictionaries
        
    Returns:
        Formatted string for inclusion in prompt
    """
    if not retrieved_docs:
        return "No documents retrieved."
    
    context_parts = []
    for doc in retrieved_docs:
        doc_text = f"[{doc['id']}] {doc['title']}\n{doc['content']}"
        # Truncate long documents to prevent token overflow
        if len(doc_text) > 1500:
            doc_text = doc_text[:1500] + "..."
        context_parts.append(doc_text)
    
    return "\n\n".join(context_parts)

# All summarization functions removed - using direct truncated responses instead

# Summary loading functions removed - only using direct response loading now

def load_all_round_responses(qid, log_dir, role1, role2):
    """
    Load all responses for both agents across all rounds.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Dict with keys: 'round1_role1', 'round1_role2', 'round2_role1', 'round2_role2'
        Each containing the full response text
    """
    responses = {}
    
    # Load Round 1 responses (full, not truncated)
    r1_response1, r1_response2 = load_round1_responses_full(qid, log_dir, role1, role2)
    responses['round1_role1'] = r1_response1
    responses['round1_role2'] = r1_response2
    
    # Load Round 2 responses (full, not truncated) 
    r2_response1, r2_response2 = load_round2_responses_full(qid, log_dir, role1, role2)
    responses['round2_role1'] = r2_response1
    responses['round2_role2'] = r2_response2
    
    return responses

def load_round2_responses_full(qid, log_dir, role1, role2):
    """
    Load full Round 2 responses (not truncated).
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Tuple (response1, response2) with full response texts
    """
    try:
        # Load Round 2 log file
        round2_file = f"{log_dir}/question_{qid}_round2.jsonl"
        if not os.path.exists(round2_file):
            return None, None
            
        response1, response2 = None, None
        
        with open(round2_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('agent') == role1:
                        response1 = data.get('response', '')
                    elif data.get('agent') == role2:
                        response2 = data.get('response', '')
                except json.JSONDecodeError:
                    continue
        
        return response1, response2
        
    except Exception as e:
        print(f"[WARNING] Failed to load Round 2 full responses: {e}")
        return None, None

def load_round1_responses_full(qid, log_dir, role1, role2):
    """
    Load full Round 1 responses (not truncated).
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Tuple (response1, response2) with full response texts
    """
    try:
        # Load Round 1 log file
        round1_file = f"{log_dir}/question_{qid}_round1.jsonl"
        if not os.path.exists(round1_file):
            return None, None
            
        response1, response2 = None, None
        
        with open(round1_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('agent') == role1:
                        response1 = data.get('response', '')
                    elif data.get('agent') == role2:
                        response2 = data.get('response', '')
                except json.JSONDecodeError:
                    continue
        
        return response1, response2
        
    except Exception as e:
        print(f"[WARNING] Failed to load Round 1 full responses: {e}")
        return None, None

def load_round_responses(qid, log_dir, role1, role2, round_num):
    """
    Load responses from specified round and truncate them before the parsed answer.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        round_num: Round number (1 or 2)
        
    Returns:
        For round 1: Tuple of (role1_truncated_response, role1_answer, role2_truncated_response, role2_answer) 
        For round 2: Tuple of (role1_truncated_response, role2_truncated_response, None, None)
        Returns (None, None, None, None) if not found
    """
    try:
        response1_file = f"{log_dir}/{qid}__{role1}__round_{round_num}_response.json"
        response2_file = f"{log_dir}/{qid}__{role2}__round_{round_num}_response.json"
        
        response1_data = None
        response2_data = None
        
        if os.path.exists(response1_file):
            with open(response1_file, 'r') as f:
                response1_data = json.load(f)
                
        if os.path.exists(response2_file):
            with open(response2_file, 'r') as f:
                response2_data = json.load(f)
        
        # Truncate responses before parsed answers
        truncated1 = None
        truncated2 = None
        answer1 = None
        answer2 = None
        
        if response1_data:
            thinking = response1_data.get('step_by_step_thinking', '')
            if round_num == 1:
                answer1 = response1_data.get('answer_choice', 'NO_ANSWER')
            # Truncate before common answer patterns
            for pattern in [r'Answer:\s*[ABCD]', r'The answer is\s*[ABCD]', r'\\boxed\{[ABCD]\}', r'Final answer:\s*[ABCD]', r'My answer:\s*[ABCD]']:
                import re
                match = re.search(pattern, thinking, re.IGNORECASE)
                if match:
                    thinking = thinking[:match.start()].strip()
                    break
            truncated1 = thinking.strip()
            
        if response2_data:
            thinking = response2_data.get('step_by_step_thinking', '')
            if round_num == 1:
                answer2 = response2_data.get('answer_choice', 'NO_ANSWER')
            # Truncate before common answer patterns
            for pattern in [r'Answer:\s*[ABCD]', r'The answer is\s*[ABCD]', r'\\boxed\{[ABCD]\}', r'Final answer:\s*[ABCD]', r'My answer:\s*[ABCD]']:
                import re
                match = re.search(pattern, thinking, re.IGNORECASE)
                if match:
                    thinking = thinking[:match.start()].strip()
                    break
            truncated2 = thinking.strip()
        
        if round_num == 1:
            return truncated1, answer1, truncated2, answer2
        else:  # round_num == 2
            return truncated1, truncated2, None, None
        
    except Exception as e:
        print(f"[WARNING] Failed to load Round {round_num} responses: {e}")
        return None, None, None, None

# Convenience functions for backward compatibility
def load_round1_responses(qid, log_dir, role1, role2):
    """Load Round 1 responses - convenience wrapper."""
    return load_round_responses(qid, log_dir, role1, role2, round_num=1)

def load_round2_responses(qid, log_dir, role1, role2):
    """Load Round 2 responses - convenience wrapper.""" 
    truncated1, truncated2, _, _ = load_round_responses(qid, log_dir, role1, role2, round_num=2)
    return truncated1, truncated2

# ============================================================================
# UNIFIED RETRIEVAL WITH INTELLIGENT DOCUMENT SELECTION
# ============================================================================

def unified_retrieval(medrag, question, options, k, qid, log_dir):
    """
    Unified retrieval that both agents share, with intelligent document selection.
    
    Args:
        medrag: MedRAG instance
        question: Main question
        options: Answer options dict
        k: Number of snippets to retrieve per agent (total 2*k retrieved)
        qid: Question ID for logging
        log_dir: Directory for logs
    
    Returns:
        (analyst_snippets, skeptic_snippets): Tuple of snippet lists for each agent
    """
    print(f"\n[UNIFIED] Retrieving evidence for both agents...")
    
    if not medrag.rag:
        print("  CoT mode: No retrieval needed")
        return [], []
    
    try:
        # Use MedRAG's optimized retrieval to get larger pool
        total_k = k * 2  # Get enough for both agents
        print(f"  Retrieving {total_k} documents from MedCorp...")
        
        # Use MedRAG's medrag_answer_by_source method for optimized retrieval
        options_text = "\n".join(f"{opt}: {val}" for opt, val in (options or {}).items()) if options else ""
        query_with_options = f"{question}\n\nOptions:\n{options_text}" if options_text else question
        
        # Get retrieved documents and scores
        _, all_snippets, all_scores = medrag.medrag_answer_by_source(
            query_with_options, 
            options=options,
            k=total_k,
            save_dir=f"{log_dir}/unified_retrieval_{qid}"
        )
        
        if not all_snippets:
            print("  No documents retrieved")
            return [], []
        
        print(f"  Retrieved {len(all_snippets)} total documents")
        
        # Intelligent document selection
        analyst_snippets = []
        skeptic_snippets = []
        
        # Analyst gets top-ranked general medical documents (first half)
        analyst_snippets = all_snippets[:k]
        
        # Skeptic gets remaining documents (potentially more specialized/contradictory)
        skeptic_snippets = all_snippets[k:k*2] if len(all_snippets) > k else []
        
        # If we don't have enough for skeptic, give them some overlap with different focus
        if len(skeptic_snippets) < k//2:
            # Give skeptic the middle portion of documents (different perspective)
            start_idx = max(0, k//2)
            end_idx = min(len(all_snippets), start_idx + k)
            skeptic_snippets = all_snippets[start_idx:end_idx]
        
        # Log retrieval results
        save_json({
            'total_retrieved': len(all_snippets),
            'analyst_count': len(analyst_snippets), 
            'skeptic_count': len(skeptic_snippets),
            'analyst_snippets': analyst_snippets,
            'skeptic_snippets': skeptic_snippets
        }, f"{log_dir}/{qid}__unified_retrieval.json")
        
        print(f"  [ANALYST] Assigned {len(analyst_snippets)} top-ranked documents")
        print(f"  [SKEPTIC] Assigned {len(skeptic_snippets)} alternative documents")
        
        return analyst_snippets, skeptic_snippets
        
    except Exception as e:
        print(f"  Error in unified retrieval: {e}")
        import traceback
        traceback.print_exc()
        return [], []

# ============================================================================
# TOOL-CALLING AGENT TURN
# ============================================================================

def agent_turn_with_tools(llm, role, question, options, tools, qid, log_dir, mode="rag", debate_history=None):
    """
    Execute agent turn using tool-calling approach.
    
    Args:
        llm: VLLMWrapper instance
        role: "agent1"/"agent2" (CoT mode) or "analyst"/"skeptic" (RAG mode)
        question: Medical question
        options: Answer options dict
        tools: Available tools dictionary
        qid: Question ID
        log_dir: Log directory
        mode: "rag" or "cot" 
        debate_history: Previous debate messages
        
    Returns:
        Agent response dictionary
    """
    print(f"\n--- {role.upper()} TURN (Tool-Calling) ---")
    
    # Use appropriate prompts based on mode and debate history
    if mode == "cot":
        # CoT mode: Use different prompts based on whether we have debate history
        if debate_history:
            # Debate round with previous discussion
            cot_debate_prompts = {
                "agent1": AGENT_COT_DEBATE_SYS,
                "agent2": AGENT_COT_DEBATE_SYS,
            }
            system_prompt = cot_debate_prompts.get(role, AGENT_COT_DEBATE_SYS)
        else:
            # Initial round without previous discussion
            cot_initial_prompts = {
                "agent1": AGENT_COT_SYS,
                "agent2": AGENT_COT_SYS,
            }
            system_prompt = cot_initial_prompts.get(role, AGENT_COT_SYS)
    else:
        # RAG mode: Use role-specific prompts with tool-calling
        rag_prompts = {
            "agent1": ANALYST_SYS,
            "agent2": SKEPTIC_SYS
        }
        system_prompt = rag_prompts.get(role, ANALYST_SYS)
    
    # Format options
    options_text = "\n".join(f"{opt}: {val}" for opt, val in (options or {}).items()) if options else ""
    
    # Format debate history for context
    history_text = ""
    if debate_history:
        history_text = "\n## Previous Discussion:\n"
        for entry in debate_history[-2:]:  # Last 2 entries for context
            if isinstance(entry, dict):
                history_text += f"{entry.get('role', 'Unknown')}: {entry.get('message', '')}\n"
            else:
                history_text += f"{str(entry)}\n"
        history_text += "\n"
    
    # Build initial prompt for tool calling
    if mode == "cot":
        # In CoT mode, skip tool calling and use simple direct reasoning
        prompt = f"""{system_prompt}

Question: {question}

Options:
{options_text}

{history_text}

Provide your analysis and reasoning:"""
        
        # Generate direct response without tools
        try:
            generation_start = time.time()
            
            # CoT mode: Use different temperatures for each agent
            # Agent1: temperature 0.7, Agent2: temperature 0.3
            agent_temperature = 0.7 if role == "agent1" else 0.3
            response = llm(
                prompt, 
                max_length=8192,  # Increased from 2048 to allow longer reasoning
                temperature=agent_temperature,  # Different temperatures per agent
                return_format='string',  # Get plain string
                stop_sequences=["<|im_end|>", "</s>", "###", "\n\n\n\n"],  # Removed aggressive boxed stops
                repetition_penalty=1.2
            )
            
            generation_time = time.time() - generation_start
            print(f"[TIMING] {role.upper()} CoT generation: {generation_time:.2f}s")
            
            # Parse response using MedRAG's proven parser
            parsed = parse_into_dict(response, model_name=HF_MODEL_NAME, question_id=qid, log_dir=log_dir)
            parsed["raw_response"] = response
            parsed["generation_time"] = generation_time
            parsed["mode"] = "cot"
            
            # Save response - Note: This will be overwritten in multi-round debates
            # The complete debate history is saved in debate_question()
            save_json(parsed, f"{log_dir}/{qid}__{role}__latest_response.json")
            
            print(f"[{role.upper()}] CoT Answer: {parsed.get('answer_choice', 'NONE')}")
            return parsed
            
        except Exception as e:
            print(f"[ERROR] {role} CoT generation failed: {e}")
            return {
                "step_by_step_thinking": f"CoT Error: {e}",
                "answer_choice": "PARSE_FAILED",
                "raw_response": "",
                "generation_time": 0.0,
                "mode": "cot"
            }
    
    # RAG mode with tool calling
    prompt = f"""{system_prompt}

Question: {question}

Options:
{options_text}

{history_text}

Start by calling the retrieve tool to gather relevant evidence, then provide your analysis:"""
    
    total_start_time = time.time()
    
    try:
        # Step 1: Generate initial response with tool call
        print(f"[{role.upper()}] Step 1: Generating tool call...")
        generation_start = time.time()
        
        # Generate with stop sequences and repetition penalty
        response = llm(
            prompt,
            max_length=8192,  # Smaller for initial tool call
            temperature=0.7,
            return_format='string',
            repetition_penalty=1.2
        )
        
        tool_call_time = time.time() - generation_start
        print(f"[TIMING] {role.upper()} tool call generation: {tool_call_time:.2f}s")
        
        # Step 2: Parse tool call
        tool_name, query = parse_tool_call(response)
        
        if tool_name == "retrieve" and query:
            print(f"[{role.upper()}] Tool call parsed: retrieve('{query}')")
            
            # Step 3: Execute tool call
            retrieval_start = time.time()
            retrieved_docs = execute_tool_call(tool_name, query, tools, qid=qid, log_dir=log_dir)
            retrieval_time = time.time() - retrieval_start
            print(f"[TIMING] {role.upper()} retrieval: {retrieval_time:.2f}s")
            
            # Step 4: Format retrieved docs for context
            docs_context = format_retrieved_docs_for_context(retrieved_docs)
            
            # Step 5: Generate full reasoning with retrieved context
            print(f"[{role.upper()}] Step 2: Generating reasoning with retrieved evidence...")
            
            # Build reasoning prompt with retrieved evidence
            reasoning_prompt = f"""{system_prompt}

Question: {question}

Options:
{options_text}

{history_text}

You called: retrieve("{query}")

Retrieved Evidence:
{docs_context}

Now provide your complete analysis based on the retrieved evidence:"""

            reasoning_start = time.time()
            
            # Generate reasoning with retrieved context
            reasoning_response = llm(
                reasoning_prompt,
                max_length=8192,  # Larger for full reasoning
                temperature=0.7,
                return_format='string',  
                stop_sequences=["<|im_end|>", "</s>", "###", "\n\n\n\n"],  # Removed aggressive boxed stops
                repetition_penalty=1.2
            )
            
            reasoning_time = time.time() - reasoning_start
            print(f"[TIMING] {role.upper()} reasoning generation: {reasoning_time:.2f}s")
            
            # Clean repetitive patterns from reasoning response
            reasoning_response = clean_repetitive_response(reasoning_response, max_length=2000)
            print(f"[DEBUG] {role.upper()} reasoning response length after cleaning: {len(reasoning_response)}")
            
            # Combine tool call and reasoning
            full_response = f"Tool Call: retrieve(\"{query}\")\n\nRetrieved Documents: {len(retrieved_docs)} documents\n\n{reasoning_response}"
            
            total_generation_time = tool_call_time + reasoning_time
            
        else:
            print(f"[{role.upper()}] No valid tool call found, using direct reasoning")
            # No tool call found, treat as direct reasoning - also clean it
            response = clean_repetitive_response(response, max_length=2000)
            full_response = response
            total_generation_time = tool_call_time
            retrieval_time = 0.0
            retrieved_docs = []
        
        total_time = time.time() - total_start_time
        print(f"[TIMING] {role.upper()} total turn: {total_time:.2f}s")
        
        # Parse final response using MedRAG's proven parser
        parsed = parse_into_dict(full_response, model_name=HF_MODEL_NAME, question_id=qid, log_dir=log_dir)
        parsed["raw_response"] = full_response
        parsed["generation_time"] = total_generation_time
        parsed["retrieval_time"] = retrieval_time
        parsed["total_time"] = total_time
        parsed["mode"] = "rag_with_tools"
        parsed["retrieved_docs"] = len(retrieved_docs)
        
        # Save response - Note: This will be overwritten in multi-round debates  
        # The complete debate history is saved in debate_question()
        save_json(parsed, f"{log_dir}/{qid}__{role}__latest_response.json")
        
        print(f"[{role.upper()}] Tool-based Answer: {parsed.get('answer_choice', 'NONE')}")
        return parsed
        
    except Exception as e:
        print(f"[ERROR] {role} tool-calling failed: {e}")
        total_time = time.time() - total_start_time
        return {
            "step_by_step_thinking": f"Tool-calling error: {e}",
            "answer_choice": "PARSE_FAILED",
            "raw_response": "",
            "generation_time": 0.0,
            "total_time": total_time,
            "mode": "rag_with_tools"
        }

def calculate_majority_vote_from_responses(all_responses, role1, role2):
    """
    Calculate majority vote from all round responses.
    
    Args:
        all_responses: Dict with round responses
        role1, role2: Agent role names
        
    Returns:
        Tuple of (majority_answer, vote_breakdown_dict)
    """
    # Collect all valid answers from all rounds
    all_answers = []
    vote_breakdown = {}
    
    # Round 1 answers - parse from response text
    if all_responses.get('round1_role1'):
        answer = parse_answer_from_text(all_responses['round1_role1'])
        if answer and answer in ["A", "B", "C", "D"]:
            all_answers.append(answer)
            vote_breakdown[f'{role1}_round1'] = answer
        
    if all_responses.get('round1_role2'):
        answer = parse_answer_from_text(all_responses['round1_role2'])
        if answer and answer in ["A", "B", "C", "D"]:
            all_answers.append(answer)
            vote_breakdown[f'{role2}_round1'] = answer
    
    # Round 2 answers - parse from response text
    if all_responses.get('round2_role1'):
        answer = parse_answer_from_text(all_responses['round2_role1'])
        if answer and answer in ["A", "B", "C", "D"]:
            all_answers.append(answer)
            vote_breakdown[f'{role1}_round2'] = answer
        
    if all_responses.get('round2_role2'):
        answer = parse_answer_from_text(all_responses['round2_role2'])
        if answer and answer in ["A", "B", "C", "D"]:
            all_answers.append(answer)
            vote_breakdown[f'{role2}_round2'] = answer
    
    if not all_answers:
        return None, vote_breakdown
    
    # Count votes
    vote_counts = {}
    for answer in all_answers:
        vote_counts[answer] = vote_counts.get(answer, 0) + 1
    
    # Find majority (or plurality if no majority)
    max_votes = max(vote_counts.values())
    majority_candidates = [answer for answer, count in vote_counts.items() if count == max_votes]
    
    # If tie, prefer more recent answers (Round 2 over Round 1)
    if len(majority_candidates) > 1:
        # Check Round 2 answers first
        round2_answers = []
        if all_responses.get('round2_role1'):
            r2_answer1 = parse_answer_from_text(all_responses['round2_role1'])
            if r2_answer1 and r2_answer1 in majority_candidates:
                round2_answers.append(r2_answer1)
        if all_responses.get('round2_role2'):
            r2_answer2 = parse_answer_from_text(all_responses['round2_role2'])
            if r2_answer2 and r2_answer2 in majority_candidates:
                round2_answers.append(r2_answer2)
        
        if round2_answers:
            # Use the first Round 2 answer among tied candidates
            majority_answer = round2_answers[0]
        else:
            # Fallback to first tied candidate
            majority_answer = majority_candidates[0]
    else:
        majority_answer = majority_candidates[0]
    
    return majority_answer, vote_breakdown

# ============================================================================
# DEBATE ORCHESTRATION
# ============================================================================

def role_based_verification(llm, agent1_response, agent2_response, question, options, qid, log_dir, mode="rag", role1="agent1", role2="agent2"):
    """
    Role-based consensus verification using Round 1 summaries when available.
    Supports both analyst/skeptic (RAG) and agent1/agent2 (CoT) naming schemes.
    """
    print(f"\n--- ROLE-BASED VERIFICATION ---")
    
    # Add timing for verification process
    verification_start = time.time()
    
    # Determine display names based on mode
    if mode == "cot":
        role1_display, role2_display = "Agent1", "Agent2"
    else:
        role1_display, role2_display = "Analyst", "Skeptic"
    
    # Load all round responses for majority vote calculations
    all_responses = load_all_round_responses(qid, log_dir, role1, role2)
    
    # Load Round 2 responses directly (no summaries for verification)
    round2_response1, round2_response2 = load_round2_responses(qid, log_dir, role1, role2)
    
    # Build debate summary using Round 2 responses directly
    if round2_response1 and round2_response2:
        # Use Round 2 responses directly, truncated before parsed answers
        debate_summary = (
            f"{role1_display}: {round2_response1}\n\n"
            f"{role2_display}: {round2_response2}"
        )
        
        # Get final answers from current responses for consensus check
        agent1_choice = agent1_response.get('answer_choice', 'NONE')
        agent2_choice = agent2_response.get('answer_choice', 'NONE')
        
        print("[VERIFICATION] Using Round 2 responses directly")
        
    else:
        print("[VERIFICATION] Round 2 responses not found, using current responses")
        # Fallback: Use current response summaries (less efficient but still works)
        # Create simple summaries for verification
        agent1_choice = agent1_response.get('answer_choice', 'NONE')
        agent2_choice = agent2_response.get('answer_choice', 'NONE')
        
        # Simple summary without LLM calls for efficiency
        agent1_text = agent1_response.get('step_by_step_thinking', '')[:1000] + "..."
        agent2_text = agent2_response.get('step_by_step_thinking', '')[:1000] + "..."
        
        debate_summary = (
            f"{role1_display}: {agent1_text} Answer: {agent1_choice}\n\n"
            f"{role2_display}: {agent2_text} Answer: {agent2_choice}"
        )
    
    # Simple consensus check first
    if (agent1_choice == agent2_choice and 
        agent1_choice in ["A", "B", "C", "D"]):
        total_verification_time = time.time() - verification_start
        print(f"[TIMING] Quick consensus verification: {total_verification_time:.2f}s")
        return {
            "step_by_step_thinking": f"Consensus reached: Both {role1_display} and {role2_display} agree on {agent1_choice}.",
            "answer_choice": agent1_choice,
            "consensus_type": "direct_agreement",
            "total_verification_time": total_verification_time
        }

    # Create verification prompt with clear format guidance (minimal changes to existing prompt)
    verification_system = """You are a medical expert verifier. Review the debate and provide your analysis followed by the final answer."""

    verification_prompt = f"""Review this complete medical debate and determine the best answer based on medical evidence:

Question: {question}

Options:
{chr(10).join(f"{k}: {v}" for k, v in (options or {}).items())}

Complete Debate Summary:
{debate_summary[:3500]}

Based on the full debate progression across all rounds, which answer is most appropriate?

Provide your reasoning and answer:"""

    try:
        # Use constrained generation for verification with aggressive stop sequences
        max_total_length = 8192  # Reduced for better control
        min_generation_tokens = 50  # Reduced minimum
        
        if hasattr(llm, 'tokenizer'):
            full_prompt = f"{verification_system}\n\n{verification_prompt}"
            prompt_tokens = len(llm.tokenizer.encode(full_prompt))
        else:
            full_prompt = f"{verification_system}\n\n{verification_prompt}"
            prompt_tokens = len(full_prompt) // 4
        
        available_tokens = max_total_length - prompt_tokens
        if available_tokens < min_generation_tokens:
            print(f"[WARNING] Verification prompt too long, using majority vote as fallback")
            # Try majority vote first
            majority_answer, vote_breakdown = calculate_majority_vote_from_responses(all_responses, role1, role2)
            total_verification_time = time.time() - verification_start
            
            if majority_answer:
                print(f"[VERIFICATION] Using majority vote: {majority_answer}, breakdown: {vote_breakdown}")
                return {
                    "step_by_step_thinking": f"Verification prompt too long. Using majority vote as fallback.",
                    "answer_choice": majority_answer,
                    "consensus_type": "prompt_too_long_majority",
                    "total_verification_time": total_verification_time
                }
            else:
                # Final fallback: use first agent answer if no majority
                fallback_answer = agent1_choice if agent1_choice in ["A", "B", "C", "D"] else agent2_choice
                print(f"[VERIFICATION] No majority found, using {role1} fallback: {fallback_answer}")
                return {
                    "step_by_step_thinking": f"Verification prompt too long and no majority vote. Using {role1} answer as fallback.",
                    "answer_choice": fallback_answer,
                    "consensus_type": f"prompt_too_long_fallback_to_{role1}",
                    "total_verification_time": total_verification_time
                }
                
        # Time the verification generation with enhanced stop sequences
        verification_gen_start = time.time()
        
        # Less aggressive stop sequences for verification to allow complete responses
        verification_stops = [
            "<|im_end|>", "</s>", "\n\n\n\n",  # Basic model stop tokens - removed ### which was stopping too early
            "\n---", "---\n"  # Only obvious end markers
        ]
        
        # Use system message if available in VLLMWrapper
        if hasattr(llm, 'generate_with_system'):
            verification = llm.generate_with_system(
                verification_system, 
                verification_prompt, 
                max_new_tokens=8192,  # Use correct parameter name with reasonable limit
                temperature=0.7,
                stop_sequences=verification_stops,
            )
        else:
            # Fallback to combined prompt
            combined_prompt = f"{verification_system}\n\n{verification_prompt}"
            verification = llm(combined_prompt, max_length=max_total_length, temperature=0.7)
        
        verification_gen_time = time.time() - verification_gen_start
        print(f"[TIMING] Verification generation: {verification_gen_time:.2f}s")
        
        # Clean response and detect repetition
        if isinstance(verification, list):
            verification = verification[0] if verification else ""
            if isinstance(verification, dict):
                verification = verification.get('generated_text', str(verification))
        elif isinstance(verification, dict):
            verification = verification.get('generated_text', str(verification))
        elif not isinstance(verification, str):
            verification = str(verification)
        
        # Remove the original prompt if it appears in the response
        if verification_system in verification:
            verification = verification.replace(verification_system, "").strip()
        if verification_prompt in verification:
            verification = verification.replace(verification_prompt, "").strip()
        
        # Aggressive repetition detection and cleanup for verification
        verification = clean_repetitive_response(verification, max_length=1000)
                
        # Parse verification using our improved sequential parser
        final_answer, answer_position = parse_answer_from_text(verification)
        print(f"[DEBUG] Verification parsing result: '{final_answer}' (type: {type(final_answer)})")
        
        # Validate answer and use improved fallback if needed
        if final_answer == "PARSE_FAILED" or final_answer not in ["A", "B", "C", "D"]:
            # Try majority vote across all rounds first
            majority_answer, vote_breakdown = calculate_majority_vote_from_responses(all_responses, role1, role2)
            
            if majority_answer:
                final_answer = majority_answer
                verification_type = "fallback_majority_vote"
                print(f"[VERIFICATION] Using majority vote: {majority_answer}, breakdown: {vote_breakdown}")
            else:
                # Final fallback: use agent1 answer if available
                final_answer = agent1_choice if agent1_choice in ["A", "B", "C", "D"] else agent2_choice
                verification_type = f"fallback_to_{role1}"
                print(f"[VERIFICATION] No majority found, using {role1} fallback: {final_answer}")
        else:
            verification_type = "verification_consensus"
            print(f"[VERIFICATION] Using verification result: {final_answer}")
        
        # Calculate total verification time
        total_verification_time = time.time() - verification_start
        print(f"[TIMING] Total verification: {total_verification_time:.2f}s")
        
        # Log verification
        save_json({
            "verification_prompt": verification_prompt,
            "verification_response": verification,
            "final_answer": final_answer,
            "consensus_type": verification_type,
            "verification_gen_time": verification_gen_time,
            "total_verification_time": total_verification_time
        }, f"{log_dir}/{qid}__verification.json")
        
        result = {
            "step_by_step_thinking": f"Verification resolved disagreement: {verification}",
            "answer_choice": final_answer,
            "verification_reasoning": verification,
            "consensus_type": verification_type,
            "verification_gen_time": verification_gen_time,
            "total_verification_time": total_verification_time
        }
        
        print(f"[VERIFICATION] Final Answer: {result['answer_choice']}")
        return result
        
    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        # Try majority vote first
        try:
            majority_answer, vote_breakdown = calculate_majority_vote_from_responses(all_responses, role1, role2)
            if majority_answer:
                total_verification_time = time.time() - verification_start
                print(f"[VERIFICATION] Using majority vote after error: {majority_answer}, breakdown: {vote_breakdown}")
                return {
                    "step_by_step_thinking": f"Verification failed: {e}. Using majority vote as fallback.",
                    "answer_choice": majority_answer,
                    "consensus_type": "error_fallback_majority",
                    "total_verification_time": total_verification_time
                }
        except Exception as fallback_error:
            print(f"ERROR: Majority vote fallback also failed: {fallback_error}")
        
        # Final fallback: use first agent answer
        fallback_answer = agent1_choice if agent1_choice in ["A", "B", "C", "D"] else agent2_choice
        total_verification_time = time.time() - verification_start
        print(f"[VERIFICATION] Using final fallback: {fallback_answer}")
        return {
            "step_by_step_thinking": f"Verification failed: {e}. Majority vote also failed. Using {role1} answer as final fallback.",
            "answer_choice": fallback_answer,
            "consensus_type": "error_final_fallback",
            "total_verification_time": total_verification_time
        }

def debate_question(medrag, llm, qid, question, options, k, log_dir, mode="cot", rounds=MAX_ROUNDS):
    """
    Run sequential debate with optimized Round 1 summarization.
    
    Returns:
        Final consensus answer and comprehensive timing metrics
    """
    print(f"\n{'='*80}")
    print(f"QUESTION {qid}")
    print(f"{'='*80}")
    
    debate_start_time = time.time()
    
    # Create tools for agents
    tools = {}
    if mode == "rag":
        retrieval_tool = create_retrieval_tool(medrag, k=k, rrf_k=RRF_K)
        tools = {"retrieve": retrieval_tool}
    
    history = []
    
    # Determine role names based on mode
    if mode == "cot":
        role1, role2 = "agent1", "agent2"
        role1_display, role2_display = "Agent1", "Agent2"
    else:
        role1, role2 = "analyst", "skeptic"
        role1_display, role2_display = "Analyst", "Skeptic"
    
    # Multi-round sequential debate
    for round_num in range(1, rounds + 1):
        print(f"\n--- ROUND {round_num} ---")
        
        # First agent turn
        try:
            # For round 2, use truncated Round 1 responses directly (no summaries)
            recent_agent2 = None
            if round_num == 2:
                # Load Round 1 responses - Agent1 gets only Agent2's response
                truncated1, answer1, truncated2, answer2 = load_round1_responses(qid, log_dir, role1, role2)
                recent_agent2 = []
                
                # Add only Agent2's Round 1 response for Agent1 to consider
                if truncated2 and answer2:
                    recent_agent2.append({
                        "role": role2,
                        "message": f"{role2_display} reasoned (Round 1): {truncated2}\n\n{role2_display}'s answer: {answer2}. Do you agree or disagree? Provide your analysis.",
                        "response_type": "round1_other_truncated"
                    })
                
                # If no responses available, set to None
                if not recent_agent2:
                    recent_agent2 = None
                    
                # Log Round 2 truncated response usage for Agent1
                if truncated2 and answer2:
                    response_usage_log = {
                        'qid': qid,
                        'round': 2,
                        'agent': role1,
                        'uses_response_from': 'other_agent',
                        'truncated2_available': bool(truncated2),
                        'answer2_available': bool(answer2),
                        'timestamp': time.time()
                    }
                    response_usage_jsonl_file = f"{log_dir}/response_usage_logs.jsonl"
                    with open(response_usage_jsonl_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(response_usage_log, ensure_ascii=False) + '\n')
            
            agent1_response = agent_turn_with_tools(
                llm, role1, question, options, tools, qid, log_dir,
                mode=mode, debate_history=recent_agent2
            )
            
            # Save round-specific response for complete logging
            save_json(agent1_response, f"{log_dir}/{qid}__{role1}__round_{round_num}_response.json")
            
            history.append({
                "role": role1, 
                "round": round_num, 
                "message": f"{role1_display} Round {round_num}: {agent1_response.get('step_by_step_thinking', '')}\nAnswer: {agent1_response.get('answer_choice', 'NO_ANSWER')}",
                "answer": agent1_response
            })
            
            # Summarization removed - using direct response truncation instead
            
        except Exception as e:
            print(f"ERROR: {role1_display} Round {round_num} failed: {e}")
            continue
        
        # Second agent turn
        try:
            current_agent1 = None
            if round_num == 1:
                # Round 1: Use full context for agent2
                if history:
                    last_agent1 = history[-1]
                    current_agent1 = [{
                        "role": role1,
                        "message": f"{role1_display} concluded: {last_agent1.get('answer', {}).get('step_by_step_thinking', '')}. Answer: {last_agent1.get('answer', {}).get('answer_choice', 'NO_ANSWER')}. Do you agree or disagree? Provide your analysis.",
                        "summary_type": "round1_full_context"
                    }]
            elif round_num == 2:
                # Round 2: Use truncated Round 1 responses directly - Agent2 gets only Agent1's response
                truncated1, answer1, truncated2, answer2 = load_round1_responses(qid, log_dir, role1, role2)
                current_agent1 = []
                
                # Add only Agent1's Round 1 response for Agent2 to consider
                if truncated1 and answer1:
                    current_agent1.append({
                        "role": role1,
                        "message": f"{role1_display} reasoned (Round 1): {truncated1}\n\n{role1_display}'s answer: {answer1}. Do you agree or disagree? Provide your analysis.",
                        "response_type": "round1_other_truncated"
                    })
                
                # If no responses available, set to None
                if not current_agent1:
                    current_agent1 = None
                    
                # Log Round 2 truncated response usage for Agent2
                if truncated1 and answer1:
                    response_usage_log = {
                        'qid': qid,
                        'round': 2,
                        'agent': role2,
                        'uses_response_from': 'other_agent',
                        'truncated1_available': bool(truncated1),
                        'answer1_available': bool(answer1),
                        'timestamp': time.time()
                    }
                    response_usage_jsonl_file = f"{log_dir}/response_usage_logs.jsonl"
                    with open(response_usage_jsonl_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(response_usage_log, ensure_ascii=False) + '\n')
            
            agent2_response = agent_turn_with_tools(
                llm, role2, question, options, tools, qid, log_dir,
                mode=mode, debate_history=current_agent1
            )
            
            # Save round-specific response for complete logging
            save_json(agent2_response, f"{log_dir}/{qid}__{role2}__round_{round_num}_response.json")
            
            history.append({
                "role": role2,
                "round": round_num,
                "message": f"{role2_display} Round {round_num}: {agent2_response.get('step_by_step_thinking', '')}\nAnswer: {agent2_response.get('answer_choice', 'NO_ANSWER')}",
                "answer": agent2_response
            })
            
            # Summarization removed - using direct response truncation instead
            
        except Exception as e:
            print(f"ERROR: {role2_display} Round {round_num} failed: {e}")
            continue
    
    # Role-based verification using Round 1 summaries when available
    if len(history) >= 2:
        final_agent1 = [h for h in history if h.get("role") == role1][-1]["answer"]
        final_agent2 = [h for h in history if h.get("role") == role2][-1]["answer"]
        
        final_answer = role_based_verification(
            llm, final_agent1, final_agent2, question, options, qid, log_dir, mode, role1, role2
        )
    elif len(history) == 1:
        # Only one response available
        final_answer = history[0]["answer"]
        print("[WARNING] Only one agent response available, using as final answer")
    else:
        # No valid responses
        print("[ERROR] No valid agent responses generated")
        final_answer = {
            "step_by_step_thinking": "No valid responses generated",
            "answer_choice": "PARSE_FAILED",
            "consensus_type": "no_responses"
        }
    
    # Calculate total debate time
    total_debate_time = time.time() - debate_start_time
    
    # Add timing information to final answer
    final_answer["total_debate_time"] = total_debate_time
    
    # Collect timing information from history
    timing_summary = {
        "total_debate_time": total_debate_time,
        f"{role1}_times": [],
        f"{role2}_times": [],
        "retrieval_times": []
    }
    
    for entry in history:
        role = entry.get("role")
        answer = entry.get("answer", {})
        
        if role == role1:
            timing_summary[f"{role1}_times"].append({
                "round": entry.get("round"),
                "generation_time": answer.get("generation_time", 0),
                "retrieval_time": answer.get("retrieval_time", 0),
                "total_time": answer.get("total_time", 0)
            })
        elif role == role2:
            timing_summary[f"{role2}_times"].append({
                "round": entry.get("round"),
                "generation_time": answer.get("generation_time", 0),
                "retrieval_time": answer.get("retrieval_time", 0),
                "total_time": answer.get("total_time", 0)
            })
    
    # Add verification timing if available
    timing_summary["verification_time"] = final_answer.get("total_verification_time", 0)
    
    print(f"\n[TIMING SUMMARY]")
    print(f"Total debate time: {total_debate_time:.2f}s")
    print(f"Verification consensus: {final_answer.get('consensus_type', 'N/A')}")
    
    # Save complete debate log with enhanced debugging information
    debate_log = {
        "qid": qid,
        "question": question,
        "options": options,
        "mode": mode,
        "rounds": rounds,
        "history": history,
        "final_answer": final_answer,
        "timing_summary": timing_summary,
        # Add detailed raw responses for debugging
        "debug_info": {
            "total_history_entries": len(history),
            "roles_participated": list(set([h.get("role") for h in history])),
            "rounds_completed": list(set([h.get("round") for h in history])),
            "raw_responses_preview": {
                role: [
                    {
                        "round": h.get("round"),
                        "raw_response_length": len(h.get("answer", {}).get("raw_response", "")),
                        "raw_response_preview": h.get("answer", {}).get("raw_response", "")[:500] + "..." if len(h.get("answer", {}).get("raw_response", "")) > 500 else h.get("answer", {}).get("raw_response", ""),
                        "parsed_answer": h.get("answer", {}).get("answer_choice", "NO_ANSWER"),
                        "generation_time": h.get("answer", {}).get("generation_time", 0)
                    }
                    for h in history if h.get("role") == role
                ]
                for role in [role1, role2]
            }
        }
    }
    
    save_json(debate_log, f"{log_dir}/{qid}__complete_debate.json")
        
    return final_answer

# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_debate_benchmark(dataset_name="mmlu", mode="cot", k=DEFAULT_K, log_dir="./debate_logs", split="test"):
    """
    Run debate-based evaluation on a MedQA dataset.
    
    Args:
        dataset_name: Name of dataset (mmlu, medqa, etc.)
        mode: "cot" (chain of thought, no retrieval) or "rag" (with retrieval)
        k: Number of snippets to retrieve per role (only used in rag mode)
        log_dir: Directory for logs
        split: Dataset split ("test" or "dev" for medmcqa)
    """
    print(f"\n{'='*80}")
    print(f"DEBATE BENCHMARK: {dataset_name} ({mode.upper()} mode)")
    print(f"{'='*80}\n")
    
    # Setup paths
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize MedRAG with optimized single-source setup
    print(f"Initializing MedRAG with optimized setup ({mode.upper()} mode)...")
    patch_medrag_for_vllm()
    
    if mode == "rag":
        print(f"  RAG enabled with single source: MedCorp, retriever: MedCPT, k={k}")
        print(f"  GPU assignment: FAISS on GPU {FAISS_GPU_ID}, VLLM on GPU {VLLM_GPU_ID}")
        medrag = MedRAG(
            llm_name=HF_MODEL_NAME,
            rag=True,
            retriever_name="MedCPT", 
            corpus_name="MedCorp",
            db_dir=MEDCORP_DIR,
            corpus_cache=True,
            HNSW=True
        )
    else:  # CoT mode
        print(f"  CoT mode: No retrieval, pure reasoning")
        medrag = MedRAG(
            llm_name=HF_MODEL_NAME,
            rag=False,
            retriever_name=None,
            corpus_name=None,
            db_dir=None,
            corpus_cache=False,
            HNSW=False
        )
    
    # Initialize shared LLM instance for all debate agents
    print(f"Loading shared VLLM instance: {HF_MODEL_NAME}")
    llm = VLLMWrapper(HF_MODEL_NAME)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    dataset = QADataset(dataset_name)
    print(f"Loaded {len(dataset)} questions\n")
    
    # Determine split (medmcqa uses "dev", others use "test")
    if split == "test" and dataset_name == "medmcqa":
        split = "dev"
    
    # Run debates
    results = []
    correct = 0
    
    for idx in range(len(dataset)):
        qdata = dataset[idx]
        # Use dataset.index[idx] for proper ID matching with evaluate.py
        question_id = dataset.index[idx]
        qid_for_file = f"{split}_{question_id}"  # Format: test_001, dev_123, etc.
        
        question = qdata["question"]
        options = qdata.get("options", {})
        gold_answer = qdata.get("answer")
        
        if not gold_answer:
            print(f"WARNING: Question {qid_for_file} has no gold answer, skipping")
            continue
        
        # Skip if already processed (resume capability)
        result_file = f"{log_dir}/{qid_for_file}.json"
        if os.path.exists(result_file):
            print(f"Skipping {qid_for_file} (already exists)")
            # Load existing result to count accuracy
            try:
                with open(result_file, 'r') as f:
                    existing = json.load(f)
                    if isinstance(existing, list) and len(existing) > 0:
                        predicted = existing[0].get("answer_choice")
                        is_correct = (predicted == gold_answer) if predicted in ["A", "B", "C", "D"] else False
                        if is_correct:
                            correct += 1
                        results.append({
                            "id": qid_for_file,
                            "question": question,
                            "gold": gold_answer,
                            "predicted": predicted,
                            "correct": is_correct,
                            "reasoning": existing[0].get("step_by_step_thinking", "")
                        })
            except Exception as e:
                print(f"  Warning: Could not load existing result: {e}")
            continue
        
        # Run debate
        try:
            final = debate_question(
                medrag, llm, qid_for_file, question, options, k, log_dir, mode=mode
            )
            
            predicted = final.get("answer_choice")
            is_correct = (predicted == gold_answer) if predicted in ["A", "B", "C", "D"] else False
            
            if is_correct:
                correct += 1
            
            # Extract timing information
            timing_info = final.get("debate_timing", {})
            
            results.append({
                "id": qid_for_file,
                "question": question,
                "gold": gold_answer,
                "predicted": predicted,
                "correct": is_correct,
                "reasoning": final.get("step_by_step_thinking", ""),
                "timing": timing_info
            })
            
            # Display timing summary
            total_time = timing_info.get("total_debate_time", 0.0)
            retrieval_time = timing_info.get("retrieval_time", 0.0) 
            analyst_time = timing_info.get("analyst_gen_time", 0.0)
            skeptic_time = timing_info.get("skeptic_gen_time", 0.0)
            verification_time = timing_info.get("verification_time", 0.0)
            
            print(f"Progress: {idx+1}/{len(dataset)} | Accuracy: {correct/(idx+1):.2%}")
            print(f"  Timing - Total: {total_time:.1f}s | Retrieval: {retrieval_time:.1f}s | Analyst: {analyst_time:.1f}s | Skeptic: {skeptic_time:.1f}s | Verification: {verification_time:.1f}s")
            
        except Exception as e:
            print(f"ERROR on question {qid_for_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    results_file = f"{log_dir}/{dataset_name}_results.json"
    save_json(results, results_file)
    
    # Print summary
    accuracy = correct / len(results) if results else 0
    
    # Calculate timing statistics
    total_times = [r.get("timing", {}).get("total_debate_time", 0.0) for r in results]
    retrieval_times = [r.get("timing", {}).get("retrieval_time", 0.0) for r in results]
    analyst_times = [r.get("timing", {}).get("analyst_gen_time", 0.0) for r in results]
    skeptic_times = [r.get("timing", {}).get("skeptic_gen_time", 0.0) for r in results]
    verification_times = [r.get("timing", {}).get("verification_time", 0.0) for r in results]
    
    avg_total = sum(total_times) / len(total_times) if total_times else 0
    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0  
    avg_analyst = sum(analyst_times) / len(analyst_times) if analyst_times else 0
    avg_skeptic = sum(skeptic_times) / len(skeptic_times) if skeptic_times else 0
    avg_verification = sum(verification_times) / len(verification_times) if verification_times else 0
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total Questions: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nTiming Summary (averages):")
    print(f"  Total per question: {avg_total:.1f}s")
    if mode == "rag":
        print(f"  Retrieval: {avg_retrieval:.1f}s")
    print(f"  Analyst generation: {avg_analyst:.1f}s") 
    print(f"  Skeptic generation: {avg_skeptic:.1f}s")
    print(f"  Verification: {avg_verification:.1f}s")
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*80}\n")
    
    return results

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multi-agent debate on MedQA datasets")
    parser.add_argument("--dataset", type=str, default="mmlu",
                       choices=["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"],
                       help="Dataset to evaluate on")
    parser.add_argument("--mode", type=str, default="cot",
                       choices=["cot", "rag"],
                       help="Mode: cot (no retrieval) or rag (with retrieval) (default: cot)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                       help="Number of snippets to retrieve per role (only used in rag mode)")
    parser.add_argument("--log_dir", type=str, default="./debate_logs",
                       help="Directory for debate logs")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS,
                       help="Number of debate rounds")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split (default: test, auto-switched to dev for medmcqa)")
    parser.add_argument("--corpus_name", type=str, default="MedCorp",
                       help="Name of corpus to use (default: MedCorp)")
    
    args = parser.parse_args()
    
    # Update global rounds if specified
    MAX_ROUNDS = args.rounds
    log_dir = os.path.join(args.log_dir, args.dataset, args.corpus_name, args.mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Run benchmark
    run_debate_benchmark(
        dataset_name=args.dataset,
        mode=args.mode,
        k=args.k,
        log_dir=args.log_dir,
        split=args.split
    )
