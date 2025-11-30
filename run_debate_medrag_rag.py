#!/usr/bin/env python3
"""
Multi-agent RAG debate system for MedQA using MedRAG retrieval.
Two agents (Agent1 & Agent2) retrieve evidence, debate, and a Judge decides.
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
os.environ['CUDA_VISIBLE_DEVICES'] = '3,5'

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

# RAG mode: Equal agent prompts with tool-calling (adapted from CoT)
AGENT_RAG_SYS = """You are a medical analyst participating in a clinical discussion with access to evidence retrieval.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Instructions:
1) Call retrieve("search query") to gather relevant medical evidence
2) Read the question and options carefully, then analyze the retrieved evidence.
3) Write at most 2 sentences of clinical reasoning based on the evidence
4) State your answer choice clearly using \\boxed{{letter}} format (for example \\boxed{{A}})

Provide your initial medical analysis with evidence support."""

# RAG mode: Round 1 prompts with pre-retrieved documents (original MedRAG style)
AGENT_RAG_ROUND1_SYS = """You are a medical analyst participating in a clinical discussion. You have been provided with relevant medical evidence.

Instructions:
1) Review the provided medical evidence carefully
2) Read the question and options, then analyze based on the evidence
3) Write at most 2 sentences of clinical reasoning based on the evidence
4) State your answer choice clearly using \\boxed{{letter}} format (for example \\boxed{{A}})

Provide your medical analysis based on the provided evidence."""

# RAG mode: Debate round prompts with tool-calling (with previous discussion context)
AGENT_RAG_DEBATE_SYS = """You are a medical AI assistant participating in a clinical discussion with access to evidence retrieval.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Instructions:
1) Call retrieve("search query") to gather additional relevant evidence
2) Review the previous conclusion, then analyze the retrieved evidence.
3) Write at most 2 sentences: briefly say whether you agree (add one new reason) or disagree (give one counterpoint) based on evidence
4) State your answer choice clearly using \\boxed{{letter}} format (for example \\boxed{{A}})

Consider the previous discussion when forming your response, but gather your own evidence."""



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

def clean_repetitive_response(text):
    """
    Clean repetitive text patterns that indicate model got stuck in loops
    """
    if not isinstance(text, str) or not text.strip():
        return text
    
    
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

def summarize_round1_agent_response(llm, agent_response, role, question, qid, log_dir):
    """
    Summarize Round 1 agent response to ~500 tokens for efficient Round 2 usage.
    
    Args:
        llm: VLLMWrapper instance
        agent_response: Dict with agent's full response
        role: Agent role identifier  
        question: Original question for context
        qid: Question ID for file naming
        log_dir: Directory to save summary
    
    Returns:
        Dict with 500-token summary and answer choice
    """
    return summarize_agent_response(llm, agent_response, role, question, qid, log_dir, round_num=1)

def summarize_round2_agent_response(llm, agent_response, role, question, qid, log_dir):
    """
    Summarize Round 2 agent response to ~500 tokens for efficient verification usage.
    
    Args:
        llm: VLLMWrapper instance
        agent_response: Dict with agent's full response
        role: Agent role identifier  
        question: Original question for context
        qid: Question ID for file naming
        log_dir: Directory to save summary
    
    Returns:
        Dict with 500-token summary and answer choice
    """
    return summarize_agent_response(llm, agent_response, role, question, qid, log_dir, round_num=2)

def summarize_agent_response(llm, agent_response, role, question, qid, log_dir, round_num):
    """
    Summarize agent response to ~500 tokens for efficient usage.
    
    Args:
        llm: VLLMWrapper instance
        agent_response: Dict with agent's full response
        role: Agent role identifier  
        question: Original question for context
        qid: Question ID for file naming
        log_dir: Directory to save summary
        round_num: Round number (1 or 2)
    
    Returns:
        Dict with 500-token summary and answer choice
    """
    full_reasoning = agent_response.get('step_by_step_thinking', '')
    answer_choice = agent_response.get('answer_choice', 'NO_ANSWER')
    
    # Use the existing efficient summarization prompt, targeting 500 tokens
    summarize_prompt = f"""Summarize ONLY the following agent's reasoning.

    Requirements:
    - Keep the summarization in 2 sentences.
    - Focus on the key clinical points relevant to the question.

Question: {question[:300]}

{role.title()}'s reasoning (Round {round_num}): {full_reasoning[:3500]}

Summary:"""

    try:
        # Use stronger stop sequences and lower temperature for summarization
        summary_response = llm(
            summarize_prompt, 
            max_length=1024, 
            temperature=0.1,  # Much lower temperature for focused summarization
            stop_sequences=["<|im_end|>", "</s>", "\n\n\n", "Agent3", "Round 3", "###"],  # Stop at any mention of other agents/rounds
            repetition_penalty=1.3
        )
        
        # Clean response
        if isinstance(summary_response, list):
            summary_response = summary_response[0] if summary_response else ""
            if isinstance(summary_response, dict):
                summary_response = summary_response.get("generated_text", str(summary_response))
        elif isinstance(summary_response, dict):
            summary_response = summary_response.get("generated_text", str(summary_response))
        elif not isinstance(summary_response, str):
            summary_response = str(summary_response)
        
        # Clean up if prompt is included
        if summarize_prompt in summary_response:
            summary_response = summary_response.replace(summarize_prompt, "").strip()
        
        # Ensure we have a reasonable summary
        if len(summary_response.strip()) < 50:
            # Fallback: truncation to ~500 tokens (2000 chars)
            target_chars = 2000
            if len(full_reasoning) > target_chars:
                first_part = full_reasoning[:int(target_chars * 0.7)]
                last_part = full_reasoning[-int(target_chars * 0.3):]
                summary_response = f"{first_part}...\n\n[Reasoning continues]\n\n...{last_part}"
            else:
                summary_response = full_reasoning
        
        # Create formatted summary with role and answer
        formatted_summary = f"{role} (Round {round_num} summarized): {summary_response.strip()}. Answer choice: {answer_choice}"
        
        summary_data = {
            'role': role,
            'round': round_num,
            'summary': summary_response.strip(),
            'answer_choice': answer_choice,
            'formatted_summary': formatted_summary,
            'reasoning_type': f'round{round_num}_summarized'
        }
        
        # Save summary to file for later use
        summary_file = f"{log_dir}/{qid}__round{round_num}__{role}__summary.json"
        save_json(summary_data, summary_file)
        
        # Log summarization to JSONL for analysis
        summarization_log = {
            'qid': qid,
            'role': role,
            'round': round_num,
            'original_length': len(full_reasoning),
            'summary_length': len(summary_response.strip()),
            'summarization_prompt_length': len(summarize_prompt),
            'answer_choice': answer_choice,
            'timestamp': time.time()
        }
        summarization_jsonl_file = f"{log_dir}/summarization_logs.jsonl"
        with open(summarization_jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summarization_log, ensure_ascii=False) + '\n')
        
        return summary_data
        
    except Exception as e:
        print(f"[WARNING] Round {round_num} summarization failed for {role}: {e}")
        # Fallback: truncate to 500 tokens
        target_chars = 2000  # ~500 tokens
        if len(full_reasoning) <= target_chars:
            fallback_summary = full_reasoning
        else:
            first_part = full_reasoning[:int(target_chars * 0.7)]
            last_part = full_reasoning[-int(target_chars * 0.3):]
            fallback_summary = f"{first_part}\n\n...[continued]...\n\n{last_part}"
        
        formatted_summary = f"{role} (Round {round_num} summarized): {fallback_summary}. Answer choice: {answer_choice}"
        
        summary_data = {
            'role': role,
            'round': round_num,
            'summary': fallback_summary,
            'answer_choice': answer_choice,
            'formatted_summary': formatted_summary,
            'reasoning_type': f'round{round_num}_fallback'
        }
        
        # Save fallback summary
        summary_file = f"{log_dir}/{qid}__round{round_num}__{role}__summary.json"
        save_json(summary_data, summary_file)
        
        # Log fallback summarization to JSONL for analysis
        summarization_log = {
            'qid': qid,
            'role': role,
            'round': round_num,
            'original_length': len(full_reasoning),
            'summary_length': len(fallback_summary),
            'summarization_type': 'fallback',
            'answer_choice': answer_choice,
            'timestamp': time.time()
        }
        summarization_jsonl_file = f"{log_dir}/summarization_logs.jsonl"
        with open(summarization_jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summarization_log, ensure_ascii=False) + '\n')
        
        return summary_data

def load_round1_summaries(qid, log_dir, role1, role2):
    """
    Load Round 1 summaries for both agents.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Tuple of (role1_summary_data, role2_summary_data) or (None, None) if not found
        Note: Summaries may not exist if agents had ERROR/PARSE_FAILED answers
    """
    return load_round_summaries(qid, log_dir, role1, role2, round_num=1)

def load_round2_summaries(qid, log_dir, role1, role2):
    """
    Load Round 2 summaries for both agents.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Tuple of (role1_summary_data, role2_summary_data) or (None, None) if not found
    """
    return load_round_summaries(qid, log_dir, role1, role2, round_num=2)

def load_round_summaries(qid, log_dir, role1, role2, round_num):
    """
    Load summaries for both agents for a specific round.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        round_num: Round number (1 or 2)
        
    Returns:
        Tuple of (role1_summary_data, role2_summary_data) or (None, None) if not found
    """
    try:
        summary1_file = f"{log_dir}/{qid}__round{round_num}__{role1}__summary.json"
        summary2_file = f"{log_dir}/{qid}__round{round_num}__{role2}__summary.json"
        
        summary1_data = None
        summary2_data = None
        
        if os.path.exists(summary1_file):
            with open(summary1_file, 'r') as f:
                summary1_data = json.load(f)
                
        if os.path.exists(summary2_file):
            with open(summary2_file, 'r') as f:
                summary2_data = json.load(f)
        
        return summary1_data, summary2_data
        
    except Exception as e:
        print(f"[WARNING] Failed to load Round {round_num} summaries: {e}")
        return None, None

def load_all_round_summaries(qid, log_dir, role1, role2):
    """
    Load all summaries for both agents across all rounds.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Dict with keys: 'round1_role1', 'round1_role2', 'round2_role1', 'round2_role2'
    """
    summaries = {}
    
    # Load Round 1 summaries
    r1_summary1, r1_summary2 = load_round1_summaries(qid, log_dir, role1, role2)
    summaries['round1_role1'] = r1_summary1
    summaries['round1_role2'] = r1_summary2
    
    # Load Round 2 summaries
    r2_summary1, r2_summary2 = load_round2_summaries(qid, log_dir, role1, role2)
    summaries['round2_role1'] = r2_summary1
    summaries['round2_role2'] = r2_summary2
    
    return summaries

def load_round2_responses(qid, log_dir, role1, role2):
    """
    Load Round 2 responses and truncate them before the parsed answer.
    
    Args:
        qid: Question ID
        log_dir: Log directory
        role1, role2: Agent role identifiers
        
    Returns:
        Tuple of (role1_truncated_response, role2_truncated_response) or (None, None) if not found
    """
    try:
        response1_file = f"{log_dir}/{qid}__{role1}__round_2_response.json"
        response2_file = f"{log_dir}/{qid}__{role2}__round_2_response.json"
        
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
        
        if response1_data:
            thinking = response1_data.get('step_by_step_thinking', '')
            # Truncate the reasoning text before the parsed answer
            truncated1 = thinking.strip()
            
        if response2_data:
            thinking = response2_data.get('step_by_step_thinking', '')
            # Truncate the reasoning text before the parsed answer
            truncated2 = thinking.strip()
        
        return truncated1, truncated2
        
    except Exception as e:
        print(f"[WARNING] Failed to load Round 2 responses: {e}")
        return None, None

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
        (agent1_snippets, agent2_snippets): Tuple of snippet lists for each agent
    """
    print(f"\n[UNIFIED] Retrieving evidence for both agents...")
    
    if not medrag.rag:
        print("  RAG mode: Tool-based retrieval enabled")
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
        agent1_snippets = []
        agent2_snippets = []
        
        # Agent1 gets top-ranked general medical documents (first half)
        agent1_snippets = all_snippets[:k]
        
        # Agent2 gets remaining documents (potentially more specialized/contradictory)
        agent2_snippets = all_snippets[k:k*2] if len(all_snippets) > k else []
        
        # If we don't have enough for agent2, give them some overlap with different focus
        if len(agent2_snippets) < k//2:
            # Give agent2 the middle portion of documents (different perspective)
            start_idx = max(0, k//2)
            end_idx = min(len(all_snippets), start_idx + k)
            agent2_snippets = all_snippets[start_idx:end_idx]
        
        # Log retrieval results
        save_json({
            'total_retrieved': len(all_snippets),
            'agent1_count': len(agent1_snippets), 
            'agent2_count': len(agent2_snippets),
            'agent1_snippets': agent1_snippets,
            'agent2_snippets': agent2_snippets
        }, f"{log_dir}/{qid}__unified_retrieval.json")
        
        print(f"  [AGENT1] Assigned {len(agent1_snippets)} top-ranked documents")
        print(f"  [AGENT2] Assigned {len(agent2_snippets)} alternative documents")
        
        return agent1_snippets, agent2_snippets
        
    except Exception as e:
        print(f"  Error in unified retrieval: {e}")
        import traceback
        traceback.print_exc()
        return [], []

# ============================================================================
# AGENT TURN FUNCTIONS
# ============================================================================

def agent_turn_with_predocs(llm, role, question, options, pre_retrieved_docs, qid, log_dir, debate_history=None):
    """
    Execute agent turn using pre-retrieved documents for Round 1 (original MedRAG style).
    
    Args:
        llm: VLLMWrapper instance
        role: "agent1" or "agent2"
        question: Medical question
        options: Answer options dict
        pre_retrieved_docs: List of pre-retrieved documents for this agent
        qid: Question ID
        log_dir: Log directory
        debate_history: Previous debate messages (for Agent2 in Round 1)
        
    Returns:
        Agent response dictionary
    """
    print(f"\n--- {role.upper()} TURN (Round 1 Pre-Retrieved) ---")
    
    # Use Round 1 system prompt for pre-retrieved documents
    system_prompt = AGENT_RAG_ROUND1_SYS
    
    # Format options
    options_text = "\n".join(f"{opt}: {val}" for opt, val in (options or {}).items()) if options else ""
    
    # Format debate history for context (only for Agent2)
    history_text = ""
    if debate_history:
        history_text = "\n## Previous Discussion:\n"
        for entry in debate_history:
            if isinstance(entry, dict):
                history_text += f"{entry.get('role', 'Unknown')}: {entry.get('message', '')}\n"
            else:
                history_text += f"{str(entry)}\n"
        history_text += "\n"
    
    # Format retrieved documents for context
    docs_context = format_retrieved_docs_for_context(pre_retrieved_docs)
    
    # Build prompt with pre-retrieved evidence
    prompt = f"""{system_prompt}

Question: {question}

Options:
{options_text}

{history_text}Retrieved Medical Evidence:
{docs_context}

Based on the provided evidence, provide your analysis:"""
    
    total_start_time = time.time()
    
    try:
        # Generate response with pre-retrieved documents
        print(f"[{role.upper()}] Generating analysis with {len(pre_retrieved_docs)} pre-retrieved documents...")
        generation_start = time.time()
        
        # Use different temperatures for each agent (from CoT)
        # Agent1: temperature 0.7, Agent2: temperature 0.3
        agent_temperature = 0.7 if role == "agent1" else 0.3
        
        # Generate reasoning with pre-retrieved context
        response = llm(
            prompt,
            max_length=8192,
            temperature=agent_temperature,
            return_format='string',
            stop_sequences=["<|im_end|>", "</s>", "###", "\n\n\n\n"],
            repetition_penalty=1.2
        )
        
        generation_time = time.time() - generation_start
        print(f"[TIMING] {role.upper()} generation: {generation_time:.2f}s")
        
        # Clean repetitive patterns from response
        response = clean_repetitive_response(response)
        print(f"[DEBUG] {role.upper()} response length after cleaning: {len(response)}")
        
        total_time = time.time() - total_start_time
        print(f"[TIMING] {role.upper()} total turn: {total_time:.2f}s")
        
        # Parse final response using MedRAG's proven parser
        parsed = parse_into_dict(response, model_name=HF_MODEL_NAME, question_id=qid, log_dir=log_dir)
        parsed["raw_response"] = response
        parsed["generation_time"] = generation_time
        parsed["retrieval_time"] = 0.0  # No retrieval time in pre-doc mode
        parsed["total_time"] = total_time
        parsed["mode"] = "rag_with_predocs"
        parsed["retrieved_docs"] = len(pre_retrieved_docs)
        
        # Save response
        save_json(parsed, f"{log_dir}/{qid}__{role}__latest_response.json")
        
        print(f"[{role.upper()}] Pre-docs Answer: {parsed.get('answer_choice', 'NONE')}")
        return parsed
        
    except Exception as e:
        print(f"[ERROR] {role} pre-docs failed: {e}")
        total_time = time.time() - total_start_time
        return {
            "step_by_step_thinking": f"Pre-docs error: {e}",
            "answer_choice": "PARSE_FAILED",
            "raw_response": "",
            "generation_time": 0.0,
            "total_time": total_time,
            "mode": "rag_with_predocs"
        }

def agent_turn_with_tools(llm, role, question, options, tools, qid, log_dir, debate_history=None):
    """
    Execute agent turn using tool-calling approach for RAG mode.
    
    Args:
        llm: VLLMWrapper instance
        role: "agent1" or "agent2"
        question: Medical question
        options: Answer options dict
        tools: Available tools dictionary
        qid: Question ID
        log_dir: Log directory
        debate_history: Previous debate messages
        
    Returns:
        Agent response dictionary
    """
    print(f"\n--- {role.upper()} TURN (RAG Tool-Calling) ---")
    
    # Use appropriate prompts based on debate history (adapted from CoT)
    if debate_history:
        # Debate round with previous discussion
        system_prompt = AGENT_RAG_DEBATE_SYS
    else:
        # Initial round without previous discussion
        system_prompt = AGENT_RAG_SYS
    
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
        
        # Use different temperatures for each agent (from CoT)
        # Agent1: temperature 0.7, Agent2: temperature 0.3
        agent_temperature = 0.7 if role == "agent1" else 0.3
        
        # Generate with stop sequences and repetition penalty
        response = llm(
            prompt,
            max_length=8192,  # Smaller for initial tool call
            temperature=agent_temperature,
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
            
            # Generate reasoning with retrieved context using same agent temperature
            reasoning_response = llm(
                reasoning_prompt,
                max_length=8192,  # Larger for full reasoning
                temperature=agent_temperature,
                return_format='string',  
                stop_sequences=["<|im_end|>", "</s>", "###", "\n\n\n\n"],  # Removed aggressive boxed stops
                repetition_penalty=1.2
            )
            
            reasoning_time = time.time() - reasoning_start
            print(f"[TIMING] {role.upper()} reasoning generation: {reasoning_time:.2f}s")
            
            # Clean repetitive patterns from reasoning response
            reasoning_response = clean_repetitive_response(reasoning_response)
            print(f"[DEBUG] {role.upper()} reasoning response length after cleaning: {len(reasoning_response)}")
            
            # Combine tool call and reasoning
            full_response = f"Tool Call: retrieve(\"{query}\")\n\nRetrieved Documents: {len(retrieved_docs)} documents\n\n{reasoning_response}"
            
            total_generation_time = tool_call_time + reasoning_time
            
        else:
            print(f"[{role.upper()}] No valid tool call found, using direct reasoning")
            # No tool call found, treat as direct reasoning - also clean it
            response = clean_repetitive_response(response)
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

def calculate_majority_vote_from_summaries(all_summaries, role1, role2):
    """
    Calculate majority vote from Round 2 summaries only.
    If no consensus in Round 2, return None to trigger reasoning verification.
    
    Args:
        all_summaries: Dict with round summaries
        role1, role2: Agent role names
        
    Returns:
        Tuple of (majority_answer, vote_breakdown_dict)
    """
    # Only collect Round 2 answers for majority vote
    round2_answers = []
    vote_breakdown = {}
    
    # Round 2 answers only
    if all_summaries.get('round2_role1') and all_summaries['round2_role1'].get('answer_choice') in ["A", "B", "C", "D"]:
        answer = all_summaries['round2_role1']['answer_choice']
        round2_answers.append(answer)
        vote_breakdown[f'{role1}_round2'] = answer
        
    if all_summaries.get('round2_role2') and all_summaries['round2_role2'].get('answer_choice') in ["A", "B", "C", "D"]:
        answer = all_summaries['round2_role2']['answer_choice']
        round2_answers.append(answer)
        vote_breakdown[f'{role2}_round2'] = answer
    
    # Add Round 1 to breakdown for logging purposes only (not used in voting)
    if all_summaries.get('round1_role1') and all_summaries['round1_role1'].get('answer_choice') in ["A", "B", "C", "D"]:
        vote_breakdown[f'{role1}_round1'] = all_summaries['round1_role1']['answer_choice']
    if all_summaries.get('round1_role2') and all_summaries['round1_role2'].get('answer_choice') in ["A", "B", "C", "D"]:
        vote_breakdown[f'{role2}_round1'] = all_summaries['round1_role2']['answer_choice']
    
    if not round2_answers:
        return None, vote_breakdown
    
    # Check for Round 2 consensus (both agents agree)
    if len(round2_answers) == 2 and round2_answers[0] == round2_answers[1]:
        return round2_answers[0], vote_breakdown
    
    # No consensus in Round 2 - return None to trigger reasoning verification
    return None, vote_breakdown

# ============================================================================
# DEBATE ORCHESTRATION
# ============================================================================

def role_based_verification(llm, agent1_response, agent2_response, question, options, qid, log_dir, role1="agent1", role2="agent2"):
    """
    Role-based consensus verification for RAG mode using agent1/agent2 roles (same as CoT).
    """
    print(f"\n--- ROLE-BASED VERIFICATION ---")
    
    # Add timing for verification process
    verification_start = time.time()
    
    # Display names for RAG mode (same as CoT)
    role1_display, role2_display = "Agent1", "Agent2"
    
    # Load all round summaries for majority vote calculations
    all_summaries = load_all_round_summaries(qid, log_dir, role1, role2)
    
    # Get final answers from current responses for consensus check
    agent1_choice = agent1_response.get('answer_choice', 'NONE')
    agent2_choice = agent2_response.get('answer_choice', 'NONE')
    
    # Use current agent responses directly (most reliable, adapted from CoT)
    # Truncate to reasonable length to avoid model hallucination
    agent1_text = agent1_response.get('step_by_step_thinking', '')
    agent2_text = agent2_response.get('step_by_step_thinking', '')
    
    # Clean responses to avoid garbage text (from CoT)
    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Remove any non-ASCII characters that might cause issues
        text = ''.join(char for char in text if ord(char) < 128 or char.isspace())
        return text.strip()
    
    agent1_clean = clean_text(agent1_text)
    agent2_clean = clean_text(agent2_text)
    
    # Build clean debate summary with clear agent labels (from CoT)
    debate_summary = (
        f"{role1_display}: {agent1_clean} Answer: {agent1_choice}\n\n"
        f"{role2_display}: {agent2_clean} Answer: {agent2_choice}"
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
{debate_summary[:4000]}

Based on the full debate progression across all rounds, which answer is most appropriate?

Provide your reasoning and final answer using \\boxed{{letter}} format:"""

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
            majority_answer, vote_breakdown = calculate_majority_vote_from_summaries(all_summaries, role1, role2)
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
            "<|im_end|>", "</s>", "\n\n\n\n",  # Basic model stop tokens
        ]
        
        # Use combined prompt approach to avoid problematic default stop sequences in generate_with_system
        # The generate_with_system method has hardcoded stop sequences that cut off reasoning
        combined_prompt = f"{verification_system}\n\n{verification_prompt}"
        verification = llm(
            combined_prompt, 
            max_length=8192, 
            temperature=0.7,
            do_sample=True,
            stop_sequences=verification_stops,
            repetition_penalty=1.1
        )
        
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
        
        # Log the raw verification response for debugging
        print(f"[DEBUG] Raw verification response length: {len(verification)}")
        print(f"[DEBUG] Raw verification response (first 300 chars): {verification[:300]}")
        if len(verification) > 300:
            print(f"[DEBUG] Raw verification response (last 300 chars): {verification[-300:]}")
        
        # Aggressive repetition detection and cleanup for verification
        verification = clean_repetitive_response(verification)
                
        # Parse verification using our improved sequential parser
        final_answer, _ = parse_answer_from_text(verification)  # parse_answer_from_text returns (answer, position)
        
        # Validate answer and use improved fallback if needed
        if final_answer == "PARSE_FAILED" or final_answer not in ["A", "B", "C", "D"]:
            # Try majority vote across all rounds first
            majority_answer, vote_breakdown = calculate_majority_vote_from_summaries(all_summaries, role1, role2)
            
            if majority_answer:
                final_answer = majority_answer
                verification_type = "fallback_majority_vote"
                print(f"[VERIFICATION] Using majority vote: {majority_answer}, breakdown: {vote_breakdown}")
            else:
                # Final fallback: use agent2 (Round 2 final answer) if available, then agent1
                if agent2_choice in ["A", "B", "C", "D"]:
                    final_answer = agent2_choice
                    verification_type = f"fallback_to_{role2}"
                    print(f"[VERIFICATION] No majority found, using {role2} (Round 2 final) fallback: {final_answer}")
                elif agent1_choice in ["A", "B", "C", "D"]:
                    final_answer = agent1_choice
                    verification_type = f"fallback_to_{role1}"
                    print(f"[VERIFICATION] No majority found, {role2} invalid, using {role1} fallback: {final_answer}")
                else:
                    # Both agents failed - use ERROR
                    final_answer = "ERROR"
                    verification_type = "fallback_both_agents_failed"
                    print(f"[VERIFICATION] No majority found, both agents failed, using ERROR")
        else:
            verification_type = "verification_consensus"
            print(f"[VERIFICATION] Verification consensus reached: {final_answer}")
        
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
            majority_answer, vote_breakdown = calculate_majority_vote_from_summaries(all_summaries, role1, role2)
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

def debate_question(medrag, llm, qid, question, options, k, log_dir, rounds=MAX_ROUNDS):
    """
    Run sequential RAG debate with optimized Round 1 summarization.
    
    Returns:
        Final consensus answer and comprehensive timing metrics
    """
    print(f"\n{'='*80}")
    print(f"QUESTION {qid}")
    print(f"{'='*80}")
    
    debate_start_time = time.time()
    
    # Create tools for RAG agents (for Round 2+ only)
    retrieval_tool = create_retrieval_tool(medrag, k=k, rrf_k=RRF_K)
    tools = {"retrieve": retrieval_tool}
    
    # Round 1: Pre-retrieve documents for both agents (original MedRAG style)
    print(f"\n[ROUND 1] Pre-retrieving documents for both agents...")
    agent1_predocs, agent2_predocs = unified_retrieval(medrag, question, options, k, qid, log_dir)
    
    history = []
    
    # RAG mode: use agent1 and agent2 roles (same as CoT)
    role1, role2 = "agent1", "agent2"
    role1_display, role2_display = "Agent1", "Agent2"
    
    # Multi-round sequential debate
    for round_num in range(1, rounds + 1):
        print(f"\n--- ROUND {round_num} ---")
        
        # First agent turn
        try:
            # Round 1: Use pre-retrieved documents, Round 2+: Use summaries and tool-calling
            recent_agent2 = None
            if round_num == 1:
                # Round 1: Use pre-retrieved documents
                agent1_response = agent_turn_with_predocs(
                    llm, role1, question, options, agent1_predocs, qid, log_dir,
                    debate_history=None  # No history for first agent in Round 1
                )
            elif round_num == 2:
                # Load Round 1 summaries for Round 2 - Agent1 gets both summaries
                summary1_data, summary2_data = load_round1_summaries(qid, log_dir, role1, role2)
                recent_agent2 = []
                
                # Add Agent1's own Round 1 summary first for context
                if summary1_data:
                    recent_agent2.append({
                        "role": role1,
                        "message": f"Your previous conclusion (Round 1): {summary1_data['formatted_summary']}",
                        "summary_type": "round1_self"
                    })
                
                # Add Agent2's Round 1 summary for comparison
                if summary2_data:
                    recent_agent2.append({
                        "role": role2,
                        "message": f"{role2_display} concluded (Round 1): {summary2_data['formatted_summary']}. Do you agree or disagree? Provide your analysis.",
                        "summary_type": "round1_other"
                    })
                
                # If no summaries available, set to None
                if not recent_agent2:
                    recent_agent2 = None
                    
                # Log Round 2 summary usage for Agent1
                if summary1_data or summary2_data:
                    summary_usage_log = {
                        'qid': qid,
                        'round': 2,
                        'agent': role1,
                        'uses_summary_from': 'both_agents' if (summary1_data and summary2_data) else ('self_only' if summary1_data else 'other_only'),
                        'summary1_available': bool(summary1_data),
                        'summary2_available': bool(summary2_data),
                        'timestamp': time.time()
                    }
                    summarization_jsonl_file = f"{log_dir}/summarization_logs.jsonl"
                    with open(summarization_jsonl_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(summary_usage_log, ensure_ascii=False) + '\n')
                
                # Round 2+: Use tool-calling with summaries
                agent1_response = agent_turn_with_tools(
                    llm, role1, question, options, tools, qid, log_dir,
                    debate_history=recent_agent2
                )
            
            # Save round-specific response for complete logging
            save_json(agent1_response, f"{log_dir}/{qid}__{role1}__round_{round_num}_response.json")
            
            history.append({
                "role": role1, 
                "round": round_num, 
                "message": f"{role1_display} Round {round_num}: {agent1_response.get('step_by_step_thinking', '')}\nAnswer: {agent1_response.get('answer_choice', 'NO_ANSWER')}",
                "answer": agent1_response
            })
            
            # Summarize responses for efficiency (skip if answer is ERROR)
            if round_num == 1:
                agent1_answer = agent1_response.get('answer_choice', 'NO_ANSWER')
                if agent1_answer not in ["ERROR", "PARSE_FAILED", "NO_ANSWER"]:
                    summarize_round1_agent_response(llm, agent1_response, role1, question, qid, log_dir)
                else:
                    print(f"[SUMMARY] Skipping Round 1 summary for {role1} - invalid answer: {agent1_answer}")
            elif round_num == 2:
                agent1_answer = agent1_response.get('answer_choice', 'NO_ANSWER')
                if agent1_answer not in ["ERROR", "PARSE_FAILED", "NO_ANSWER"]:
                    summarize_round2_agent_response(llm, agent1_response, role1, question, qid, log_dir)
                else:
                    print(f"[SUMMARY] Skipping Round 2 summary for {role1} - invalid answer: {agent1_answer}")
            
        except Exception as e:
            print(f"ERROR: {role1_display} Round {round_num} failed: {e}")
            continue
        
        # Second agent turn
        try:
            current_agent1 = None
            if round_num == 1:
                # Round 1: INDEPENDENT - Agent2 gets NO context from Agent1 (matches CoT design)
                current_agent1 = None  # Ensure true independence in Round 1
                
                # Round 1: Use pre-retrieved documents independently
                agent2_response = agent_turn_with_predocs(
                    llm, role2, question, options, agent2_predocs, qid, log_dir,
                    debate_history=None  # No history for independence
                )
            elif round_num == 2:
                # Round 2: Use Round 1 summaries - Agent2 gets both summaries
                summary1_data, summary2_data = load_round1_summaries(qid, log_dir, role1, role2)
                current_agent1 = []
                
                # Add Agent2's own Round 1 summary first for context
                if summary2_data:
                    current_agent1.append({
                        "role": role2,
                        "message": f"Your previous conclusion (Round 1): {summary2_data['formatted_summary']}",
                        "summary_type": "round1_self"
                    })
                
                # Add Agent1's Round 1 summary for comparison
                if summary1_data:
                    current_agent1.append({
                        "role": role1,
                        "message": f"{role1_display} concluded (Round 1): {summary1_data['formatted_summary']}. Do you agree or disagree? Provide your analysis.",
                        "summary_type": "round1_other"
                    })
                
                # If no summaries available, set to None
                if not current_agent1:
                    current_agent1 = None
                    
                # Log Round 2 summary usage for Agent2
                if summary1_data or summary2_data:
                    summary_usage_log = {
                        'qid': qid,
                        'round': 2,
                        'agent': role2,
                        'uses_summary_from': 'both_agents' if (summary1_data and summary2_data) else ('self_only' if summary2_data else 'other_only'),
                        'summary1_available': bool(summary1_data),
                        'summary2_available': bool(summary2_data),
                        'timestamp': time.time()
                    }
                    summarization_jsonl_file = f"{log_dir}/summarization_logs.jsonl"
                    with open(summarization_jsonl_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(summary_usage_log, ensure_ascii=False) + '\n')
                
                # Round 2+: Use tool-calling with summaries
                agent2_response = agent_turn_with_tools(
                    llm, role2, question, options, tools, qid, log_dir,
                    debate_history=current_agent1
                )
            
            # Save round-specific response for complete logging
            save_json(agent2_response, f"{log_dir}/{qid}__{role2}__round_{round_num}_response.json")
            
            history.append({
                "role": role2,
                "round": round_num,
                "message": f"{role2_display} Round {round_num}: {agent2_response.get('step_by_step_thinking', '')}\nAnswer: {agent2_response.get('answer_choice', 'NO_ANSWER')}",
                "answer": agent2_response
            })
            
            # Summarize responses for efficiency (skip if answer is ERROR)
            if round_num == 1:
                agent2_answer = agent2_response.get('answer_choice', 'NO_ANSWER')
                if agent2_answer not in ["ERROR", "PARSE_FAILED", "NO_ANSWER"]:
                    summarize_round1_agent_response(llm, agent2_response, role2, question, qid, log_dir)
                else:
                    print(f"[SUMMARY] Skipping Round 1 summary for {role2} - invalid answer: {agent2_answer}")
            elif round_num == 2:
                agent2_answer = agent2_response.get('answer_choice', 'NO_ANSWER')
                if agent2_answer not in ["ERROR", "PARSE_FAILED", "NO_ANSWER"]:
                    summarize_round2_agent_response(llm, agent2_response, role2, question, qid, log_dir)
                else:
                    print(f"[SUMMARY] Skipping Round 2 summary for {role2} - invalid answer: {agent2_answer}")
            
        except Exception as e:
            print(f"ERROR: {role2_display} Round {round_num} failed: {e}")
            continue
    
    # Role-based verification using Round 1 summaries when available
    if len(history) >= 2:
        final_agent1 = [h for h in history if h.get("role") == role1][-1]["answer"]
        final_agent2 = [h for h in history if h.get("role") == role2][-1]["answer"]
        
        final_answer = role_based_verification(
            llm, final_agent1, final_agent2, question, options, qid, log_dir, role1, role2
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
        "mode": "rag",
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

def run_debate_benchmark(dataset_name="mmlu", k=DEFAULT_K, log_dir="./debate_logs", split="test", resume=True, corpus_name="MedCorp"):
    """
    Run RAG debate-based evaluation on a MedQA dataset.
    
    Args:
        dataset_name: Name of dataset (mmlu, medqa, etc.)
        k: Number of snippets to retrieve per role
        log_dir: Base directory for logs
        split: Dataset split ("test" or "dev" for medmcqa)
        resume: Skip already processed questions (default: True)
        corpus_name: Name of corpus for directory structure
    """
    print(f"\n{'='*80}")
    print(f"RAG DEBATE BENCHMARK: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Setup paths with correct structure: log_dir/dataset_retriever_corpus/
    retriever_name = "MedCPT"  # Fixed retriever name
    structured_log_dir = os.path.join(log_dir, f"{dataset_name}_{retriever_name}_{corpus_name}")
    structured_log_dir = os.path.abspath(structured_log_dir)
    os.makedirs(structured_log_dir, exist_ok=True)
    
    print(f"Log directory: {structured_log_dir}")
    
    # Set memory management for PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            # Clear any existing GPU cache
            torch.cuda.empty_cache()
            print(f"[GPU] Cleared CUDA cache")
            print(f"[GPU] Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                if i < 2:  # Only check our visible devices
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                    print(f"[GPU] Device {i}: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")
    except ImportError:
        print("[WARNING] PyTorch not available for memory management")
    
    # Initialize MedRAG with optimized single-source setup
    print(f"Initializing MedRAG with optimized setup...")
    patch_medrag_for_vllm()
    
    # Map corpus_name to actual MedRAG corpus name
    corpus_mapping = {
        "MedCorp": "MedCorp2",
        "MedCorp2": "MedCorp2"
    }
    medrag_corpus_name = corpus_mapping.get(corpus_name, "MedCorp2")
    
    print(f"  RAG enabled with single source: {corpus_name}, retriever: MedCPT, k={k}")
    print(f"  GPU assignment: FAISS on GPU {FAISS_GPU_ID}, VLLM on GPU {VLLM_GPU_ID}")
    medrag = MedRAG(
        llm_name=HF_MODEL_NAME,
        rag=True,
        retriever_name="MedCPT", 
        corpus_name=medrag_corpus_name,
        db_dir=MEDCORP_DIR,
        corpus_cache=True,
        HNSW=True
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
    processed = 0
    skipped = 0
    
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
        
        # Check if already processed (resume capability)
        result_file = f"{structured_log_dir}/{qid_for_file}.json"
        if resume and os.path.exists(result_file):
            skipped += 1
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
                medrag, llm, qid_for_file, question, options, k, structured_log_dir
            )
            
            predicted = final.get("answer_choice")
            is_correct = (predicted == gold_answer) if predicted in ["A", "B", "C", "D"] else False
            
            if is_correct:
                correct += 1
            
            # Extract timing information
            timing_info = final.get("debate_timing", {})
            
            result = {
                "step_by_step_thinking": final.get("step_by_step_thinking", ""),
                "answer_choice": predicted
            }
            
            results.append({
                "id": qid_for_file,
                "question": question,
                "gold": gold_answer,
                "predicted": predicted,
                "correct": is_correct,
                "reasoning": final.get("step_by_step_thinking", ""),
                "timing": timing_info
            })
            
            # Save individual result for resume capability
            save_json([result], result_file)
            processed += 1
            
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
    results_file = f"{structured_log_dir}/{dataset_name}_results.json"
    save_json(results, results_file)
    
    # Print summary
    accuracy = correct / len(results) if results else 0
    
    # Calculate timing statistics (updated for agent1/agent2)
    total_times = [r.get("timing", {}).get("total_debate_time", 0.0) for r in results]
    retrieval_times = [r.get("timing", {}).get("retrieval_time", 0.0) for r in results]
    agent1_times = [r.get("timing", {}).get("agent1_gen_time", 0.0) for r in results]
    agent2_times = [r.get("timing", {}).get("agent2_gen_time", 0.0) for r in results]
    verification_times = [r.get("timing", {}).get("verification_time", 0.0) for r in results]
    
    avg_total = sum(total_times) / len(total_times) if total_times else 0
    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0  
    avg_agent1 = sum(agent1_times) / len(agent1_times) if agent1_times else 0
    avg_agent2 = sum(agent2_times) / len(agent2_times) if agent2_times else 0
    avg_verification = sum(verification_times) / len(verification_times) if verification_times else 0
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total Questions: {len(dataset)}")
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nTiming Summary (averages):")
    print(f"  Total per question: {avg_total:.1f}s")
    print(f"  Retrieval: {avg_retrieval:.1f}s")
    print(f"  Agent1 generation: {avg_agent1:.1f}s") 
    print(f"  Agent2 generation: {avg_agent2:.1f}s")
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
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                       help="Number of snippets to retrieve per role")
    parser.add_argument("--log_dir", type=str, default="./debate_logs_rag",
                       help="Directory for debate logs")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS,
                       help="Number of debate rounds")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split (default: test, auto-switched to dev for medmcqa)")
    parser.add_argument("--corpus_name", type=str, default="MedCorp2",
                       help="Name of corpus to use (default: MedCorp2)")
    parser.add_argument("--retriever_name", type=str, default="MedCPT",
                       help="Name of retriever to use (default: MedCPT)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Reprocess all questions (default: skip already processed)")
    
    args = parser.parse_args()
    
    # Extract model name from HF_MODEL_NAME for directory structure
    model_name = HF_MODEL_NAME.split("/")[-1].replace("-", "_")
    
    # Create structured log directory following rag_{dataset}_{retriever}_{model_name} pattern
    structured_log_dir = f"rag_{args.dataset}_{args.retriever_name}_{model_name}"
    args.log_dir = os.path.join(args.log_dir, structured_log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"Using log directory: {args.log_dir}")

    # Update global rounds if specified
    MAX_ROUNDS = args.rounds
    
    # Run benchmark with structured logging
    run_debate_benchmark(
        dataset_name=args.dataset,
        k=args.k,
        log_dir=args.log_dir,
        split=args.split,
        resume=not args.no_resume,
        corpus_name=args.corpus_name
    )
