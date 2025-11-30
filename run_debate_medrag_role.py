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

from run_medrag_vllm import patch_medrag_for_vllm, VLLMWrapper
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
HF_MODEL_NAME = "Qwen/Qwen3-8B"

# FAISS GPU configuration
FAISS_GPU_ID = 0  # First visible GPU (GPU 0) for FAISS index
VLLM_GPU_ID = 1   # Second visible GPU (GPU 1) for VLLM

MEDCORP_DIR = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"

# Retrieval parameters
DEFAULT_K = 32
RRF_K = 60
MAX_ROUNDS = 2

# ============================================================================
# TOOL-CALLING SYSTEM PROMPTS 
# ============================================================================

ANALYST_SYS = """You are MD-Analyst, a medical AI assistant with access to a retrieval tool for evidence-based reasoning.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Process:
1) First call retrieve(query) to gather relevant medical evidence
2) Review the retrieved documents  
3) Provide clinical reasoning based on the evidence
4) Give your final answer choice using \\boxed{} format and STOP

Format your response:
Tool Call: retrieve("your search query")
Evidence Review: Analyze retrieved documents [cite doc_ids]
Reasoning: Clinical logic based on evidence (max 800 tokens)
\\boxed{A} or \\boxed{B} or \\boxed{C} or \\boxed{D}

IMPORTANT: End with EXACTLY \\boxed{X} where X is your answer choice (A, B, C, or D) and then stop immediately. Do not repeat or elaborate further.

Focus on evidence-based medicine and clinical guidelines."""

SKEPTIC_SYS = """You are MD-Skeptic, a critical medical AI assistant with access to a retrieval tool for evidence-based analysis.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Process:
1) First call retrieve(query) to gather counter-evidence or confirmatory evidence
2) Review the retrieved documents
3) Critically analyze the Analyst's reasoning
4) Provide your assessment using \\boxed{} format and STOP

Format your response:
Tool Call: retrieve("your search query")
Evidence Review: Analyze retrieved documents [cite doc_ids]
Critique: Critical analysis of reasoning (max 800 tokens)
\\boxed{A} or \\boxed{B} or \\boxed{C} or \\boxed{D}

IMPORTANT: End with EXACTLY \\boxed{X} where X is your answer choice (A, B, C, or D) and then stop immediately. Do not repeat or elaborate further.

Be thorough but constructive in your analysis."""

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
    """Extract answer choice from agent output"""
    # Ensure text is a string
    if text is None:
        return "PARSE_FAILED"
    if not isinstance(text, str):
        text = str(text)
    
    # PRIORITY: Look for \boxed{answer} format first (most reliable)
    boxed_pattern = r'\\boxed\{([ABCD])\}'
    boxed_match = re.search(boxed_pattern, text, re.IGNORECASE)
    if boxed_match:
        answer = boxed_match.group(1).upper()
        print(f"[DEBUG] Found boxed answer '{answer}' using \\boxed{{}} format")
        return answer
    
    # FALLBACK: Look for other answer patterns
    patterns = [
        r'Answer:\s*([ABCD])',
        r'\*\*Answer:\s*([ABCD])\*\*',
        r'The correct answer is\s*\*\*([ABCD])\*\*',
        r'The correct answer is\s*([ABCD])',
        r'answer_choice["\']?\s*:\s*["\']?([ABCD])',
        r'\b([ABCD])\s*is\s+(?:the\s+)?(?:correct|best|right)',
        r'option\s+([ABCD])',
        r'Therefore,?\s*(?:the answer is\s*)?([ABCD])',
        r'Final answer:\s*([ABCD])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            print(f"[DEBUG] Found answer '{answer}' using pattern: {pattern}")
            return answer
    
    # Last resort: look for standalone letters near end of text
    last_part = text[-200:] if len(text) > 200 else text
    standalone_match = re.search(r'\b([ABCD])\b', last_part)
    if standalone_match:
        answer = standalone_match.group(1).upper()
        print(f"[DEBUG] Found standalone answer '{answer}' in text ending")
        return answer
    
    print(f"[DEBUG] No answer pattern found in text: {text[:200]}...")
    # Return invalid answer that won't match any correct answer
    return "PARSE_FAILED"

def parse_into_dict(response):
    """Parse response string into structured dict"""
    if isinstance(response, dict):
        return response
    
    # Ensure response is a string
    if response is None:
        response = ""
    elif not isinstance(response, str):
        response = str(response)
    
    # Try JSON parsing first
    try:
        # Look for JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        pass
    
    # Fallback: extract answer and use full text as reasoning
    answer = parse_answer_from_text(response)
    return {
        "step_by_step_thinking": response,
        "answer_choice": answer
    }

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

def summarize_agent_response(llm, agent_response, role, question, mode="rag"):
    """
    Intelligently summarize an agent's response to preserve key conclusions
    while reducing token count for subsequent rounds.
    Only performs summarization in RAG mode for efficiency.
    
    Args:
        llm: VLLMWrapper instance
        agent_response: Dict with agent's full response
        role: "analyst" or "skeptic"
        question: Original question for context
        mode: "rag" or "cot" - only summarizes in RAG mode
    
    Returns:
        Summarized response dict with preserved key information
    """
    full_reasoning = agent_response.get('step_by_step_thinking', '')
    answer_choice = agent_response.get('answer_choice', 'NO_ANSWER')
    
    # In CoT mode, return original response without summarization for efficiency
    if mode == "cot":
        return {
            'summary': full_reasoning,
            'answer_choice': answer_choice,
            'reasoning_type': 'original_cot'
        }
    
    # If reasoning is already short, no need to summarize
    if len(full_reasoning) < 300:
        return {
            'summary': full_reasoning,
            'answer_choice': answer_choice,
            'reasoning_type': 'original'
        }
    
    # Create summarization prompt
    summarize_prompt = f"""Summarize this medical reasoning in 2-3 sentences, preserving the key clinical logic and final conclusion:

Question: {question[:200]}...

{role.title()}'s reasoning: {full_reasoning}

Final answer: {answer_choice}

Provide a concise summary that captures the essential medical reasoning and conclusion:"""

    try:
        summary_response = llm(summarize_prompt, max_length=1024, temperature=0.0)
        
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
        if len(summary_response.strip()) < 20:
            # Fallback: extract first and last sentences
            sentences = full_reasoning.split('.')
            if len(sentences) >= 2:
                summary_response = f"{sentences[0].strip()}. {sentences[-2].strip()}."
            else:
                summary_response = full_reasoning[:200] + "..."
        
        return {
            'summary': summary_response.strip(),
            'answer_choice': answer_choice,
            'reasoning_type': 'summarized'
        }
        
    except Exception as e:
        print(f"[WARNING] Summarization failed for {role}: {e}")
        # Fallback: simple truncation with key info preserved
        key_sentences = full_reasoning.split('.')[:3]  # First 3 sentences
        fallback_summary = '. '.join(s.strip() for s in key_sentences if s.strip()) + '.'
        
        return {
            'summary': fallback_summary,
            'answer_choice': answer_choice,
            'reasoning_type': 'fallback'
        }

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
        role: "analyst" or "skeptic"
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
    
    # Use role-specific system prompt
    role_prompts = {
        "analyst": ANALYST_SYS,
        "skeptic": SKEPTIC_SYS
    }
    
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
        # In CoT mode, skip tool calling and use direct reasoning
        prompt = f"""{role_prompts.get(role, '')}

Question: {question}

Options:
{options_text}

{history_text}

Provide your analysis without using tools (chain-of-thought reasoning):"""
        
        # Generate direct response without tools
        try:
            generation_start = time.time()
            
            # Add stop sequences and repetition penalty for better generation
            response = llm(
                prompt, 
                max_length=4096,
                temperature=0.0,
                return_format='string',  # Get plain string
                stop_sequences=["<|im_end|>", "</s>", "###", "\n\n\n", "**Answer:", "Answer:**", "\n**", "**\n", "\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}"],
                repetition_penalty=1.2
            )
            
            generation_time = time.time() - generation_start
            print(f"[TIMING] {role.upper()} CoT generation: {generation_time:.2f}s")
            
            # Parse response
            parsed = parse_into_dict(response)
            parsed["raw_response"] = response
            parsed["generation_time"] = generation_time
            parsed["mode"] = "cot"
            
            # Save response
            save_json(parsed, f"{log_dir}/{qid}__{role}__response.json")
            
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
    prompt = f"""{role_prompts.get(role, '')}

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
            max_length=3072,  # Smaller for initial tool call
            temperature=0.0,
            return_format='string',
            stop_sequences=["Evidence Review:", "Reasoning:", "Critique:", "<|im_end|>", "</s>", "**Answer:", "Answer:**", "\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}"],
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
            reasoning_prompt = f"""{role_prompts.get(role, '')}

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
                max_length=4096,  # Larger for full reasoning
                temperature=0.0,
                return_format='string',
                stop_sequences=["<|im_end|>", "</s>", "###", "\n\n\n", "**Answer:", "Answer:**", "\n**", "**\n", "\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}"],
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
        
        # Parse final response
        parsed = parse_into_dict(full_response)
        parsed["raw_response"] = full_response
        parsed["generation_time"] = total_generation_time
        parsed["retrieval_time"] = retrieval_time
        parsed["total_time"] = total_time
        parsed["mode"] = "rag_with_tools"
        parsed["retrieved_docs"] = len(retrieved_docs)
        
        # Save response
        save_json(parsed, f"{log_dir}/{qid}__{role}__response.json")
        
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

# ============================================================================
# DEBATE ORCHESTRATION
# ============================================================================

def role_based_verification(llm, analyst_response, skeptic_response, question, options, qid, log_dir, mode="rag"):
    """
    Role-based consensus verification using summarized debate context.
    """
    print(f"\n--- ROLE-BASED VERIFICATION ---")
    
    # Add timing for verification process
    verification_start = time.time()
    
    # Create concise debate summary using our summarization function
    analyst_summary = summarize_agent_response(llm, analyst_response, "analyst", question, mode)
    skeptic_summary = summarize_agent_response(llm, skeptic_response, "skeptic", question, mode)
    
    # Simple consensus check first
    analyst_choice = analyst_response.get('answer_choice', 'NONE')
    skeptic_choice = skeptic_response.get('answer_choice', 'NONE')
    
    # If both agents agree and answer is valid, use direct consensus
    if (analyst_choice == skeptic_choice and 
        analyst_choice in ["A", "B", "C", "D"]):
        print(f"[CONSENSUS] Both agents agree on answer: {analyst_choice}")
        total_verification_time = time.time() - verification_start
        print(f"[TIMING] Quick consensus verification: {total_verification_time:.2f}s")
        return {
            "step_by_step_thinking": f"Consensus reached: Both analyst and skeptic agree on {analyst_choice}.",
            "answer_choice": analyst_choice,
            "consensus_type": "direct_agreement",
            "total_verification_time": total_verification_time
        }
    
    # If disagreement, use verification with longer context window
    debate_summary = (
        f"Analyst: {analyst_summary['summary']}\nAnswer: {analyst_summary['answer_choice']}\n\n"
        f"Skeptic: {skeptic_summary['summary']}\nAnswer: {skeptic_summary['answer_choice']}"
    )

    # Create constrained verification prompt with system message and token limits
    verification_system = """You are a medical expert verifier. Provide ONLY a brief rationale (max 200 words) followed by the answer in \\boxed{} format. Do not repeat content. Stop after giving your boxed answer.

Format:
[Brief reasoning]
\\boxed{A} or \\boxed{B} or \\boxed{C} or \\boxed{D}

IMPORTANT: Keep response under 1000 tokens. End with EXACTLY \\boxed{X} where X is your answer choice and stop immediately."""

    verification_prompt = f"""Review this medical debate and determine the best answer based on medical evidence:

Question: {question}

Options:
{chr(10).join(f"{k}: {v}" for k, v in (options or {}).items())}

Debate Summary:
{debate_summary}

The agents disagree. Based on the medical evidence and clinical reasoning presented, which answer is most appropriate? Consider standard medical practice and evidence-based medicine.

Provide your reasoning and answer:"""

    try:
        # Use constrained generation for verification with aggressive stop sequences
        max_total_length = 2048  # Reduced for better control
        min_generation_tokens = 50  # Reduced minimum
        
        if hasattr(llm, 'tokenizer'):
            full_prompt = f"{verification_system}\n\n{verification_prompt}"
            prompt_tokens = len(llm.tokenizer.encode(full_prompt))
        else:
            full_prompt = f"{verification_system}\n\n{verification_prompt}"
            prompt_tokens = len(full_prompt) // 4
        
        available_tokens = max_total_length - prompt_tokens
        if available_tokens < min_generation_tokens:
            print(f"[WARNING] Verification prompt too long, using analyst answer as fallback")
            return analyst_response
        
        print(f"[DEBUG] Verification: prompt_tokens={prompt_tokens}, available_tokens={available_tokens}")
        
        # Time the verification generation with enhanced stop sequences
        verification_gen_start = time.time()
        
        # Enhanced stop sequences for verification
        verification_stops = [
            "\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}",  # Primary stop patterns
            "Answer: A", "Answer: B", "Answer: C", "Answer: D",
            "**Answer: A**", "**Answer: B**", "**Answer: C**", "**Answer: D**",
            "\n\nQuestion:", "\n\nOptions:", "The correct answer is A",
            "The correct answer is B", "The correct answer is C", "The correct answer is D",
            "**A**", "**B**", "**C**", "**D**",
            "A\n", "B\n", "C\n", "D\n",
            "\n---", "---\n", "\n\n\n"
        ]
        
        # Use system message if available in VLLMWrapper
        if hasattr(llm, 'generate_with_system'):
            verification = llm.generate_with_system(
                verification_system, 
                verification_prompt, 
                max_length=max_total_length, 
                temperature=0.0,
                stop_sequences=verification_stops,
            )
        else:
            # Fallback to combined prompt
            combined_prompt = f"{verification_system}\n\n{verification_prompt}"
            verification = llm(combined_prompt, max_length=max_total_length, temperature=0.0)
        
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
        
        print(f"[DEBUG] Verification response length after cleaning: {len(verification)}")
        print(f"[DEBUG] Verification response preview: {verification[:200]}...")
        
        # Extract final answer
        final_answer = parse_answer_from_text(verification)
        
        # Validate answer and use fallback if needed
        if final_answer == "PARSE_FAILED" or final_answer not in ["A", "B", "C", "D"]:
            # Use analyst answer as fallback since it typically has better reasoning
            final_answer = analyst_choice if analyst_choice in ["A", "B", "C", "D"] else skeptic_choice
            verification_type = "fallback_to_analyst"
        else:
            verification_type = "verification_consensus"
        
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
        # Return analyst answer as fallback
        fallback_answer = analyst_choice if analyst_choice in ["A", "B", "C", "D"] else skeptic_choice
        total_verification_time = time.time() - verification_start
        return {
            "step_by_step_thinking": f"Verification failed: {e}. Using analyst answer as fallback.",
            "answer_choice": fallback_answer,
            "consensus_type": "error_fallback",
            "total_verification_time": total_verification_time
        }

def debate_question(medrag, llm, qid, question, options, k, log_dir, mode="cot", rounds=MAX_ROUNDS):
    """
    Run sequential debate with tool-calling approach.
    
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
    
    # Multi-round sequential debate
    for round_num in range(1, rounds + 1):
        print(f"\n--- ROUND {round_num} ---")
        
        # Analyst turn first
        try:
            # For round 2+, include previous skeptic response
            recent_skeptic = None
            if round_num > 1 and history:
                skeptic_entries = [h for h in history if h.get("role") == "skeptic"]
                if skeptic_entries:
                    last_skeptic = skeptic_entries[-1]
                    # Summarize the skeptic's response for token efficiency
                    skeptic_summary = summarize_agent_response(
                        llm, last_skeptic.get("answer", {}), "skeptic", question, mode
                    )
                    recent_skeptic = [{
                        "role": "skeptic",
                        "message": f"Skeptic's conclusion: {skeptic_summary['summary']} Answer: {skeptic_summary['answer_choice']}",
                        "summary_type": skeptic_summary['reasoning_type']
                    }]
            
            analyst_response = agent_turn_with_tools(
                llm, "analyst", question, options, tools, qid, log_dir,
                mode=mode, debate_history=recent_skeptic
            )
            
            history.append({
                "role": "analyst", 
                "round": round_num, 
                "message": f"Analyst Round {round_num}: {analyst_response.get('step_by_step_thinking', '')}\nAnswer: {analyst_response.get('answer_choice', 'NO_ANSWER')}",
                "answer": analyst_response
            })
            
        except Exception as e:
            print(f"ERROR: Analyst Round {round_num} failed: {e}")
            continue
        
        # Skeptic turn (sees summarized analyst response)
        try:
            current_analyst = None
            if history:
                last_analyst = history[-1]
                # Summarize analyst response for token efficiency
                analyst_summary = summarize_agent_response(
                    llm, last_analyst.get("answer", {}), "analyst", question, mode
                )
                current_analyst = [{
                    "role": "analyst",
                    "message": f"Analyst's conclusion: {analyst_summary['summary']} Answer: {analyst_summary['answer_choice']}",
                    "summary_type": analyst_summary['reasoning_type']
                }]
            
            skeptic_response = agent_turn_with_tools(
                llm, "skeptic", question, options, tools, qid, log_dir,
                mode=mode, debate_history=current_analyst
            )
            
            history.append({
                "role": "skeptic",
                "round": round_num,
                "message": f"Skeptic Round {round_num}: {skeptic_response.get('step_by_step_thinking', '')}\nAnswer: {skeptic_response.get('answer_choice', 'NO_ANSWER')}",
                "answer": skeptic_response
            })
            
        except Exception as e:
            print(f"ERROR: Skeptic Round {round_num} failed: {e}")
            continue
    
    # Role-based verification to determine final answer
    if len(history) >= 2:
        final_analyst = [h for h in history if h.get("role") == "analyst"][-1]["answer"]
        final_skeptic = [h for h in history if h.get("role") == "skeptic"][-1]["answer"]
        
        final_answer = role_based_verification(
            llm, final_analyst, final_skeptic, question, options, qid, log_dir, mode
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
        "analyst_times": [],
        "skeptic_times": [],
        "retrieval_times": []
    }
    
    for entry in history:
        role = entry.get("role")
        answer = entry.get("answer", {})
        
        if role == "analyst":
            timing_summary["analyst_times"].append({
                "round": entry.get("round"),
                "generation_time": answer.get("generation_time", 0),
                "retrieval_time": answer.get("retrieval_time", 0),
                "total_time": answer.get("total_time", 0)
            })
        elif role == "skeptic":
            timing_summary["skeptic_times"].append({
                "round": entry.get("round"),
                "generation_time": answer.get("generation_time", 0),
                "retrieval_time": answer.get("retrieval_time", 0),
                "total_time": answer.get("total_time", 0)
            })
    
    # Add verification timing if available
    timing_summary["verification_time"] = final_answer.get("total_verification_time", 0)
    
    print(f"\n[TIMING SUMMARY]")
    print(f"Total debate time: {total_debate_time:.2f}s")
    print(f"Verification time: {timing_summary['verification_time']:.2f}s")
    
    # Save complete debate log
    debate_log = {
        "qid": qid,
        "question": question,
        "options": options,
        "mode": mode,
        "rounds": rounds,
        "history": history,
        "final_answer": final_answer,
        "timing_summary": timing_summary
    }
    
    save_json(debate_log, f"{log_dir}/{qid}__complete_debate.json")
    
    print(f"[FINAL ANSWER] {final_answer.get('answer_choice', 'NONE')}")
    print(f"[CONSENSUS TYPE] {final_answer.get('consensus_type', 'unknown')}")
    
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
