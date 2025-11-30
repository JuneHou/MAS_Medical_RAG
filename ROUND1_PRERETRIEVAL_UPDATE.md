# Round 1 Pre-Retrieval Update

## Overview
Updated the RAG debate system to use different retrieval approaches for Round 1 vs Round 2+:

- **Round 1**: Pre-retrieve documents using the original question (like original MedRAG)
- **Round 2+**: Keep existing tool-calling approach where agents generate their own queries

## Key Changes

### 1. New Functions Added

#### `pre_retrieve_documents_for_round1(medrag, question, options, k, qid, log_dir)`
- Pre-retrieves documents using MedRAG's built-in retrieval system
- Uses the original question directly (like original MedRAG approach)
- Reuses `medrag.retrieval_system.retrieve(question, k=k, rrf_k=RRF_K)`
- Logs retrieval information for debugging

#### `agent_turn_round1_with_preretrieval(llm, role, question, options, retrieved_docs, qid, log_dir, debate_history=None)`
- Handles Round 1 agent turns with pre-retrieved documents
- No tool calling - agents analyze provided evidence directly
- Modified system prompt that removes tool calling instructions
- Formats retrieved documents as evidence context

### 2. Modified Functions

#### `debate_question()`
- Added pre-retrieval step before the main debate loop
- Modified Round 1 logic to use `agent_turn_round1_with_preretrieval()`
- Round 2+ continues to use existing `agent_turn_with_tools()`
- Both agents in Round 1 use the same pre-retrieved documents

#### `agent_turn_with_tools()`
- Updated docstring to clarify it's for Round 2+
- No functional changes to the tool-calling logic

### 3. Retrieval Strategy

#### Round 1:
```python
# Pre-retrieve once for both agents
round1_retrieved_docs = pre_retrieve_documents_for_round1(medrag, question, options, k, qid, log_dir)

# Both agents use same documents
agent1_response = agent_turn_round1_with_preretrieval(llm, role1, question, options, round1_retrieved_docs, ...)
agent2_response = agent_turn_round1_with_preretrieval(llm, role2, question, options, round1_retrieved_docs, ...)
```

#### Round 2+:
```python
# Agents use tool calling to generate their own queries
agent1_response = agent_turn_with_tools(llm, role1, question, options, tools, ...)
agent2_response = agent_turn_with_tools(llm, role2, question, options, tools, ...)
```

## Benefits

1. **Round 1 Efficiency**: No need for agents to generate retrieval queries - uses question directly
2. **Consistent with Original MedRAG**: Round 1 follows original MedRAG approach of using question as query
3. **Round 2+ Flexibility**: Agents can still generate specific queries for follow-up retrieval
4. **Reuse Existing Code**: Leverages existing MedRAG retrieval system and utility functions
5. **Logging**: Comprehensive logging for both retrieval types

## Usage

The system now automatically:
1. Pre-retrieves documents for Round 1 using the original question
2. Provides same documents to both agents in Round 1
3. Allows agents to use tool calling for custom queries in Round 2+

No changes needed to command line interface or configuration.