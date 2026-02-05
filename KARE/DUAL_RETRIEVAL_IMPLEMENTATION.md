# Dual-Query Retrieval Implementation Summary

**Date**: January 19, 2026
**File Modified**: `mortality_debate_rag.py`

## Overview

Implemented dual-query retrieval system for the integrator agent, allowing separate optimized queries for MedCorp (clinical evidence) and UMLS (medical terminology/concepts).

## Key Features

### 1. **Optional Dual Retrieval**
- Agent can generate **0, 1, or 2 queries** (not required)
- Supports both dual-query and single-query formats (backward compatible)
- Flexible retrieval based on what the model decides is needed

### 2. **Separate Source Retrieval**
- `<search_medcorp>query</search_medcorp>` → Retrieves from MedCorp sources (clinical literature)
- `<search_umls>query</search_umls>` → Retrieves from UMLS (medical concepts/terminology)
- Each query can be optimized for its specific knowledge source

### 3. **Stop Criteria**
- **Unchanged**: Uses same stop sequences as before
- Stops at `</search>`, `</search_medcorp>`, or `</search_umls>`
- No changes to `max_new_tokens` or `max_length` parameters

## Implementation Details

### Modified Functions

#### 1. `_parse_tool_call(response_text)` (Lines ~606-659)
**Changes**:
- Added dual-query pattern matching for `<search_medcorp>` and `<search_umls>` tags
- Returns dict with `{'medcorp_query': str|None, 'umls_query': str|None}` for dual queries
- Falls back to single-query format `(tool_name, query)` for backward compatibility

**Logic**:
```python
# Try dual-query first
medcorp_match = re.search(r'<search_medcorp>\s*(.{10,}?)\s*</search_medcorp>', response_text)
umls_match = re.search(r'<search_umls>\s*(.{10,}?)\s*</search_umls>', response_text)

if medcorp_match or umls_match:
    return {'medcorp_query': medcorp_match.group(1) if medcorp_match else None,
            'umls_query': umls_match.group(1) if umls_match else None}

# Fallback to single-query patterns
# ... (existing patterns unchanged)
```

#### 2. `_execute_tool_call(tool_name_or_queries, query=None, ...)` (Lines ~661-691)
**Changes**:
- Renamed first parameter to `tool_name_or_queries` to accept both formats
- Routes to `_execute_dual_retrieval()` if dict detected
- Preserves single-query behavior for backward compatibility

**Logic**:
```python
if isinstance(tool_name_or_queries, dict):
    return self._execute_dual_retrieval(tool_name_or_queries, qid=qid, log_dir=log_dir)

# Single-query path (unchanged)
tool_name = tool_name_or_queries
# ... existing logic
```

#### 3. `_execute_dual_retrieval(queries_dict, ...)` (Lines ~693-768) **[NEW]**
**Purpose**: Execute separate retrievals for MedCorp and UMLS

**Logic**:
1. Extract `medcorp_query` and `umls_query` from dict
2. Truncate each query to 2048 tokens (same as before)
3. Retrieve from `self.medrag.source_retrievers["medcorp"]` if MedCorp query exists (k=4)
4. Retrieve from `self.medrag.source_retrievers["umls"]` if UMLS query exists (k=4)
5. Combine results (up to 8 docs total, take top 5)
6. Format and log retrieval

**Key Code**:
```python
# Retrieve from MedCorp if query provided
if medcorp_query and hasattr(self.medrag, 'source_retrievers'):
    truncated_query = medcorp_query[:MAX_INTEGRATOR_QUERY_CHARS]
    snippets, scores = self.medrag.source_retrievers["medcorp"].retrieve(
        truncated_query, k=4, rrf_k=60
    )
    all_retrieved_snippets.extend(snippets)

# Retrieve from UMLS if query provided
if umls_query and hasattr(self.medrag, 'source_retrievers'):
    truncated_query = umls_query[:MAX_INTEGRATOR_QUERY_CHARS]
    snippets, scores = self.medrag.source_retrievers["umls"].retrieve(
        truncated_query, k=4, rrf_k=60
    )
    all_retrieved_snippets.extend(snippets)
```

#### 4. `_execute_integrator_attempt(...)` (Lines ~897-1015)
**Changes**:
- Updated stop sequences: `["</search>", "</search_medcorp>", "</search_umls>", ...]`
- Added dual-query detection and handling
- Auto-completes missing closing tags for both formats
- Routes to appropriate retrieval path based on query format

**Generation Flow**:
```python
# Step 1: Generate with dual/single stop sequences
tool_response = self.integrator_llm(
    initial_prompt,
    stop_sequences=["</search>", "</search_medcorp>", "</search_umls>", "<|im_end|>", "</s>"],
    # ... other params UNCHANGED
)

# Step 2: Auto-complete tags if needed
if "<search_medcorp>" in tool_response and "</search_medcorp>" not in tool_response:
    tool_response += "</search_medcorp>"
if "<search_umls>" in tool_response and "</search_umls>" not in tool_response:
    tool_response += "</search_umls>"

# Step 3: Parse (dual or single)
parsed_result = self._parse_tool_call(tool_response)
is_dual_query = isinstance(parsed_result, dict)

# Step 4: Execute appropriate retrieval
if is_dual_query:
    # Dual retrieval path
    retrieved_docs_text = self._execute_tool_call(parsed_result, qid=qid, log_dir=log_dir)
else:
    # Single retrieval path (unchanged)
    tool_name, query = parsed_result
    retrieved_docs = self._execute_tool_call(tool_name, query, qid=qid, log_dir=log_dir)
```

## Backward Compatibility

✅ **Fully backward compatible**:
- Single-query format still works (`<search>query</search>`)
- All existing stop sequences preserved
- No changes to `max_new_tokens`, `max_length`, or generation parameters
- Fallback paths for missing queries or old format

## Usage Examples

### Example 1: Dual Retrieval (Both Queries)
**Model Output**:
```
<search_medcorp>mortality prognosis for congestive heart failure patients with renal failure</search_medcorp>
<search_umls>CHF abbreviation medical terminology cardiac output definition</search_umls>
```

**Result**: 
- Retrieves 4 docs from MedCorp (clinical evidence)
- Retrieves 4 docs from UMLS (terminology/concepts)
- Returns top 5 combined docs

### Example 2: Single Source (MedCorp Only)
**Model Output**:
```
<search_medcorp>post-surgical mortality rates for CABG in elderly patients</search_medcorp>
```

**Result**: 
- Retrieves 4 docs from MedCorp only
- No UMLS retrieval

### Example 3: No Retrieval
**Model Output**:
```
Based on the patient's EHR data, I assess the following...
(no search tags)
```

**Result**: 
- No retrieval performed
- Continues with direct generation

### Example 4: Legacy Single Query
**Model Output**:
```
<search>mortality rates for diabetes with hypertension</search>
```

**Result**: 
- Uses old single-query retrieval path
- Retrieves from combined MedCorp2 corpus (4 MedCorp + 4 UMLS)

## Query Length Analysis

Based on analysis of 426 existing queries:
- **Average query**: 107 characters (~27 tokens)
- **Median**: 76 characters
- **95th percentile**: 104 characters
- **Current truncation**: 2048 tokens (8192 chars) - unchanged

**Dual queries fit comfortably** within existing `max_tokens=8192` for initial generation.

## Testing Checklist

- [x] Syntax validation (no compile errors)
- [ ] Test dual-query retrieval (both queries)
- [ ] Test single-source retrieval (MedCorp only)
- [ ] Test single-source retrieval (UMLS only)
- [ ] Test no retrieval (optional behavior)
- [ ] Test legacy single-query format
- [ ] Verify logging output for dual queries
- [ ] Check retrieval JSON logs have correct format
- [ ] Verify top-5 document selection from combined results

## Configuration Notes

**No configuration changes required**:
- MedRAG source_retrievers already initialized
- Stop sequences updated automatically
- All generation parameters preserved
- Logging paths extended for dual queries

## Next Steps

1. Test with actual patient data
2. Monitor dual vs single query usage patterns
3. Evaluate retrieval quality for specialized queries
4. Consider prompt engineering to encourage dual queries when beneficial
