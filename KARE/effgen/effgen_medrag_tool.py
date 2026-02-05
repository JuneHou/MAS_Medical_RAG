#!/usr/bin/env python3
"""
Custom MedRAG Retrieval Tool for effGen Framework
Wraps MedRAG retrieval system for use with effGen agents
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add MedRAG paths
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))

try:
    from effgen.tools.base import BaseTool
except ImportError:
    print("Warning: effgen not installed. This tool requires effgen.")
    BaseTool = object


class MedRAGRetrievalTool(BaseTool):
    """
    Custom retrieval tool that wraps MedRAG system for effGen agents.
    Supports both single-query and dual-query (MedCorp + UMLS) retrieval.
    """
    
    def __init__(self, 
                 medrag_instance,
                 k: int = 8,
                 max_query_tokens: int = 2048,
                 log_dir: Optional[str] = None):
        """
        Initialize MedRAG retrieval tool.
        
        Args:
            medrag_instance: Pre-initialized MedRAG instance
            k: Number of documents to retrieve
            max_query_tokens: Maximum query length in tokens
            log_dir: Directory for logging retrievals
        """
        # Call parent init without arguments
        super().__init__()
        
        # Set tool metadata as attributes (effgen BaseTool expects these)
        self.name = "retrieve_medical_evidence"
        self.description = "Retrieve medical evidence from MedCorp2 corpus and UMLS knowledge base. Use this tool to find clinical evidence, prognosis information, and medical terminology."
        
        # Define tool parameters (accept both 'input' and 'query' for compatibility)
        self.parameters = {
            "input": {
                "type": "string",
                "description": "Medical query to search for in the knowledge base",
                "required": False
            },
            "query": {
                "type": "string",
                "description": "Medical query to search for (alternative parameter name)",
                "required": False
            },
            "qid": {
                "type": "string",
                "description": "Query ID for logging (optional)",
                "required": False
            }
        }
        
        # Set our custom attributes
        self.medrag = medrag_instance
        self.k = k
        self.max_query_tokens = max_query_tokens
        self.max_query_chars = max_query_tokens * 4  # Rough estimate
        self.log_dir = log_dir
        
        print(f"[MedRAG Tool] Initialized with k={k}, max_query_tokens={max_query_tokens}")
    
    def execute(self, input: str = None, query: str = None, qid: Optional[str] = None) -> str:
        """
        Execute retrieval and return formatted results.
        
        Args:
            input: Search query (effgen standard parameter name)
            query: Alternative parameter name (for backward compatibility)
            qid: Query ID for logging
            
        Returns:
            Formatted string with retrieved documents
        """
        # Accept both 'input' and 'query' parameter names
        query = input if input is not None else query
        if query is None:
            return "Error: No query provided. Please provide a search query."
        
        try:
            print(f"[MedRAG Tool] Executing retrieval for query length: {len(query)} chars")
            
            # Truncate query if too long
            if len(query) > self.max_query_chars:
                print(f"[MedRAG Tool] Truncating query from {len(query)} to {self.max_query_chars} chars")
                query = query[:self.max_query_chars]
            
            # Use direct retrieval (bypass LLM generation)
            retrieved_snippets, scores = self._retrieve_direct(query)
            
            # Format results
            formatted_docs = self._format_results(retrieved_snippets, scores)
            
            # Log retrieval if requested
            if self.log_dir and qid:
                self._log_retrieval(query, formatted_docs, qid)
            
            print(f"[MedRAG Tool] Retrieved {len(formatted_docs)} documents")
            
            # Return formatted string for effgen
            return self._format_for_agent(formatted_docs)
            
        except Exception as e:
            print(f"[MedRAG Tool ERROR] Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return "Error: Retrieval failed. No medical evidence retrieved."
    
    def _retrieve_direct(self, query: str) -> tuple:
        """
        Direct retrieval bypass (avoid LLM generation issues).
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (snippets, scores)
        """
        # Check if MedRAG has source_retrievers (MedCorp2 format)
        if hasattr(self.medrag, 'source_retrievers') and self.medrag.corpus_name == "MedCorp2":
            print(f"[MedRAG Tool] Using source-specific retrieval for MedCorp2")
            
            # Retrieve from both sources
            k_medcorp = self.k // 2 + self.k % 2
            k_umls = self.k // 2
            
            all_snippets = []
            all_scores = []
            
            for source, k_source in [("medcorp", k_medcorp), ("umls", k_umls)]:
                if source in self.medrag.source_retrievers:
                    retrieval_system = self.medrag.source_retrievers[source]
                    snippets, scores = retrieval_system.retrieve(query, k=k_source, rrf_k=60)
                    all_snippets.extend(snippets)
                    all_scores.extend(scores)
            
            return all_snippets, all_scores
            
        elif hasattr(self.medrag, 'retrieval_system') and self.medrag.retrieval_system:
            print(f"[MedRAG Tool] Using unified retrieval system")
            return self.medrag.retrieval_system.retrieve(query, k=self.k, rrf_k=60)
        else:
            raise ValueError("MedRAG instance has no retrieval system")
    
    def _format_results(self, snippets: List[Dict], scores: List[float]) -> List[Dict]:
        """
        Format retrieved snippets for output.
        
        Args:
            snippets: Retrieved document snippets
            scores: Relevance scores
            
        Returns:
            List of formatted document dictionaries
        """
        formatted_docs = []
        for i, doc in enumerate(snippets):
            formatted_docs.append({
                'id': i + 1,
                'title': doc.get('title', 'Unknown'),
                'content': doc.get('content', ''),
                'score': scores[i] if i < len(scores) else 0.0,
                'source': doc.get('source_type', 'unknown')
            })
        return formatted_docs
    
    def _format_for_agent(self, formatted_docs: List[Dict]) -> str:
        """
        Format documents as string for agent consumption.
        
        Args:
            formatted_docs: List of formatted documents
            
        Returns:
            Formatted string with all documents
        """
        if not formatted_docs:
            return "No relevant medical evidence found."
        
        doc_strings = []
        for doc in formatted_docs:
            doc_str = f"[Document {doc['id']}] (Score: {doc['score']:.3f}, Source: {doc['source']})\n"
            doc_str += f"Title: {doc['title']}\n"
            doc_str += f"{doc['content'][:1000]}..."  # Limit content length
            doc_strings.append(doc_str)
        
        return "\n\n".join(doc_strings)
    
    def _log_retrieval(self, query: str, formatted_docs: List[Dict], qid: str):
        """
        Log retrieval results to file.
        
        Args:
            query: Search query
            formatted_docs: Formatted documents
            qid: Query ID
        """
        try:
            log_path = Path(self.log_dir) / f"retrieve_{qid}.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_data = {
                'query_id': qid,
                'query': query,
                'query_length': len(query),
                'num_retrieved': len(formatted_docs),
                'k_requested': self.k,
                'retrieved_documents': formatted_docs,
                'timestamp': time.time()
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"[MedRAG Tool] Logged retrieval to {log_path}")
            
        except Exception as e:
            print(f"[MedRAG Tool] Failed to log retrieval: {e}")


class DualQueryMedRAGTool(MedRAGRetrievalTool):
    """
    Extended MedRAG tool supporting dual-query retrieval.
    Allows separate queries for MedCorp and UMLS sources.
    """
    
    def __init__(self, *args, **kwargs):
        # Call parent init
        super().__init__(*args, **kwargs)
        
        # Override name and description for dual-query mode
        self.name = "retrieve_dual_medical_evidence"
        self.description = "Retrieve medical evidence using separate queries for MedCorp (clinical literature) and UMLS (terminology). Provide queries as JSON: {\"medcorp\": \"query1\", \"umls\": \"query2\"}"
        
        # Override parameters for dual-query mode (accept multiple names for compatibility)
        self.parameters = {
            "input": {
                "type": "string",
                "description": "Medical query for dual-source retrieval. Can be a plain query string or JSON with 'medcorp' and 'umls' keys for source-specific queries",
                "required": False
            },
            "query": {
                "type": "string",
                "description": "Medical query (alternative parameter name)",
                "required": False
            },
            "queries": {
                "type": "string",
                "description": "Medical queries for dual sources (alternative parameter name)",
                "required": False
            },
            "qid": {
                "type": "string",
                "description": "Query ID for logging (optional)",
                "required": False
            }
        }
    
    def execute(self, input: str = None, query: str = None, queries: str = None, qid: Optional[str] = None) -> str:
        """
        Execute dual-query retrieval.
        
        Args:
            input: Query input (effgen standard parameter name) - can be plain string or JSON
            query: Alternative single parameter name
            queries: Alternative parameter name for dual queries
            qid: Query ID for logging
            
        Returns:
            Formatted string with retrieved documents
        """
        try:
            print(f"[Dual MedRAG Tool] Executing dual-query retrieval")
            
            # Accept multiple parameter names for compatibility
            query_input = input if input is not None else (query if query is not None else queries)
            if query_input is None:
                return "Error: No query provided. Please provide a search query."
            
            # Parse input - handle string, JSON string, or dict
            if isinstance(query_input, str):
                try:
                    # Try to parse as JSON
                    queries_dict = json.loads(query_input)
                except json.JSONDecodeError:
                    # Plain string - use for both sources
                    print(f"[Dual MedRAG Tool] Using plain query for both sources")
                    queries_dict = {"medcorp": query_input, "umls": query_input}
            elif isinstance(query_input, dict):
                queries_dict = query_input
            else:
                # Fallback: convert to string and use for both
                queries_dict = {"medcorp": str(query_input), "umls": str(query_input)}
            
            medcorp_query = queries_dict.get('medcorp')
            umls_query = queries_dict.get('umls')
            
            if not medcorp_query and not umls_query:
                return "Error: No queries provided for retrieval."
            
            all_snippets = []
            all_scores = []
            
            # Retrieve from MedCorp
            if medcorp_query and hasattr(self.medrag, 'source_retrievers'):
                if "medcorp" in self.medrag.source_retrievers:
                    truncated = medcorp_query[:self.max_query_chars]
                    print(f"[Dual MedRAG] Retrieving from MedCorp: {len(truncated)} chars")
                    
                    snippets, scores = self.medrag.source_retrievers["medcorp"].retrieve(
                        truncated, k=4, rrf_k=60
                    )
                    all_snippets.extend(snippets)
                    all_scores.extend(scores)
                    print(f"[Dual MedRAG] Retrieved {len(snippets)} from MedCorp")
            
            # Retrieve from UMLS
            if umls_query and hasattr(self.medrag, 'source_retrievers'):
                if "umls" in self.medrag.source_retrievers:
                    truncated = umls_query[:self.max_query_chars]
                    print(f"[Dual MedRAG] Retrieving from UMLS: {len(truncated)} chars")
                    
                    snippets, scores = self.medrag.source_retrievers["umls"].retrieve(
                        truncated, k=4, rrf_k=60
                    )
                    all_snippets.extend(snippets)
                    all_scores.extend(scores)
                    print(f"[Dual MedRAG] Retrieved {len(snippets)} from UMLS")
            
            # Format results
            formatted_docs = self._format_results(all_snippets, all_scores)
            
            # Log if requested
            if self.log_dir and qid:
                log_path = Path(self.log_dir) / f"retrieve_dual_{qid}.json"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(log_path, 'w') as f:
                    json.dump({
                        'query_id': qid,
                        'medcorp_query': medcorp_query,
                        'umls_query': umls_query,
                        'num_retrieved': len(formatted_docs),
                        'documents': formatted_docs
                    }, f, indent=2)
            
            print(f"[Dual MedRAG Tool] Retrieved {len(formatted_docs)} total documents")
            
            return self._format_for_agent(formatted_docs)
            
        except Exception as e:
            print(f"[Dual MedRAG Tool ERROR] Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return "Error: Dual retrieval failed. No medical evidence retrieved."


# Test the tool
if __name__ == "__main__":
    print("Testing MedRAG Tool...")
    
    try:
        # Initialize MedRAG
        sys.path.insert(0, "/data/wang/junh/githubs/mirage_medrag/MedRAG/src")
        from medrag import MedRAG
        
        print("Initializing MedRAG...")
        medrag = MedRAG(
            llm_name="Qwen/Qwen2.5-7B-Instruct",
            rag=True,
            retriever_name="MedCPT",
            corpus_name="MedCorp2",
            db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
            corpus_cache=True,
            HNSW=True,
            retriever_device="cuda:0"
        )
        
        # Create tool
        print("Creating MedRAG tool...")
        tool = MedRAGRetrievalTool(medrag, k=8)
        
        # Test retrieval
        print("Testing retrieval...")
        query = "What are the risk factors for mortality in patients with acute myocardial infarction?"
        result = tool.execute(query, qid="test_001")
        
        print(f"\nRetrieval Result:\n{result[:500]}...")
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error testing tool: {e}")
        import traceback
        traceback.print_exc()
