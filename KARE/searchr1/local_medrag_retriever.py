"""
Local MedRAG retriever that runs in-process without HTTP server.
For use in SLURM jobs where running background services is difficult.
"""

import sys
import torch
from typing import List, Dict, Any
sys.path.insert(0, '/data/wang/junh/githubs/mirage_medrag/MedRAG/src')

from medrag import MedRAG


class LocalMedRAGRetriever:
    """
    In-process MedRAG retriever compatible with Search-R1.
    Replaces HTTP server for SLURM environments.
    """
    
    def __init__(
        self,
        corpus_name: str = "MedCorp",
        retriever_name: str = "MedCPT",
        db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
        gpu_id: int = 0
    ):
        """
        Initialize MedRAG retriever on specified GPU.
        
        Args:
            corpus_name: Medical corpus to use
            retriever_name: Retrieval model name
            db_dir: Path to corpus database
            gpu_id: GPU ID to use for retrieval (should NOT overlap with training GPUs)
        """
        print(f"[LocalMedRAG] Initializing on GPU {gpu_id}...")
        
        # Set GPU
        self.device = f"cuda:{gpu_id}"
        
        # Initialize MedRAG
        self.rag = MedRAG(
            llm_name=None,  # We don't need LLM, only retriever
            rag=True,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir,
            cache_dir=None
        )
        
        # Move retriever to specified GPU
        if hasattr(self.rag.retriever, 'model'):
            self.rag.retriever.model = self.rag.retriever.model.to(self.device)
        
        print(f"[LocalMedRAG] Ready on {self.device}")
        print(f"[LocalMedRAG] Corpus: {corpus_name}, Retriever: {retriever_name}")
    
    def retrieve(
        self,
        queries: List[str],
        topk: int = 5,
        return_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve documents for batch of queries.
        Compatible with Search-R1's HTTP retriever interface.
        
        Args:
            queries: List of search queries
            topk: Number of documents to retrieve per query
            return_scores: Whether to return relevance scores
            
        Returns:
            {
                "result": [
                    [
                        {
                            "document": {"contents": "..."},
                            "score": 0.95
                        },
                        ...
                    ],
                    ...
                ]
            }
        """
        if not queries:
            return {"result": []}
        
        # Retrieve documents for each query
        all_results = []
        
        with torch.no_grad():
            for query in queries:
                # MedRAG retrieve method
                retrieved_snippets = self.rag.retrieve(
                    question=query,
                    k=topk
                )
                
                # Format to match HTTP server response
                formatted_results = []
                for snippet in retrieved_snippets:
                    formatted_results.append({
                        "document": {
                            "contents": snippet["content"] if isinstance(snippet, dict) else snippet
                        },
                        "score": snippet.get("score", 1.0) if isinstance(snippet, dict) else 1.0
                    })
                
                all_results.append(formatted_results)
        
        return {"result": all_results}
    
    def __call__(self, queries: List[str], topk: int = 5) -> Dict[str, Any]:
        """Convenience method for direct calling."""
        return self.retrieve(queries, topk)


# Singleton instance for Search-R1 to use
_retriever_instance = None


def get_retriever(
    corpus_name: str = "MedCorp",
    retriever_name: str = "MedCPT",
    db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
    gpu_id: int = 0
) -> LocalMedRAGRetriever:
    """
    Get or create singleton retriever instance.
    Ensures only one retriever is loaded (saves memory).
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = LocalMedRAGRetriever(
            corpus_name=corpus_name,
            retriever_name=retriever_name,
            db_dir=db_dir,
            gpu_id=gpu_id
        )
    
    return _retriever_instance
