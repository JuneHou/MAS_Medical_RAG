"""
Patch for Search-R1 generation.py to support local MedRAG retriever.

Add this code to /data/wang/junh/githubs/Search-R1/search_r1/llm_agent/generation.py

Replace the _batch_search method (around line 450) with this modified version:
"""

def _batch_search(self, queries):
    """
    Batch search with support for both HTTP and local retriever.
    
    Automatically detects retriever type from config:
    - retriever.local=true: Use in-process LocalMedRAGRetriever
    - retriever.local=false (or not set): Use HTTP server (default)
    """
    
    # Check if local retriever is enabled
    use_local = getattr(self.config, 'local', False)
    
    if use_local:
        # Local in-process retriever (for SLURM jobs)
        if not hasattr(self, '_local_retriever'):
            # Lazy initialization
            print(f"[Search-R1] Initializing local MedRAG retriever on GPU {self.config.gpu_id}...")
            
            import sys
            sys.path.insert(0, '/data/wang/junh/githubs/Debate/KARE/searchr1')
            from local_medrag_retriever import get_retriever
            
            self._local_retriever = get_retriever(
                corpus_name=getattr(self.config, 'corpus_name', 'MedCorp'),
                retriever_name=getattr(self.config, 'retriever_name', 'MedCPT'),
                db_dir=getattr(self.config, 'db_dir', '/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus'),
                gpu_id=self.config.gpu_id
            )
            print("[Search-R1] Local retriever ready!")
        
        # Call local retriever
        return self._local_retriever.retrieve(
            queries=queries,
            topk=self.config.topk,
            return_scores=True
        )
    
    else:
        # HTTP-based retriever (original implementation)
        import requests
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        # Use 'url' field (backwards compatible)
        url = getattr(self.config, 'url', getattr(self.config, 'search_url', None))
        if url is None:
            raise ValueError("retriever.url must be set when using HTTP mode")
        
        return requests.post(url, json=payload).json()
