#!/usr/bin/env python3
"""
MedRAG Retrieval Server for Search-R1.
Wraps MedRAG+MedCPT+MedCorp2 as FastAPI server compatible with Search-R1 rollout.
"""

import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging

# Add MedRAG paths
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
sys.path.insert(0, medrag_root)
sys.path.insert(0, os.path.join(medrag_root, "src"))

from medrag import MedRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="MedRAG Retrieval Server for Search-R1")

# Global MedRAG instance
medrag_retriever = None


class QueryRequest(BaseModel):
    """Request schema for retrieval (Search-R1 compatible)"""
    queries: List[str]
    topk: Optional[int] = 8
    return_scores: bool = False


class RetrievalResponse(BaseModel):
    """Response schema for retrieval"""
    result: List[List[Dict[str, str]]]
    scores: Optional[List[List[float]]] = None


def initialize_medrag(
    retriever_name: str = "MedCPT",
    corpus_name: str = "MedCorp2",
    db_dir: str = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"
):
    """Initialize MedRAG retriever once at startup"""
    global medrag_retriever
    
    logger.info("=" * 80)
    logger.info("Initializing MedRAG Retrieval Server for Search-R1")
    logger.info("=" * 80)
    logger.info(f"Retriever: {retriever_name}")
    logger.info(f"Corpus: {corpus_name}")
    logger.info(f"DB Directory: {db_dir}")
    
    try:
        # Use a real model name even though LLM won't be used for retrieval
        # MedRAG requires valid model for tokenizer initialization
        medrag_retriever = MedRAG(
            llm_name="Qwen/Qwen2.5-7B-Instruct",  # Lightweight model, only for tokenizer init
            rag=True,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir
        )
        logger.info("✓ MedRAG initialized successfully")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ Failed to initialize MedRAG: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize MedRAG on server startup"""
    initialize_medrag()


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "MedRAG Retrieval Server for Search-R1",
        "retriever": "MedCPT",
        "corpus": "MedCorp2",
        "version": "1.0"
    }


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Search-R1 compatible retrieval endpoint.
    
    Args:
        request: QueryRequest with queries list and topk
        
    Returns:
        Dict with 'result' key containing retrieved documents in Search-R1 format
    """
    if medrag_retriever is None:
        raise HTTPException(status_code=500, detail="MedRAG not initialized")
    
    try:
        logger.info(f"📥 Retrieval request: {len(request.queries)} queries, topk={request.topk}")
        
        results = []
        scores = []
        
        for idx, query in enumerate(request.queries):
            logger.debug(f"  Query {idx+1}: {query[:100]}...")
            
            # Use direct source-specific retrieval for MedCorp2 (same pattern as working GRPO code)
            # This bypasses LLM generation entirely
            if hasattr(medrag_retriever, 'source_retrievers') and medrag_retriever.corpus_name == "MedCorp2":
                logger.info(f"Using direct source retrieval for MedCorp2 (bypassing LLM generation)")
                
                # Split retrieval between medcorp and umls
                k_medcorp = (request.topk or 8) // 2 + (request.topk or 8) % 2
                k_umls = (request.topk or 8) // 2
                
                all_retrieved_snippets = []
                all_scores = []
                
                for source, k_source in [("medcorp", k_medcorp), ("umls", k_umls)]:
                    if source in medrag_retriever.source_retrievers:
                        logger.debug(f"  Retrieving {k_source} docs from {source}")
                        source_retrieval_system = medrag_retriever.source_retrievers[source]
                        snippets, scores = source_retrieval_system.retrieve(query, k=k_source, rrf_k=60)
                        all_retrieved_snippets.extend(snippets)
                        all_scores.extend(scores)
                
                retrieved_snippets = all_retrieved_snippets
                snippet_scores = all_scores
                
            elif hasattr(medrag_retriever, 'retrieval_system') and medrag_retriever.retrieval_system:
                # Use direct retrieval system (bypass LLM generation)
                logger.info(f"Using direct retrieval system (bypassing LLM generation)")
                retrieved_snippets, snippet_scores = medrag_retriever.retrieval_system.retrieve(
                    query, k=request.topk or 8, rrf_k=60
                )
            else:
                logger.warning("No direct retrieval system found, using fallback")
                retrieved_snippets = []
                snippet_scores = []
            
            # Format for Search-R1
            # Search-R1 expects: List[Dict] with 'document' wrapper containing 'title', 'text', 'contents'
            formatted_docs = []
            
            for i, snippet in enumerate(retrieved_snippets[:request.topk or 8]):
                # Handle both dict and string formats
                if isinstance(snippet, dict):
                    content = snippet.get('content', snippet.get('text', str(snippet)))
                    title = snippet.get('title', f'Document {i+1}')
                else:
                    content = str(snippet)
                    title = f'Document {i+1}'
                
                # Wrap in 'document' key as Search-R1 expects
                formatted_docs.append({
                    'document': {
                        'title': title,
                        'text': content,
                        'contents': content
                    }
                })
            
            results.append(formatted_docs)
            
            if request.return_scores:
                # Use actual scores from retrieval if available
                scores.append(snippet_scores[:request.topk or 8] if snippet_scores else [1.0] * len(formatted_docs))
        
        logger.info(f"✓ Retrieved {sum(len(r) for r in results)} total documents")
        
        # Return in Search-R1 expected format
        response = {"result": results}
        if request.return_scores:
            response["scores"] = scores
            
        return response
    
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.get("/stats")
def stats():
    """Get retriever statistics"""
    if medrag_retriever is None:
        raise HTTPException(status_code=500, detail="MedRAG not initialized")
    
    return {
        "retriever": "MedCPT",
        "corpus": "MedCorp2",
        "status": "ready",
        "backend": "MedRAG"
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MedRAG Retrieval Server for Search-R1')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Server port (default: 8000)')
    parser.add_argument('--retriever', type=str, default='MedCPT', 
                        help='Retriever name (default: MedCPT)')
    parser.add_argument('--corpus', type=str, default='MedCorp2', 
                        help='Corpus name (default: MedCorp2)')
    parser.add_argument('--db_dir', type=str, 
                        default='/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus',
                        help='MedRAG database directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Starting MedRAG Retrieval Server for Search-R1")
    print("=" * 80)
    print(f"Host: {args.host}:{args.port}")
    print(f"Retriever: {args.retriever}")
    print(f"Corpus: {args.corpus}")
    print(f"DB Directory: {args.db_dir}")
    print("=" * 80)
    print()
    print("Why you need this server for Search-R1 but not veRL GRPO:")
    print("  - veRL GRPO: Imports MedRAG directly in Python (in-process)")
    print("  - Search-R1: Rollout workers run separately, need HTTP API")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
