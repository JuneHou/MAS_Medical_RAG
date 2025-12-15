"""
Parse retrieval JSON logs for MedRAG retrieved documents.

This script loads and formats retrieved medical documents from JSON files
to include in the training context.
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """Container for a single retrieved document."""
    id: str
    title: str
    content: str
    score: float
    source: str


class RetrievalLogParser:
    """Parser for retrieval JSON log files with hierarchical fallback."""
    
    def __init__(self, primary_retrieval_dir: str, fallback_retrieval_dir: Optional[str] = None):
        """
        Initialize parser with primary and optional fallback retrieval directories.
        
        Args:
            primary_retrieval_dir: Primary retrieval logs directory
            fallback_retrieval_dir: Optional fallback retrieval logs directory
        """
        self.primary_retrieval_dir = primary_retrieval_dir
        self.fallback_retrieval_dir = fallback_retrieval_dir
        
        if not os.path.exists(self.primary_retrieval_dir):
            raise ValueError(f"Primary retrieval directory not found: {self.primary_retrieval_dir}")
        
        if fallback_retrieval_dir and not os.path.exists(fallback_retrieval_dir):
            print(f"Warning: Fallback retrieval directory not found: {fallback_retrieval_dir}")
    
    def parse_retrieval_file(self, filepath: str) -> Optional[List[RetrievedDocument]]:
        """
        Parse a single retrieval JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of RetrievedDocument objects or None if parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
        
        if not isinstance(data, dict):
            print(f"Invalid JSON structure in {filepath}")
            return None
        
        # Extract retrieved documents
        retrieved_docs = []
        
        if "retrieved_documents" in data:
            for doc in data["retrieved_documents"]:
                retrieved_docs.append(RetrievedDocument(
                    id=doc.get("id", ""),
                    title=doc.get("title", ""),
                    content=doc.get("content", ""),
                    score=doc.get("score", 0.0),
                    source=doc.get("source", "")
                ))
        
        return retrieved_docs
    
    def get_retrieved_documents(
        self,
        patient_id: str,
        visit_id: str
    ) -> Optional[List[RetrievedDocument]]:
        """
        Get retrieved documents for a specific patient/visit with hierarchical fallback.
        
        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            
        Returns:
            List of RetrievedDocument objects or None if file not found in both locations
        """
        filename = f"retrieve_mortality_assessment_{patient_id}_{visit_id}.json"
        
        # Try primary directory first
        primary_filepath = os.path.join(self.primary_retrieval_dir, filename)
        if os.path.exists(primary_filepath):
            return self.parse_retrieval_file(primary_filepath)
        
        # Try fallback directory if configured
        if self.fallback_retrieval_dir:
            fallback_filepath = os.path.join(self.fallback_retrieval_dir, filename)
            if os.path.exists(fallback_filepath):
                return self.parse_retrieval_file(fallback_filepath)
        
        # File not found in either location
        return None
    
    def format_retrieved_documents(
        self,
        documents: List[RetrievedDocument],
        max_docs: int = 5
    ) -> str:
        """
        Format retrieved documents into a readable string.
        
        Args:
            documents: List of retrieved documents
            max_docs: Maximum number of documents to include
            
        Returns:
            Formatted string of retrieved documents
        """
        if not documents:
            return "No retrieved documents available."
        
        formatted = []
        for i, doc in enumerate(documents[:max_docs], 1):
            formatted.append(f"Document {i}:")
            formatted.append(f"Title: {doc.title}")
            formatted.append(f"Source: {doc.source}")
            formatted.append(f"Content: {doc.content}")
            formatted.append(f"Relevance Score: {doc.score:.4f}")
            formatted.append("")
        
        return "\n".join(formatted)


def test_parser():
    """Test the retrieval parser on a single example."""
    primary_retrieval_dir = "/data/wang/junh/githubs/Debate/KARE/results/arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8/debate_logs"
    fallback_retrieval_dir = "/data/wang/junh/githubs/Debate/KARE/results/fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8/debate_logs"
    
    parser = RetrievalLogParser(primary_retrieval_dir, fallback_retrieval_dir)
    
    # Test on patient 34_0
    docs = parser.get_retrieved_documents("34", "0")
    
    if docs:
        print(f"Successfully parsed {len(docs)} retrieved documents")
        print("\nFormatted output (first 2 docs):")
        print(parser.format_retrieved_documents(docs, max_docs=2))
    else:
        print("Failed to parse retrieval documents")


if __name__ == "__main__":
    test_parser()
