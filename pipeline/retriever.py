import os
import faiss
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

class EmbeddingRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize the retriever with FAISS index and metadata
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to pickle file containing chunks metadata
        """
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
       
        # Load chunks metadata
        self.chunks_metadata = self._load_metadata(metadata_path)
       
        # Initialize model and tokenizer
        self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_metadata(self, metadata_path: str) -> List[Dict]:
        """Load chunks metadata from pickle file"""
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text query"""
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
           
            # Convert to numpy and normalize
            embedding = embedding.cpu().numpy()
            faiss.normalize_L2(embedding)
           
            return embedding
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve k most similar chunks for the query with full metadata
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of dictionaries containing chunk text, metadata, and similarity score
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
       
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
       
        # Format results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            metadata = self.chunks_metadata[idx]
            result = {
                'chunk': metadata.get('text', ''),
                'similarity_score': float(dist),
                'article_number': metadata.get('article_number', 'N/A'),
                'chapter_number': metadata.get('chapter_number', 'N/A'),
                'chapter_title': metadata.get('chapter_title', 'N/A'),
                'source': metadata.get('source', 'N/A'),
                'status': metadata.get('status', 'actif'),
                'modified_by': metadata.get('modified_by'),
                'abrogated_by': metadata.get('abrogated_by')
            }
            results.append(result)
       
        return results

def main():
    """
    Example usage of the retriever
    """
    index_path = "faiss_index.index"
    metadata_path = "chunks_metadata.pkl"
   
    # Initialize retriever
    retriever = EmbeddingRetriever(index_path, metadata_path)
   
    # Interactive query loop
    print("\nEnter your questions (type 'exit' to quit):")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == 'exit':
            break
           
        results = retriever.retrieve(query)
       
        print("\nRelevant passages:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['similarity_score']:.4f}")
            print(f"   Article {result['article_number']}", end="")
            
            if result['chapter_number'] != 'N/A':
                chapter_info = f" - Chapitre {result['chapter_number']}"
                if result['chapter_title'] != 'N/A':
                    chapter_info += f" ({result['chapter_title']})"
                print(chapter_info)
            else:
                print()
            
            if result['status'] == 'abrogé':
                print(f"    STATUS: Abrogé par {result['abrogated_by']}")
            
            print(f"   Text: {result['chunk'][:200]}...")

if __name__ == "__main__":
    main()