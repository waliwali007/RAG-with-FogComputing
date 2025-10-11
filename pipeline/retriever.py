import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingRetriever:
    def __init__(self, index_path: str, chunks_dir: str):
        """
        Initialize the retriever with FAISS index and model
        """
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        self.chunks = self._load_chunks(chunks_dir)
        
        # Initialize model and tokenizer
        self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_chunks(self, directory: str) -> list:
        """Load all text chunks from the directory"""
        chunks = []
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    chunks.append(f.read())
        return chunks

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

    def retrieve(self, query: str, k: int = 3) -> list:
        """
        Retrieve k most similar chunks for the query
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'chunk': self.chunks[idx],
                'similarity_score': float(dist)
            })
        
        return results

def main():
    """
    Example usage of the retriever
    """
    index_path = "faiss_index.index"
    chunks_dir = "chunks"
    
    # Initialize retriever
    retriever = EmbeddingRetriever(index_path, chunks_dir)
    
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
            print(f"Text: {result['chunk'][:200]}...")

if __name__ == "__main__":
    main()