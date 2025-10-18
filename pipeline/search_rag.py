import faiss
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def search_faiss_with_metadata(query, index_file, metadata_file, top_k=5):
    """
    Search FAISS index and return results with full metadata
    """
    # Load FAISS index
    index = faiss.read_index(index_file)
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        chunks_metadata = pickle.load(f)
    
    # Load model for query embedding
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Embed query
    encoded_input = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        query_embedding = query_embedding.cpu().numpy().astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve results with metadata
    results = []
    for i, idx in enumerate(indices[0]):
        metadata = chunks_metadata[idx]
        result = {
            "rank": i + 1,
            "score": float(distances[0][i]),
            "article_number": metadata.get('article_number', 'N/A'),
            "chapter_number": metadata.get('chapter_number', 'N/A'),
            "chapter_title": metadata.get('chapter_title', 'N/A'),
            "source": metadata.get('source', 'N/A'),
            "text": metadata.get('text', ''),
            "status": metadata.get('status', 'actif'),
            "modified_by": metadata.get('modified_by'),
            "abrogated_by": metadata.get('abrogated_by')
        }
        results.append(result)
    
    return results

if __name__ == "__main__":
    query = "contrat de travail à durée déterminée"
    results = search_faiss_with_metadata(
        query, 
        "faiss_index.index", 
        "chunks_metadata.pkl",
        top_k=3
    )
    
    print(f"Query: {query}\n")
    for result in results:
        print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
        print(f"Article {result['article_number']}", end="")
        
        if result['chapter_number'] != 'N/A':
            chapter_info = f" - Chapitre {result['chapter_number']}"
            if result['chapter_title'] != 'N/A':
                chapter_info += f" ({result['chapter_title']})"
            print(chapter_info)
        else:
            print()
        
        if result['status'] == 'abrogé':
            print(f"⚠️  STATUS: Abrogé par {result['abrogated_by']}")
        
        print(f"Text: {result['text'][:200]}...")
        print("-" * 80)