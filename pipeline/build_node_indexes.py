import os
import pickle
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings"""
    token_embeddings = model_output[0]  # First element: last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_chunks_from_metadata(chunks_metadata, batch_size=32):
    """Generate embeddings for chunks loaded from metadata pickle"""
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    texts = [chunk.get('text', '') for chunk in chunks_metadata]
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    return embeddings

def build_faiss_index(embeddings, output_index_path):
    """Build and save FAISS index from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine similarity
    index.add(embeddings)
    faiss.write_index(index, output_index_path)
    print(f"✓ Saved FAISS index to {output_index_path}")

def main():
    script_dir = os.path.dirname(__file__)

    nodes = [
        {"meta": "chunks_metadata1.pkl", "index": "faiss_index_node1.index"},
        {"meta": "chunks_metadata2.pkl", "index": "faiss_index_node2.index"}
    ]

    for node in nodes:
        meta_path = os.path.join(script_dir, node["meta"])
        index_path = os.path.join(script_dir, node["index"])

        if not os.path.exists(meta_path):
            print(f"Error: {meta_path} not found")
            continue

        print(f"\nProcessing {meta_path}")
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"  Loaded {len(metadata)} chunks")

        embeddings = embed_chunks_from_metadata(metadata)
        build_faiss_index(embeddings, index_path)

    print("\n✓ Done! Node indexes created.")

if __name__ == "__main__":
    main()
