import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import faiss
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take average of all tokens
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_chunks(directory):
    """
    Load JSON chunks with metadata from the specified directory.
    Returns both the texts and their metadata.
    """
    chunks_data = []
    filenames = sorted(os.listdir(directory), key=lambda x: int(x.split('_')[1].split('.')[0]) if x.startswith('chunk_') else 0)
    
    for filename in filenames:
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks_data.append(data)
    
    return chunks_data

def embed_chunks(chunks_data, batch_size=32):
    """
    Generate embeddings using HuggingFace model.
    Takes list of chunk dictionaries with metadata.
    """
    # Load model and tokenizer
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    embeddings = []
    
    # Extract texts for embedding
    texts = [chunk['text'] for chunk in chunks_data]
    
    # Process chunks in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        
        # Tokenize batch
        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)

def save_faiss_index(embeddings, chunks_data, output_index_file, output_metadata_file):
    """
    Save the embeddings to a FAISS index file and metadata to a separate file.
    """
    # Ensure embeddings are float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    
    # Normalize the vectors to unit length for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create an Inner Product index (for cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    # Save index to file
    faiss.write_index(index, output_index_file)
    
    # Save metadata separately using pickle
    with open(output_metadata_file, 'wb') as f:
        pickle.dump(chunks_data, f)

def main():
    """
    Main function to load chunks, generate embeddings, and save them to a FAISS index with metadata.
    """
    chunks_dir = "chunks"
    output_index_file = "faiss_index.index"
    output_metadata_file = "chunks_metadata.pkl"

    print("Starting embedding process...")

    # Load JSON chunks with metadata
    if not os.path.exists(chunks_dir):
        print(f"Error: Directory '{chunks_dir}' does not exist.")
        return

    chunks_data = load_chunks(chunks_dir)
    if not chunks_data:
        print(f"Error: No JSON chunks found in '{chunks_dir}'.")
        return

    print(f"Loaded {len(chunks_data)} chunks from '{chunks_dir}'")
    
    # Display sample metadata
    if chunks_data:
        sample = chunks_data[0]
        print(f"\nSample chunk metadata:")
        print(f"  Article: {sample.get('article_number', 'N/A')}")
        print(f"  Chapter: {sample.get('chapter_number', 'N/A')}")
        print(f"  Source: {sample.get('source', 'N/A')}")
        print(f"  Text preview: {sample.get('text', '')[:100]}...\n")

    # Embed chunks
    embeddings = embed_chunks(chunks_data)
    print(f"Generated embeddings for {len(chunks_data)} chunks")

    # Save embeddings and metadata
    save_faiss_index(embeddings, chunks_data, output_index_file, output_metadata_file)
    print(f"Saved FAISS index to '{output_index_file}'")
    print(f"Saved metadata to '{output_metadata_file}'")

    print("\nEmbedding process completed.")
    print(f"Total chunks processed: {len(chunks_data)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()