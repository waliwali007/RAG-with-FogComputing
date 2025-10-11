import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take average of all tokens
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_chunks(directory):
    """
    Load text chunks from the specified directory.
    """
    chunks = []
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                chunks.append(f.read())
    return chunks

def embed_chunks(chunks, batch_size=32):
    """
    Generate embeddings using HuggingFace model
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
    
    # Process chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        
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

def save_faiss_index(embeddings, output_file):
    """
    Save the embeddings to a FAISS index file using cosine similarity.
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
    faiss.write_index(index, output_file)

def main():
    """
    Main function to load chunks, generate embeddings, and save them to a FAISS index.
    """
    chunks_dir = "chunks"
    output_file = "faiss_index.index"

    print("Starting embedding process...")

    # Load text chunks
    if not os.path.exists(chunks_dir):
        print(f"Error: Directory '{chunks_dir}' does not exist.")
        return

    chunks = load_chunks(chunks_dir)
    if not chunks:
        print(f"Error: No text chunks found in '{chunks_dir}'.")
        return

    print(f"Loaded {len(chunks)} chunks from '{chunks_dir}'")

    # Embed chunks
    embeddings = embed_chunks(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks")

    # Save embeddings to FAISS index
    save_faiss_index(embeddings, output_file)
    print(f"Saved FAISS index to '{output_file}'")

    print("Embedding process completed.")

if __name__ == "__main__":
    main()