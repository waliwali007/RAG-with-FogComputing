import os
import tiktoken

def split_text_by_single_article(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    enc = tiktoken.get_encoding("cl100k_base")
    
    # Split sur "Article" en préservant le mot "Article"
    parts = text.split("Article")
    
    chunks = []
    
    for i, part in enumerate(parts):
        if i == 0:
            # Texte avant le premier article (préambule, titre, etc.)
            if part.strip():
                chunks.append(part.strip())
            continue
        
        # Reconstitue "Article" + son contenu
        article_text = "Article" + part
        
        # Nettoie et ajoute
        if article_text.strip():
            chunks.append(article_text.strip())
    
    # Sauvegarde les chunks
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        # Calcule le nombre de tokens pour info
        token_count = len(enc.encode(chunk, disallowed_special=()))
        
        with open(os.path.join(output_dir, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)
        
        print(f"Chunk {i+1}: {token_count} tokens")
    
    print(f"\nTotal: {len(chunks)} chunks (1 article par chunk)")

# Example usage
if __name__ == "__main__":
    print("Splitting text file - 1 article per chunk...\n")
    split_text_by_single_article("output/cleaned_text.txt", "chunks")
