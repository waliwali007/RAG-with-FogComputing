import os
import json
import re
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
                chunks.append({
                    "article_id": "preambule",
                    "text": part.strip()
                })
            continue
        
        # Reconstitue "Article" + son contenu
        article_text = "Article" + part
        
        # Extrait le numéro d'article avec regex
        article_match = re.match(r'Article\s+(\d+(?:\s+à\s+\d+)?)', article_text)
        article_id = article_match.group(1) if article_match else f"unknown_{i}"
        
        # Nettoie et ajoute avec métadonnées
        if article_text.strip():
            chunks.append({
                "article_id": article_id,
                "text": article_text.strip()
            })
    
    # Sauvegarde les chunks au format JSON
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        # Calcule le nombre de tokens pour info
        token_count = len(enc.encode(chunk["text"], disallowed_special=()))
        
        # Sauvegarde en JSON
        with open(os.path.join(output_dir, f"chunk_{i+1}.json"), "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        print(f"Chunk {i+1}: Article {chunk['article_id']}, {token_count} tokens")
    
    print(f"\nTotal: {len(chunks)} chunks (1 article par chunk)")

# Example usage
if __name__ == "__main__":
    print("Splitting text file - 1 article per chunk with metadata...\n")
    split_text_by_single_article("output/cleaned_text.txt", "chunks")