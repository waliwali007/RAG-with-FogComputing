import os
import json
import re
import tiktoken

def extract_article_number(article_text):
    """
    Extrait le numéro d'article en gérant tous les cas spéciaux
    """
    # Cas 1: Article premier
    if re.match(r'Articles?\s+premier', article_text, re.IGNORECASE):
        return "1"
    
    # Cas 2: Articles X à Y (pluriel)
    match = re.match(r'Articles\s+(\d+)\s+à\s+(\d+)', article_text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    # Cas 3: Article X à Y (singulier)
    match = re.match(r'Article\s+(\d+)\s+à\s+(\d+)', article_text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    # Cas 4: Article avec tirets (6-2, 6-3, etc.)
    match = re.match(r'Articles?\s+(\d+(?:-\d+)?)', article_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Cas 5: Article bis, ter, quater
    match = re.match(r'Articles?\s+(\d+)\s+(bis|ter|quater|quinquies)', article_text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    return "unknown"

def extract_chapter_info(text):
    """
    Extrait le numéro et le titre du chapitre
    """
    patterns = [
        r'Chapitre\s+([IVXLCDM]+)\s*[:\-]?\s*(.+?)(?=\n|$)',
        r'Chapitre\s+(\d+)\s*[:\-]?\s*(.+?)(?=\n|$)',
        r'Chapitre\s+([IVXLCDM]+)\s*$',
        r'Chapitre\s+(\d+)\s*$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            chapter_number = match.group(1)
            chapter_title = match.group(2).strip() if len(match.groups()) > 1 else None
            return chapter_number, chapter_title
    
    return None, None

def split_text_by_single_article(input_file, output_dir, source_name="Code de Travail Tunisien"):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    enc = tiktoken.get_encoding("cl100k_base")
    
    # Split sur "Article" OU "Articles" (pluriel)
    # On utilise une regex pour split tout en conservant le mot
    parts = re.split(r'(Articles?\s+)', text)
    
    chunks = []
    current_chapter_number = None
    current_chapter_title = None
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Premier segment (avant le premier article)
        if i == 0 and not re.match(r'Articles?\s+', part):
            if part.strip():
                chapter_num, chapter_title = extract_chapter_info(part)
                if chapter_num:
                    current_chapter_number = chapter_num
                    current_chapter_title = chapter_title
                
                chunks.append({
                    "article_number": "preambule",
                    "source": source_name,
                    "text": part.strip()
                })
            i += 1
            continue
        
        # Si c'est "Article" ou "Articles", on le combine avec la partie suivante
        if re.match(r'Articles?\s+', part) and i + 1 < len(parts):
            article_text = part + parts[i + 1]
            i += 2
        else:
            i += 1
            continue
        
        # Vérifie s'il y a un nouveau chapitre
        chapter_num, chapter_title = extract_chapter_info(article_text)
        if chapter_num:
            current_chapter_number = chapter_num
            current_chapter_title = chapter_title
        
        # Extrait le numéro d'article
        article_number = extract_article_number(article_text)
        
        # Extrait les modifications
        modification_match = re.search(r'\(Modifié par la loi n°([^)]+)\)', article_text)
        modification = modification_match.group(1) if modification_match else None
        
        # Extrait abrogation
        abrogation_match = re.search(r'\(Abrogé[és]* par la loi n°([^)]+)\)', article_text)
        abrogation = abrogation_match.group(1) if abrogation_match else None
        
        # Nettoie et ajoute avec métadonnées
        if article_text.strip():
            chunk_metadata = {
                "article_number": article_number,
                "source": source_name,
                "text": article_text.strip()
            }
            
            # Ajoute le chapitre
            if current_chapter_number:
                chunk_metadata["chapter_number"] = current_chapter_number
                if current_chapter_title:
                    chunk_metadata["chapter_title"] = current_chapter_title
            
            # Ajoute métadonnées optionnelles
            if modification:
                chunk_metadata["modified_by"] = modification
            if abrogation:
                chunk_metadata["abrogated_by"] = abrogation
                chunk_metadata["status"] = "abrogé"
            
            chunks.append(chunk_metadata)
    
    # Sauvegarde les chunks
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        token_count = len(enc.encode(chunk["text"], disallowed_special=()))
        
        with open(os.path.join(output_dir, f"chunk_{i+1}.json"), "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        status = f" [{chunk.get('status', 'actif')}]" if 'status' in chunk else ""
        chapter_info = f" - Chapitre {chunk.get('chapter_number', 'N/A')}" if 'chapter_number' in chunk else ""
        print(f"Chunk {i+1}: Article {chunk['article_number']}{chapter_info}{status}, {token_count} tokens")
    
    print(f"\nTotal: {len(chunks)} chunks (1 article par chunk)")

# Example usage
if __name__ == "__main__":
    print("Splitting text file - 1 article per chunk with chapter metadata...\n")
    split_text_by_single_article("output/cleaned_text.txt", "chunks", 
                                 source_name="Code de Travail Tunisien")