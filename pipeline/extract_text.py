import fitz  # PyMuPDF
import os  # For directory handling

def extract_text_from_pdf(pdf_path, output_file):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text("text")

    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

# Example usage
if __name__ == "__main__":
    print("Extracting text from PDF...")
    extract_text_from_pdf("data/code_de_travail.pdf", "output/extracted_text.txt")