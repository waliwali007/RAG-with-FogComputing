import os
import tiktoken

def split_text_file(input_file, output_dir, max_tokens=500):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get the encoding and allow special tokens
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        # Allow special tokens by passing `disallowed_special=()`
        if len(enc.encode(" ".join(chunk), disallowed_special=())) > max_tokens:
            chunks.append(" ".join(chunk[:-1]))
            chunk = [word]

    if chunk:
        chunks.append(" ".join(chunk))

    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_dir, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)

# Example usage
if __name__ == "__main__":
    print("Splitting text file into chunks...")
    split_text_file("output/cleaned_text.txt", "chunks", max_tokens=500)