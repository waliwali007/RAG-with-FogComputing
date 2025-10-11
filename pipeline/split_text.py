# code avec overlap
import os
import tiktoken

def split_text_file(input_file, output_dir, max_tokens=500, overlap=0.20):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get the encoding and allow special tokens
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    chunk = []
    i = 0

    while i < len(words):
        chunk.append(words[i])
        # Check if we've reached max_tokens
        if len(enc.encode(" ".join(chunk), disallowed_special=())) > max_tokens:
            # Save the chunk without the last word
            chunks.append(" ".join(chunk[:-1]))
            
            # Calculate overlap size
            overlap_size = int(len(chunk[:-1]) * overlap)
            
            # Start next chunk with overlapping words
            if overlap_size > 0:
                chunk = chunk[-overlap_size-1:]  # Keep overlap + the word that exceeded limit
            else:
                chunk = [words[i]]  # Just keep the current word
        else:
            i += 1

    # Add the last chunk if it exists
    if chunk:
        chunks.append(" ".join(chunk))

    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_dir, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)
    
    print(f"Created {len(chunks)} chunks with {int(overlap*100)}% overlap")

# Example usage
if __name__ == "__main__":
    print("Splitting text file into chunks...")
    split_text_file("output/cleaned_text.txt", "chunks", max_tokens=500, overlap=0.20)


    
# code sans overlap
# import os
# import tiktoken

# def split_text_file(input_file, output_dir, max_tokens=500):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         text = f.read()

#     # Get the encoding and allow special tokens
#     enc = tiktoken.get_encoding("cl100k_base")
#     words = text.split()
#     chunks = []
#     chunk = []

#     for word in words:
#         chunk.append(word)
#         # Allow special tokens by passing `disallowed_special=()`
#         if len(enc.encode(" ".join(chunk), disallowed_special=())) > max_tokens:
#             chunks.append(" ".join(chunk[:-1]))
#             chunk = [word]

#     if chunk:
#         chunks.append(" ".join(chunk))

#     os.makedirs(output_dir, exist_ok=True)
#     for i, chunk in enumerate(chunks):
#         with open(os.path.join(output_dir, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
#             f.write(chunk)

# # Example usage
# if __name__ == "__main__":
#     print("Splitting text file into chunks...")
#     split_text_file("output/cleaned_text.txt", "chunks", max_tokens=500)

