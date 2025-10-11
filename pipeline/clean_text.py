import re

def clean_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # text = re.sub(r'\[image:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text.strip())

# Example usage
if __name__ == "__main__":
    print("Cleaning text file...")
    clean_text_file("output/extracted_text.txt", "output/cleaned_text.txt")
