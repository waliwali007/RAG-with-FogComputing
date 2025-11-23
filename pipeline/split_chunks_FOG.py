import pickle
import os

def split_metadata_file(input_path="chunks_metadata.pkl"):
    # Load full metadata list
    with open(input_path, "rb") as f:
        metadata = pickle.load(f)

    # Compute split point
    mid = len(metadata) // 2

    # Split into two halves
    part1 = metadata[:mid]
    part2 = metadata[mid:]

    # Output file names
    base_dir = os.path.dirname(input_path)
    out1 = os.path.join(base_dir, "chunks_metadata1.pkl")
    out2 = os.path.join(base_dir, "chunks_metadata2.pkl")

    # Save
    with open(out1, "wb") as f:
        pickle.dump(part1, f)

    with open(out2, "wb") as f:
        pickle.dump(part2, f)

    print(f"Saved {len(part1)} items → {out1}")
    print(f"Saved {len(part2)} items → {out2}")

if __name__ == "__main__":
    split_metadata_file()
