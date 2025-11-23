from flask import Flask, request, jsonify
import sys
import os
from pathlib import Path
from pipeline.retriever import EmbeddingRetriever

app = Flask(__name__)

# module-level retriever
retriever = None

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    k = data.get("k", 3)

    if not query:
        return jsonify({"error": "Missing query"}), 400
    if retriever is None:
        return jsonify({"error": "Retriever not initialized"}), 500

    results = retriever.retrieve(query, k=k)

    return jsonify({
        "node": os.environ.get("NODE_NAME", "node"),
        "query": query,
        "results": results,
        "count": len(results)
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "retriever_initialized": retriever is not None
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Resolve paths to absolute paths
    index_path = Path(args.index).resolve()
    metadata_path = Path(args.metadata).resolve()
    
    # Validate files exist
    if not index_path.exists():
        print(f"Error: Index file not found at {index_path}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)

    retriever = EmbeddingRetriever(str(index_path), str(metadata_path))

    print(f"Node server running on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)