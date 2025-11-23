from flask import Flask, request, jsonify
import sys
import os
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

    # no global keyword here
    retriever = EmbeddingRetriever(args.index, args.metadata)

    print(f"Node server running on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
