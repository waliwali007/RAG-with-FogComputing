import os
import requests
from dotenv import load_dotenv
from typing import List, Dict
from .retriever import EmbeddingRetriever

class RAGChain:
    def __init__(self, retriever: EmbeddingRetriever, model_name: str = "deepseek-coder"):
        """Initialize RAG chain with retriever and local Ollama model"""
        self.retriever = retriever
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string"""
        context = "\n\nRelevant information:\n"
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"\nPassage {i}:\n{doc['chunk'].strip()}\n"
        return context

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for the LLM"""
        return f"""[INST] You are a helpful AI assistant. Use the following passages to answer the question. 
If you cannot answer the question based on the passages, say "I cannot answer this question based on the provided information."

{context}

Question: {query}

Answer: [/INST]"""

    def _generate_with_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama and get response"""
        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15,
                "max_tokens": 512,
                "stream": False
            })
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"

    def generate_answer(self, query: str, k: int = 3) -> dict:
        """Generate answer using RAG pipeline"""
        try:
            retrieved_docs = self.retriever.retrieve(query, k=k)
            context = self._format_context(retrieved_docs)
            prompt = self._create_prompt(query, context)
            answer = self._generate_with_ollama(prompt)

            return {
                'query': query,
                'answer': answer,
                'context': retrieved_docs
            }
        except Exception as e:
            return {
                'query': query,
                'answer': f"Error generating answer: {str(e)}",
                'context': []
            }

def main():
    """Example usage of the RAG chain"""
    load_dotenv()
    index_path = "faiss_index.index"
    chunks_dir = "chunks"

    retriever = EmbeddingRetriever(index_path, chunks_dir)
    rag_chain = RAGChain(retriever, model_name="deepseek-coder")

    print("\nEnter your questions (type 'exit' to quit):")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == 'exit':
            break

        result = rag_chain.generate_answer(query)

        print("\n=== Generated Answer ===")
        print(result['answer'])

        print("\n=== Retrieved Passages ===")
        for i, ctx in enumerate(result['context'], 1):
            print(f"\n{i}. Score: {ctx['similarity_score']:.4f}")
            print(f"Text: {ctx['chunk'][:200]}...")

if __name__ == "__main__":
    main()
