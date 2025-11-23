import requests
from typing import List, Dict
from pipeline.rag_chain import RAGChain
from pipeline.retriever import EmbeddingRetriever

class DistributedOrchestrator:
    def __init__(self, node_urls: List[str], retriever: EmbeddingRetriever, model_name: str = "mistral"):
        self.node_urls = node_urls
        self.rag_chain = RAGChain(retriever, model_name=model_name)

    def _query_node(self, node_url: str, query: str, k: int = 3, timeout: int = 30) -> List[Dict]:
        try:
            response = requests.post(
                f"{node_url}/search",
                json={"query": query, "k": k},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception:
            return []

    def retrieve_distributed(self, query: str, k: int = 3) -> List[Dict]:
        all_results = []

        for node_url in self.node_urls:
            node_results = self._query_node(node_url, query, k=k*2)
            all_results.extend(node_results)

        all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return all_results[:k]

    def generate_answer_distributed(self, query: str, k: int = 3) -> dict:
        try:
            retrieved_docs = self.retrieve_distributed(query, k=k)

            if not retrieved_docs:
                return {
                    'query': query,
                    'answer': "Aucun document pertinent trouvé.",
                    'context': []
                }

            context = self.rag_chain._format_context(retrieved_docs)
            prompt = self.rag_chain._create_prompt(query, context)
            answer = self.rag_chain._generate_with_ollama(prompt)

            return {
                'query': query,
                'answer': answer,
                'context': retrieved_docs
            }

        except Exception as e:
            return {
                'query': query,
                'answer': f"Erreur lors du traitement distribué: {str(e)}",
                'context': []
            }
