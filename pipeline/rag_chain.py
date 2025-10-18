import os
import requests
from dotenv import load_dotenv
from typing import List, Dict
from retriever import EmbeddingRetriever

class RAGChain:
    def __init__(self, retriever: EmbeddingRetriever, model_name: str = "phi3"):  # ← Changez ici
        """Initialize RAG chain with retriever and local Ollama model"""
        self.retriever = retriever
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string with metadata"""
        context = "\n\nArticles pertinents du Code de Travail Tunisien:\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format article header with metadata
            header = f"\nArticle {doc['article_number']}"
            
            if doc['chapter_number'] != 'N/A':
                header += f" (Chapitre {doc['chapter_number']}"
                if doc['chapter_title'] != 'N/A':
                    header += f": {doc['chapter_title']}"
                header += ")"
            
            # Add status if abrogated
            if doc['status'] == 'abrogé':
                header += f" - ⚠️ ABROGÉ par {doc['abrogated_by']}"
            elif doc.get('modified_by'):
                header += f" - Modifié par {doc['modified_by']}"
            
            context += f"{header}:\n{doc['chunk'].strip()}\n"
        
        return context
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for the LLM in French"""
        return f"""[INST] Tu es un assistant juridique spécialisé dans le Code de Travail Tunisien.

    INSTRUCTIONS IMPORTANTES:
    1. Réponds de manière CONCISE et DIRECTE (maximum 150 mots)
    2. Va droit au but sans répétitions
    3. Cite l'article source pour chaque information (ex: "Selon Article 14")
    4. Structure ta réponse en 2-3 paragraphes maximum
    5. Utilise UNIQUEMENT les articles fournis ci-dessous
    6. Si un article est abrogé (⚠️ ABROGÉ), mentionne-le brièvement
    7. Si tu ne peux pas répondre avec les articles fournis, dis simplement "Je ne peux pas répondre à cette question avec les articles fournis."

    {context}

    Question: {query}

    Réponse concise: [/INST]"""
    
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
            return f"Erreur lors de l'appel à Ollama: {str(e)}"
    
    def generate_answer(self, query: str, k: int = 3) -> dict:
        """Generate answer using RAG pipeline"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, k=k)
            
            # Format context with metadata
            context = self._format_context(retrieved_docs)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate answer
            answer = self._generate_with_ollama(prompt)
            
            return {
                'query': query,
                'answer': answer,
                'context': retrieved_docs
            }
        except Exception as e:
            return {
                'query': query,
                'answer': f"Erreur lors de la génération de la réponse: {str(e)}",
                'context': []
            }

def main():
    """Example usage of the RAG chain"""
    load_dotenv()
    
    index_path = "faiss_index.index"
    metadata_path = "chunks_metadata.pkl"
    
    # Initialize retriever with metadata
    retriever = EmbeddingRetriever(index_path, metadata_path)
    
    # Initialize RAG chain
    rag_chain = RAGChain(retriever, model_name="phi3")
    
    print("\nBienvenue dans l'assistant Code de Travail Tunisien")
    print("Posez vos questions (tapez 'exit' pour quitter):")
    
    while True:
        query = input("\nQuestion: ").strip()
        
        if query.lower() == 'exit':
            break
        
        # Generate answer
        result = rag_chain.generate_answer(query)
        
        print("\n" + "="*80)
        print("RÉPONSE GÉNÉRÉE")
        print("="*80)
        print(result['answer'])
        
        print("\n" + "="*80)
        print("ARTICLES UTILISÉS")
        print("="*80)
        for i, ctx in enumerate(result['context'], 1):
            print(f"\n{i}. Article {ctx['article_number']}", end="")
            
            if ctx['chapter_number'] != 'N/A':
                chapter_info = f" - Chapitre {ctx['chapter_number']}"
                if ctx['chapter_title'] != 'N/A':
                    chapter_info += f" ({ctx['chapter_title']})"
                print(chapter_info)
            else:
                print()
            
            print(f"   Score de similarité: {ctx['similarity_score']:.4f}")
            
            if ctx['status'] == 'abrogé':
                print(f"   ⚠️  STATUS: Abrogé par {ctx['abrogated_by']}")
            
            print(f"   Extrait: {ctx['chunk'][:200]}...")

if __name__ == "__main__":
    main()
