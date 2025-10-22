import os
import requests
from dotenv import load_dotenv
from typing import List, Dict
from retriever import EmbeddingRetriever

class RAGChain:
    def __init__(self, retriever: EmbeddingRetriever, model_name: str = "mistral"):
        """Initialize RAG chain with retriever and local Ollama model (optimized for Mistral)"""
        self.retriever = retriever
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string with metadata"""
        context = "\n\nArticles pertinents du Code de Travail Tunisien:\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format article header with metadata
            header = f"\nArticle {doc.get('article_number', 'N/A')}"
            
            if doc.get('chapter_number', 'N/A') != 'N/A':
                header += f" (Chapitre {doc['chapter_number']}"
                if doc.get('chapter_title', 'N/A') != 'N/A':
                    header += f": {doc['chapter_title']}"
                header += ")"
            
            # Add status if abrogated
            if doc.get('status') == 'abrogé':
                header += f" - ⚠️ ABROGÉ par {doc.get('abrogated_by', 'N/A')}"
            elif doc.get('modified_by'):
                header += f" - Modifié par {doc['modified_by']}"
            
            context += f"{header}:\n{doc.get('chunk', '').strip()}\n"
        
        return context
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for Mistral in French"""
        return f"""<s>[INST] Tu es un assistant juridique expert spécialisé dans le Code de Travail Tunisien.

INSTRUCTIONS IMPORTANTES:
1. Réponds de manière CONCISE et DIRECTE (maximum 150 mots)
2. Va droit au but sans répétitions
3. Cite OBLIGATOIREMENT l'article source pour chaque information (ex: "Selon l'Article 14")
4. Structure ta réponse en 2-3 paragraphes maximum
5. Utilise UNIQUEMENT les articles fournis ci-dessous - ne spécule pas
6. Si un article est abrogé (⚠️ ABROGÉ), mentionne-le clairement
7. Si les articles fournis ne permettent pas de répondre, dis simplement "Je ne peux pas répondre à cette question avec les articles fournis."
8. Utilise un langage juridique précis et professionnel

{context}

Question: {query}

Réponse concise et précise: [/INST]"""
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama and get response (optimized for Mistral)"""
        try:
            # Vérifier que le service Ollama est accessible
            response = requests.post(
                self.api_url, 
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "num_predict": 200,
                    "stop": ["</s>", "[INST]"],
                    "stream": False
                },
                timeout=60  # Timeout de 60 secondes
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Vérifier que la réponse contient bien du texte
            generated_text = data.get("response", "").strip()
            
            if not generated_text:
                return "Aucune réponse générée par le modèle. Veuillez réessayer."
            
            return generated_text
            
        except requests.exceptions.ConnectionError:
            return "❌ Erreur: Impossible de se connecter à Ollama. Assurez-vous que le service Ollama est démarré (ollama serve)."
        except requests.exceptions.Timeout:
            return "❌ Erreur: Le modèle a mis trop de temps à répondre. Veuillez réessayer."
        except requests.exceptions.HTTPError as e:
            return f"❌ Erreur HTTP lors de l'appel à Ollama: {str(e)}\nVérifiez que le modèle '{self.model_name}' est installé (ollama pull {self.model_name})."
        except Exception as e:
            return f"❌ Erreur inattendue lors de l'appel à Ollama: {str(e)}"
    
    def generate_answer(self, query: str, k: int = 3) -> dict:
        """Generate answer using RAG pipeline"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, k=k)
            
            if not retrieved_docs:
                return {
                    'query': query,
                    'answer': "❌ Aucun document pertinent trouvé pour cette question.",
                    'context': []
                }
            
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
                'answer': f"❌ Erreur lors de la génération de la réponse: {str(e)}",
                'context': []
            }

def main():
    """Example usage of the RAG chain with Mistral"""
    load_dotenv()
    
    index_path = "faiss_index.index"
    metadata_path = "chunks_metadata.pkl"
    
    try:
        # Initialize retriever with metadata
        retriever = EmbeddingRetriever(index_path, metadata_path)
        
        # Initialize RAG chain with Mistral
        rag_chain = RAGChain(retriever, model_name="mistral")
        
        print("\n" + "="*80)
        print("🤖 Assistant Code de Travail Tunisien - Propulsé par Mistral 7B")
        print("="*80)
        print("\n💡 Conseils:")
        print("  • Posez des questions précises sur le Code du Travail")
        print("  • L'assistant citera les articles pertinents")
        print("  • Tapez 'exit' pour quitter")
        print("  • Tapez 'help' pour des exemples de questions")
        print("\n" + "="*80)
        
        # Exemples de questions
        example_questions = [
            "Quelle est la durée légale hebdomadaire du travail?",
            "Quels sont les différents types de congés?",
            "Quelle est la procédure de licenciement pour faute grave?",
            "Quelles sont les obligations de l'employeur en matière de sécurité?"
        ]
        
        while True:
            query = input("\n❓ Question: ").strip()
            
            if query.lower() == 'exit':
                print("\n👋 Au revoir!")
                break
            
            if query.lower() == 'help':
                print("\n📋 Exemples de questions:")
                for i, q in enumerate(example_questions, 1):
                    print(f"   {i}. {q}")
                continue
            
            if not query:
                print("⚠️  Veuillez poser une question.")
                continue
            
            # Generate answer
            print("\n⏳ Recherche et analyse en cours...")
            result = rag_chain.generate_answer(query)
            
            print("\n" + "="*80)
            print("📝 RÉPONSE GÉNÉRÉE")
            print("="*80)
            print(result['answer'])
            
            if result['context']:
                print("\n" + "="*80)
                print("📚 ARTICLES UTILISÉS")
                print("="*80)
                for i, ctx in enumerate(result['context'], 1):
                    print(f"\n{i}. Article {ctx.get('article_number', 'N/A')}", end="")
                    
                    if ctx.get('chapter_number', 'N/A') != 'N/A':
                        chapter_info = f" - Chapitre {ctx['chapter_number']}"
                        if ctx.get('chapter_title', 'N/A') != 'N/A':
                            chapter_info += f" ({ctx['chapter_title']})"
                        print(chapter_info)
                    else:
                        print()
                    
                    print(f"   📊 Score de similarité: {ctx.get('similarity_score', 0):.4f}")
                    
                    if ctx.get('status') == 'abrogé':
                        print(f"   ⚠️  STATUS: Abrogé par {ctx.get('abrogated_by', 'N/A')}")
                    
                    chunk_text = ctx.get('chunk', '')
                    print(f"   📄 Extrait: {chunk_text[:200]}...")
            
            print("\n" + "="*80)
    
    except FileNotFoundError as e:
        print(f"❌ Erreur: Fichier introuvable - {e}")
        print("Assurez-vous que les fichiers 'faiss_index.index' et 'chunks_metadata.pkl' existent.")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {str(e)}")

if __name__ == "__main__":
    main()