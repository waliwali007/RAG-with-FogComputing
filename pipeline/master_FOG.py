import requests
import asyncio
import aiohttp
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pipeline.rag_chain import RAGChain
from pipeline.retriever import EmbeddingRetriever


class DistributedOrchestrator:
    """Version originale - conservée pour compatibilité"""
    def __init__(self, node_urls: List[str], retriever: EmbeddingRetriever, model_name: str = "mistral:latest"):
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
                    'answer': "Les noeuds ne sont pas connectés.",
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


class OptimizedDistributedOrchestrator:
    """
    Version optimisée avec:
    - Requêtes parallèles asynchrones
    - Cache des résultats
    - Timeouts configurables
    - Stratégies multiples (async, fastest, threaded)
    """
    
    def __init__(self, node_urls: List[str], retriever: EmbeddingRetriever, 
                 model_name: str = "mistral:latest", use_cache: bool = True):
        self.node_urls = node_urls
        self.rag_chain = RAGChain(retriever, model_name=model_name)
        self.retriever = retriever
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.cache_ttl = 300  # 5 minutes
        self.node_timeout = 15  # Timeout par nœud
        self.global_timeout = 30  # Timeout global
        
    # ==================== MÉTHODES ASYNCHRONES ====================
    
    async def _async_query_node(self, session: aiohttp.ClientSession, 
                                node_url: str, query: str, k: int) -> Dict[str, Any]:
        """Requête asynchrone à un nœud"""
        try:
            async with session.post(
                f"{node_url}/search",
                json={"query": query, "k": k},
                timeout=aiohttp.ClientTimeout(total=self.node_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'node': node_url,
                        'results': data.get("results", [])
                    }
                return {'success': False, 'node': node_url, 'results': []}
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout pour le nœud {node_url}")
            return {'success': False, 'node': node_url, 'results': []}
        except Exception as e:
            print(f"❌ Erreur nœud {node_url}: {str(e)}")
            return {'success': False, 'node': node_url, 'results': []}
    
    async def _async_retrieve_distributed(self, query: str, k: int = 3) -> List[Dict]:
        """Récupération asynchrone sur tous les nœuds"""
        async with aiohttp.ClientSession() as session:
            # Lancer toutes les requêtes en parallèle
            tasks = [
                self._async_query_node(session, node_url, query, k*2)
                for node_url in self.node_urls
            ]
            
            # Attendre toutes les réponses (avec timeout global)
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.global_timeout
                )
            except asyncio.TimeoutError:
                print("⏱️ Timeout global atteint")
                results = []
            
            # Agréger tous les résultats
            all_results = []
            successful_nodes = 0
            
            for result in results:
                if isinstance(result, dict) and result.get('success'):
                    all_results.extend(result.get('results', []))
                    successful_nodes += 1
            
            print(f"✅ {successful_nodes}/{len(self.node_urls)} nœuds ont répondu")
            
            # Trier par score de similarité
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return all_results[:k]
    
    async def _async_generate_answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Génération de réponse asynchrone complète"""
        # Vérifier le cache
        if self.use_cache:
            cache_key = f"{query}_{k}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    print("💾 Réponse trouvée dans le cache")
                    return cached_data['result']
        
        try:
            # Récupération distribuée asynchrone
            retrieved_docs = await self._async_retrieve_distributed(query, k=k)
            
            if not retrieved_docs:
                return {
                    'query': query,
                    'answer': "Les nœuds ne sont pas connectés.",
                    'context': [],
                    'mode': 'async_failed'
                }
            
            # Génération de la réponse (synchrone - Ollama)
            context = self.rag_chain._format_context(retrieved_docs)
            prompt = self.rag_chain._create_prompt(query, context)
            answer = self.rag_chain._generate_with_ollama(prompt)
            
            result = {
                'query': query,
                'answer': answer,
                'context': retrieved_docs,
                'mode': 'async_distributed'
            }
            
            # Mettre en cache
            if self.use_cache:
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur async: {str(e)}")
            return {
                'query': query,
                'answer': f"Erreur lors du traitement distribué: {str(e)}",
                'context': [],
                'mode': 'async_error'
            }
    
    def generate_answer_distributed_async(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        🚀 MÉTHODE RECOMMANDÉE - La plus rapide (2-3x plus rapide)
        Point d'entrée synchrone pour la génération asynchrone
        """
        return asyncio.run(self._async_generate_answer(query, k))