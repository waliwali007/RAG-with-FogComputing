import requests
import asyncio
import aiohttp
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import nest_asyncio
from pipeline.rag_chain import RAGChain
from pipeline.retriever import EmbeddingRetriever

# IMPORTANT: Permet d'utiliser asyncio.run() dans Streamlit
try:
    nest_asyncio.apply()
except:
    pass


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
    - Requêtes parallèles asynchrones (CORRIGÉES pour Streamlit)
    - Cache des résultats
    - Timeouts configurables
    - Gestion d'erreurs robuste
    """
    
    def __init__(self, node_urls: List[str], retriever: EmbeddingRetriever, 
                 model_name: str = "mistral:latest", use_cache: bool = True):
        self.node_urls = node_urls
        self.rag_chain = RAGChain(retriever, model_name=model_name)
        self.retriever = retriever
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.cache_ttl = 300  # 5 minutes
        self.node_timeout = 15  # Timeout par nœud (réduit de 30s à 15s)
        self.global_timeout = 25  # Timeout global (réduit pour plus de réactivité)
        
        # ThreadPoolExecutor pour les appels synchrones (Ollama)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    # ==================== MÉTHODES ASYNCHRONES CORRIGÉES ====================
    
    async def _async_query_node(self, session: aiohttp.ClientSession, 
                                node_url: str, query: str, k: int) -> Dict[str, Any]:
        """Requête asynchrone à un nœud avec gestion d'erreurs robuste"""
        try:
            async with session.post(
                f"{node_url}/search",
                json={"query": query, "k": k},
                timeout=aiohttp.ClientTimeout(total=self.node_timeout),
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'node': node_url,
                        'results': data.get("results", [])
                    }
                else:
                    print(f"⚠️ Nœud {node_url}: Status {response.status}")
                    return {'success': False, 'node': node_url, 'results': []}
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout pour le nœud {node_url}")
            return {'success': False, 'node': node_url, 'results': [], 'error': 'timeout'}
        except aiohttp.ClientError as e:
            print(f"❌ Erreur réseau nœud {node_url}: {str(e)}")
            return {'success': False, 'node': node_url, 'results': [], 'error': str(e)}
        except Exception as e:
            print(f"❌ Erreur inattendue nœud {node_url}: {str(e)}")
            return {'success': False, 'node': node_url, 'results': [], 'error': str(e)}
    
    async def _async_retrieve_distributed(self, query: str, k: int = 3) -> tuple[List[Dict], int]:
        """
        Récupération asynchrone sur tous les nœuds
        Retourne: (résultats, nombre de nœuds réussis)
        """
        # Configuration du connector pour réutiliser les connexions
        connector = aiohttp.TCPConnector(
            limit=len(self.node_urls),
            limit_per_host=1,
            ttl_dns_cache=300
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Lancer toutes les requêtes en parallèle
            tasks = [
                self._async_query_node(session, node_url, query, k*2)
                for node_url in self.node_urls
            ]
            
            # Attendre toutes les réponses avec timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.global_timeout
                )
            except asyncio.TimeoutError:
                print("⏱️ Timeout global atteint")
                # Récupérer les résultats partiels si possible
                results = [task.result() for task in tasks if task.done()]
            
            # Agréger tous les résultats
            all_results = []
            successful_nodes = 0
            
            for result in results:
                if isinstance(result, dict) and result.get('success'):
                    all_results.extend(result.get('results', []))
                    successful_nodes += 1
            
            print(f"✅ {successful_nodes}/{len(self.node_urls)} nœuds ont répondu")
            
            # Dédupliquer par article_number
            seen_articles = set()
            unique_results = []
            for res in all_results:
                article = res.get('article_number')
                if article and article not in seen_articles:
                    seen_articles.add(article)
                    unique_results.append(res)
                elif not article:
                    unique_results.append(res)
            
            # Trier par score de similarité
            unique_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return unique_results[:k], successful_nodes
    
    async def _async_generate_answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Génération de réponse asynchrone complète"""
        # Vérifier le cache
        if self.use_cache:
            cache_key = f"{query}_{k}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    print("💾 Réponse trouvée dans le cache")
                    cached_result = cached_data['result'].copy()
                    cached_result['from_cache'] = True
                    return cached_result
        
        try:
            # Récupération distribuée asynchrone
            retrieved_docs, successful_nodes = await self._async_retrieve_distributed(query, k=k)
            
            if not retrieved_docs:
                return {
                    'query': query,
                    'answer': "Les nœuds ne sont pas connectés ou n'ont pas retourné de résultats.",
                    'context': [],
                    'mode': 'async_failed',
                    'nodes_used': successful_nodes
                }
            
            # Génération de la réponse dans un thread séparé (non-bloquant)
            loop = asyncio.get_event_loop()
            context = self.rag_chain._format_context(retrieved_docs)
            prompt = self.rag_chain._create_prompt(query, context)
            
            # Exécuter Ollama dans un thread pour ne pas bloquer la boucle async
            answer = await loop.run_in_executor(
                self.executor,
                self.rag_chain._generate_with_ollama,
                prompt
            )
            
            result = {
                'query': query,
                'answer': answer,
                'context': retrieved_docs,
                'mode': 'async_distributed',
                'nodes_used': successful_nodes,
                'from_cache': False
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
            import traceback
            traceback.print_exc()
            return {
                'query': query,
                'answer': f"Erreur lors du traitement distribué: {str(e)}",
                'context': [],
                'mode': 'async_error',
                'error_details': str(e)
            }
    
    def generate_answer_distributed_async(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        🚀 MÉTHODE RECOMMANDÉE - La plus rapide (2-3x plus rapide)
        Point d'entrée synchrone pour la génération asynchrone
        Compatible avec Streamlit grâce à nest_asyncio
        """
        try:
            # Vérifier si une boucle existe déjà
            try:
                loop = asyncio.get_running_loop()
                # Si on est déjà dans une boucle async, créer une tâche
                return asyncio.run_coroutine_threadsafe(
                    self._async_generate_answer(query, k),
                    loop
                ).result()
            except RuntimeError:
                # Pas de boucle en cours, en créer une nouvelle
                return asyncio.run(self._async_generate_answer(query, k))
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution async: {str(e)}")
            # Fallback sur la version synchrone en cas d'erreur
            return self._local_fallback(query, k)
    
    # ==================== STRATÉGIE FASTEST (CORRIGÉE) ====================
    
    async def _async_first_responder(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Utilise la première réponse valide (encore plus rapide)"""
        connector = aiohttp.TCPConnector(limit=len(self.node_urls))
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._async_query_node(session, node_url, query, k)
                for node_url in self.node_urls
            ]
            
            # Attendre la première réponse réussie
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result.get('success') and result.get('results'):
                        print(f"⚡ Premier nœud: {result['node']}")
                        
                        # Utiliser immédiatement ces résultats
                        retrieved_docs = result['results'][:k]
                        
                        # Générer la réponse dans un thread
                        loop = asyncio.get_event_loop()
                        context = self.rag_chain._format_context(retrieved_docs)
                        prompt = self.rag_chain._create_prompt(query, context)
                        answer = await loop.run_in_executor(
                            self.executor,
                            self.rag_chain._generate_with_ollama,
                            prompt
                        )
                        
                        return {
                            'query': query,
                            'answer': answer,
                            'context': retrieved_docs,
                            'fastest_node': result['node'],
                            'mode': 'first_responder'
                        }
                except Exception:
                    continue
            
            # Aucun nœud n'a répondu - fallback local
            return self._local_fallback(query, k)
    
    def generate_answer_fastest(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        ⚡ STRATÉGIE LA PLUS RAPIDE (3-5x plus rapide)
        Utilise la première réponse valide
        """
        try:
            try:
                loop = asyncio.get_running_loop()
                return asyncio.run_coroutine_threadsafe(
                    self._async_first_responder(query, k),
                    loop
                ).result()
            except RuntimeError:
                return asyncio.run(self._async_first_responder(query, k))
        except Exception as e:
            print(f"❌ Erreur fastest: {str(e)}")
            return self._local_fallback(query, k)
    
    # ==================== STRATÉGIE THREADPOOL (PLUS SÛRE) ====================
    
    def _threaded_query_node(self, node_url: str, query: str, k: int) -> Dict[str, Any]:
        """Requête synchrone pour ThreadPool"""
        try:
            response = requests.post(
                f"{node_url}/search",
                json={"query": query, "k": k},
                timeout=self.node_timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return {
                'success': True,
                'node': node_url,
                'results': response.json().get("results", [])
            }
        except requests.Timeout:
            print(f"⏱️ Timeout pour le nœud {node_url}")
            return {'success': False, 'node': node_url, 'results': [], 'error': 'timeout'}
        except Exception as e:
            print(f"❌ Erreur nœud {node_url}: {str(e)}")
            return {'success': False, 'node': node_url, 'results': [], 'error': str(e)}
    
    def generate_answer_threaded(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        🔄 ALTERNATIVE RECOMMANDÉE (1.5-2x plus rapide)
        Utilise ThreadPoolExecutor - PLUS STABLE avec Streamlit
        """
        try:
            # Requêtes parallèles avec ThreadPool
            with ThreadPoolExecutor(max_workers=len(self.node_urls)) as executor:
                futures = {
                    executor.submit(self._threaded_query_node, node_url, query, k*2): node_url
                    for node_url in self.node_urls
                }
                
                all_results = []
                successful_nodes = 0
                
                for future in as_completed(futures, timeout=self.global_timeout):
                    try:
                        result = future.result(timeout=1)  # Timeout additionnel
                        if result.get('success'):
                            all_results.extend(result.get('results', []))
                            successful_nodes += 1
                    except Exception as e:
                        print(f"⚠️ Erreur future: {str(e)}")
                        continue
                
                print(f"✅ {successful_nodes}/{len(self.node_urls)} nœuds ont répondu")
                
                if not all_results:
                    return {
                        'query': query,
                        'answer': "Les nœuds ne sont pas connectés.",
                        'context': [],
                        'mode': 'threaded_failed',
                        'nodes_used': successful_nodes
                    }
                
                # Dédupliquer et trier
                seen_articles = set()
                unique_results = []
                for res in all_results:
                    article = res.get('article_number')
                    if article and article not in seen_articles:
                        seen_articles.add(article)
                        unique_results.append(res)
                    elif not article:
                        unique_results.append(res)
                
                unique_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                retrieved_docs = unique_results[:k]
                
                # Générer la réponse
                context = self.rag_chain._format_context(retrieved_docs)
                prompt = self.rag_chain._create_prompt(query, context)
                answer = self.rag_chain._generate_with_ollama(prompt)
                
                return {
                    'query': query,
                    'answer': answer,
                    'context': retrieved_docs,
                    'nodes_used': successful_nodes,
                    'mode': 'threaded_distributed'
                }
                
        except Exception as e:
            print(f"❌ Erreur threaded: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'query': query,
                'answer': f"Erreur lors du traitement distribué: {str(e)}",
                'context': [],
                'mode': 'threaded_error'
            }
    
    # ==================== MÉTHODE PAR DÉFAUT ====================
    
    def generate_answer_distributed(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Méthode par défaut - Utilise ThreadPool (PLUS STABLE pour Streamlit)
        Change en async si vous êtes sûr de la compatibilité
        """
        return self.generate_answer_threaded(query, k)
    
    # ==================== UTILITAIRES ====================
    
    def _local_fallback(self, query: str, k: int) -> Dict[str, Any]:
        """Fallback sur le retriever local si tous les nœuds échouent"""
        try:
            print("🔄 Fallback sur recherche locale")
            retrieved_docs = self.retriever.retrieve(query, k=k)
            
            context = self.rag_chain._format_context(retrieved_docs)
            prompt = self.rag_chain._create_prompt(query, context)
            answer = self.rag_chain._generate_with_ollama(prompt)
            
            return {
                'query': query,
                'answer': answer,
                'context': retrieved_docs,
                'mode': 'local_fallback'
            }
        except Exception as e:
            return {
                'query': query,
                'answer': f"Erreur: {str(e)}",
                'context': [],
                'mode': 'error'
            }
    
    def clear_cache(self):
        """Vider le cache"""
        if self.use_cache and self.cache:
            self.cache.clear()
            print("🗑️ Cache vidé")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques du cache"""
        if not self.use_cache or not self.cache:
            return {'enabled': False}
        
        valid_entries = sum(
            1 for data in self.cache.values()
            if time.time() - data['timestamp'] < self.cache_ttl
        )
        
        return {
            'enabled': True,
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifier la santé des nœuds (synchrone, rapide)"""
        status = {}
        for node_url in self.node_urls:
            try:
                response = requests.get(
                    f"{node_url}/health", 
                    timeout=2,
                    headers={'Content-Type': 'application/json'}
                )
                status[node_url] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'code': response.status_code
                }
            except requests.Timeout:
                status[node_url] = {
                    'status': 'timeout',
                    'error': 'Timeout after 2s'
                }
            except Exception as e:
                status[node_url] = {
                    'status': 'unreachable',
                    'error': str(e)
                }
        return status
    
    def __del__(self):
        """Cleanup du ThreadPoolExecutor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)