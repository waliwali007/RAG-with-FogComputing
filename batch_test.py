import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class BatchTester:
    """Handles batch testing and comparison between basic and distributed modes"""
    
    def __init__(self, rag_chain=None, orchestrator=None):
        self.rag_chain = rag_chain
        self.orchestrator = orchestrator
    
    def run_single_query(self, query, mode='basique', k=3):
        """Run a single query and return timing + result"""
        start_time = time.time()
        
        try:
            if mode == 'basique' and self.rag_chain:
                result = self.rag_chain.generate_answer(query, k=k)
            elif mode == 'distribué' and self.orchestrator:
                result = self.orchestrator.generate_answer_distributed(query, k=k)
            else:
                return None, None, "System not initialized"
            
            elapsed = time.time() - start_time
            return result, elapsed, None
        except Exception as e:
            elapsed = time.time() - start_time
            return None, elapsed, str(e)
    
    def run_batch_sequential(self, queries, mode='basique', k=3):
        """Run queries sequentially"""
        results = []
        for query in queries:
            result, elapsed, error = self.run_single_query(query, mode, k)
            results.append({
                'query': query,
                'time': elapsed,
                'success': error is None,
                'error': error
            })
        return results
    
    def run_batch_concurrent(self, queries, mode='basique', k=3, max_workers=5):
        """Run queries concurrently (simulates multiple users)"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(self.run_single_query, query, mode, k): query 
                for query in queries
            }
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result, elapsed, error = future.result()
                    results.append({
                        'query': query,
                        'time': elapsed,
                        'success': error is None,
                        'error': error
                    })
                except Exception as e:
                    results.append({
                        'query': query,
                        'time': 0,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    @staticmethod
    def calculate_statistics(results):
        """Calculate statistics from results"""
        times = [r['time'] for r in results if r['success']]
        
        if not times:
            return None
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'total': sum(times),
            'success_rate': sum(1 for r in results if r['success']) / len(results) * 100
        }


def create_comparison_chart(basic_stats, distributed_stats):
    """Create a comparison chart"""
    categories = ['Temps Moyen', 'Temps Médian', 'Temps Min', 'Temps Max', 'Temps Total']
    
    basic_values = [
        basic_stats['mean'],
        basic_stats['median'],
        basic_stats['min'],
        basic_stats['max'],
        basic_stats['total']
    ]
    
    distributed_values = [
        distributed_stats['mean'],
        distributed_stats['median'],
        distributed_stats['min'],
        distributed_stats['max'],
        distributed_stats['total']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Mode Basique', x=categories, y=basic_values, marker_color='#3b82f6'),
        go.Bar(name='Mode Distribué', x=categories, y=distributed_values, marker_color='#22c55e')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Comparaison des Performances',
        yaxis_title='Temps (secondes)',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_throughput_chart(basic_results, distributed_results):
    """Create throughput comparison over time"""
    basic_times = [r['time'] for r in basic_results if r['success']]
    distributed_times = [r['time'] for r in distributed_results if r['success']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=basic_times,
        mode='lines+markers',
        name='Mode Basique',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        y=distributed_times,
        mode='lines+markers',
        name='Mode Distribué',
        line=dict(color='#22c55e', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Temps de Réponse par Requête',
        xaxis_title='Numéro de la Requête',
        yaxis_title='Temps (secondes)',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


# Sample test queries for Tunisian labor code
SAMPLE_QUERIES = [
    "Quelle est la durée légale hebdomadaire du travail?",
    "Quels sont les droits aux congés payés?",
    "Quelle est la période d'essai légale?",
    "Quelles sont les règles concernant le préavis de démission?",
    "Quels sont les cas de licenciement abusif?",
    "Quelle est l'indemnité de licenciement?",
    "Quelles sont les heures supplémentaires autorisées?",
    "Quels sont les jours fériés légaux?",
    "Quelle est la durée du congé de maternité?",
    "Quels sont les droits syndicaux des travailleurs?"
]