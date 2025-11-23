import streamlit as st
import time
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import rag_chain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_chain import RAGChain
from pipeline.retriever import EmbeddingRetriever

# Page configuration
st.set_page_config(
    page_title="Assistant Juridique Tunisien",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, elegant design
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .main {
        background-color: #fafbfc;
    }
    .st-emotion-cache-10p9htt {
    display: flex;
    -webkit-box-pack: justify;
    justify-content: space-between;
    -webkit-box-align: center;
    align-items: center;
    margin-bottom: 1rem;
    height: 0.75rem;
}
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7f8fa 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    /* Mode buttons container */
    .mode-selector {
        display: flex;
        gap: 12px;
        margin-bottom: 30px;
        padding: 8px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    /* Timer styling */
    .timer-box {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    .timer-label {
        font-size: 13px;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .timer-value {
        font-size: 32px;
        font-weight: 600;
        color: #1f2937;
    }
    
    /* Question input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        padding: 14px;
        font-size: 15px;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Answer box styling */
    .answer-box {
        background: white;
        padding: 30px;
        border-radius: 12px;
        border-left: 3px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin: 25px 0;
    }
    
    /* Context card styling */
    .context-card {
        background: white;
        padding: 24px;
        border-radius: 10px;
        margin: 12px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }
    
    .context-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #d1d5db;
    }
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 40px;
        border-radius: 12px;
        margin-bottom: 35px;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 12px 0 0 0;
        opacity: 0.95;
        font-size: 16px;
        font-weight: 400;
    }
    
    /* Logo placeholder */
    .logo-placeholder {
        background: white;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 30px 20px;
        text-align: center;
        margin: 0 0 25px 0;
        background: linear-gradient(180deg, #ffffff 0%, #f7f8fa 100%);
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    
    /* Description box */
    .description-box {
        background: white;
        padding: 24px;
        border-radius: 10px;
        margin: 0 0 25px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    .description-box h3 {
        color: #1f2937;
        margin: 0 0 16px 0;
        font-size: 18px;
        font-weight: 600;
    }
    
    .description-box p {
        color: #4b5563;
        line-height: 1.7;
        margin: 0 0 12px 0;
        font-size: 14px;
    }
    
    .description-box ul {
        margin: 16px 0 0 0;
        padding-left: 0;
        list-style: none;
    }
    
    .description-box li {
        padding: 8px 0;
        color: #4b5563;
        font-size: 14px;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .description-box li:last-child {
        border-bottom: none;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.2);
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 8px;
        color: #1e40af;
        font-size: 14px;
        font-weight: 500;
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 3px solid #22c55e;
        padding: 16px 20px;
        border-radius: 8px;
        color: #166534;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Section divider */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 30px 0;
    }
    
    /* Footer info */
    .footer-info {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        border-top: 1px solid #e5e7eb;
    }
    
    .footer-info p {
        margin: 8px 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'basique'
if 'query_time' not in st.session_state:
    st.session_state.query_time = 0.0
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize RAG Chain
@st.cache_resource
def initialize_rag():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # pipeline is inside the project folder where app.py lives
        pipeline_dir = os.path.join(script_dir, "pipeline")
        
        index_path = os.path.join(pipeline_dir, "faiss_index.index")
        metadata_path = os.path.join(pipeline_dir, "chunks_metadata.pkl")
        
        # sanity checks with clearer errors
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Faiss index not found at: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
        retriever = EmbeddingRetriever(index_path, metadata_path)
        rag_chain = RAGChain(retriever, model_name="mistral")
        return rag_chain, None
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    # Logo Placeholder
    st.markdown("""
    <div class="logo-placeholder">
        <a href="https://ibb.co/DDZB2FB6"><img src="https://i.ibb.co/tM90r60N/fog.png" alt="fog" border="0">
    </div>
    """, unsafe_allow_html=True)
    
    
    
    # Mode Selection (moved from main)
    st.markdown('<div style="margin-bottom: 12px; font-weight: 600; color: #1f2937; font-size: 15px;">MODE DE TRAITEMENT</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Mode Basique", use_container_width=True, type="primary" if st.session_state.mode == 'basique' else "secondary"):
            st.session_state.mode = 'basique'
            st.rerun()
    
    with col2:
        if st.button("Mode Distribué", use_container_width=True, type="primary" if st.session_state.mode == 'distribué' else "secondary"):
            st.session_state.mode = 'distribué'
            st.rerun()
    
    # Mode info
    if st.session_state.mode == 'basique':
        st.markdown('<div class="info-box">Mode Basique: Traitement local standard</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">Mode Distribué: Architecture fog computing haute performance</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Timer Display with placeholder for dynamic updates
    st.markdown('<div class="timer-label">TEMPS DE RÉPONSE</div>', unsafe_allow_html=True)
    timer_placeholder = st.empty()
    
    # Initial timer display
    with timer_placeholder.container():
        st.markdown(f"""
        <div class="timer-box">
            <div class="timer-value">{st.session_state.query_time:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Info
    st.markdown("""
    <div class="footer-info">
        <p style="font-size: 12px; opacity: 0.7;">© 2024 - Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize RAG if not already done
if st.session_state.rag_chain is None:
    with st.spinner("Initialisation du système..."):
        rag_chain, error = initialize_rag()
        if error:
            st.error(f"Erreur d'initialisation: {error}")
            st.stop()
        st.session_state.rag_chain = rag_chain
        st.success("Système initialisé avec succès")

# Main content - Chat-like interface
st.markdown('<div style="margin-bottom: 20px; font-weight: 600; color: #1f2937; font-size: 18px;">Poserez votre question</div>', unsafe_allow_html=True)

query = st.text_input(
    "Posez votre question juridique",
    placeholder="Ex: Quelle est la durée légale hebdomadaire du travail?",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("Rechercher", use_container_width=True, type="primary")

if search_button and query:
    # Check if RAG chain is initialized
    if st.session_state.rag_chain is None:
        st.error(" Le système n'est pas initialisé correctement. Veuillez rafraîchir la page.")
        st.stop()
    
    st.session_state.is_processing = True
    start_time = time.time()

    if st.session_state.mode == 'basique':
        try:
            with st.spinner("Analyse en cours..."):
                result = st.session_state.rag_chain.generate_answer(query, k=3)

            # STOP TIMER
            end_time = time.time()
            st.session_state.query_time = end_time - start_time
            st.session_state.is_processing = False

            # --- DISPLAY ANSWER ---
            st.markdown('<div style="margin: 35px 0 15px 0; font-weight: 600; color: #1f2937; font-size: 16px;">RÉPONSE GÉNÉRÉE</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)

            # --- DISPLAY REFERENCES ---
            if result['context']:
                st.markdown('<div style="margin: 35px 0 15px 0; font-weight: 600; color: #1f2937; font-size: 16px;">ARTICLES RÉFÉRENCÉS</div>', unsafe_allow_html=True)
                
                for i, ctx in enumerate(result['context'], 1):
                    article_num = ctx.get('article_number', 'N/A')
                    chapter_num = ctx.get('chapter_number', 'N/A')
                    chapter_title = ctx.get('chapter_title', 'N/A')
                    similarity = ctx.get('similarity_score', 0)
                    chunk_text = ctx.get('chunk', '')[:300]

                    chapter_info = ""
                    if chapter_num != 'N/A':
                        chapter_info = f"Chapitre {chapter_num}"
                        if chapter_title != 'N/A':
                            chapter_info += f" - {chapter_title}"

                    st.markdown(f"""
                    <div class="context-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="font-weight: 600; color: #1f2937; font-size: 15px;">Article {article_num}</div>
                            <div style="font-size: 13px; color: #6b7280;">Pertinence: {similarity:.2%}</div>
                        </div>
                        {f'<div style="color: #6b7280; font-size: 13px; margin-bottom: 12px;">{chapter_info}</div>' if chapter_info else ''}
                        <div style="color: #4b5563; line-height: 1.7; font-size: 14px;">{chunk_text}...</div>
                    </div>
                    """, unsafe_allow_html=True)
        
            # Update timer in sidebar WITHOUT rerun
            with timer_placeholder.container():
                st.markdown(f"""
                <div class="timer-box">
                    <div class="timer-value">{st.session_state.query_time:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.session_state.is_processing = False
            st.error(f" Erreur lors de la génération de la réponse: {str(e)}")

    else:
        # Distributed mode (placeholder for now)
        st.info("Mode distribué: Fonctionnalité en développement")
        st.session_state.is_processing = False

elif search_button and not query:
    st.warning("Veuillez saisir une question.")