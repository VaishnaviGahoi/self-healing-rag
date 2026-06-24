import streamlit as st
import os
import json
import time
from datetime import datetime
from rag_engine import SelfHealingRAG

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Healing RAG Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLES ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a14;
    color: #e2e8f0;
}

.stApp { background-color: #0a0a14; }

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}
.day-badge {
    display: inline-block;
    background: linear-gradient(135deg, #a78bfa22, #60a5fa22);
    border: 1px solid #a78bfa55;
    color: #a78bfa;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
}

/* Pipeline stages */
.pipeline-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 1.5rem auto;
    max-width: 720px;
    flex-wrap: wrap;
    gap: 0.3rem;
}
.stage {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    transition: all 0.3s;
}
.stage.active { border-color: #a78bfa; color: #a78bfa; background: #a78bfa11; }
.stage.done { border-color: #34d399; color: #34d399; background: #34d39911; }
.stage.healing { border-color: #f59e0b; color: #f59e0b; background: #f59e0b11; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
.arrow { color: #1e1e30; font-size: 1rem; padding: 0 0.2rem; }

/* Panels */
.panel {
    background: #0d0d1a;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* Thought step */
.thought-step {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.7rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    border-left: 3px solid;
    background: #13131f;
}
.thought-step.think { border-color: #a78bfa; }
.thought-step.search { border-color: #60a5fa; }
.thought-step.eval { border-color: #f59e0b; }
.thought-step.heal { border-color: #f87171; }
.thought-step.answer { border-color: #34d399; }
.step-icon { font-size: 1.1rem; min-width: 1.4rem; }
.step-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    margin-bottom: 0.2rem;
}
.step-text { color: #cbd5e1; line-height: 1.5; }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
}
.score-high { background: #34d39922; color: #34d399; border: 1px solid #34d39944; }
.score-mid { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
.score-low { background: #f8717122; color: #f87171; border: 1px solid #f8717144; }

/* Answer box */
.answer-box {
    background: #0d1f17;
    border: 1px solid #34d39944;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    color: #e2e8f0;
    line-height: 1.7;
    font-size: 0.95rem;
}
.answer-box.healing-answer {
    background: #1a0d0d;
    border-color: #f8717144;
}

/* Chunk card */
.chunk-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.5;
}
.chunk-score {
    float: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}

/* Input styling */
.stTextArea textarea {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px #a78bfa44 !important;
}
.stButton button {
    background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    width: 100% !important;
}
.stButton button:hover { opacity: 0.9 !important; }

/* API key section */
.key-panel {
    background: #0d0d1a;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

/* Iteration badge */
.iter-badge {
    background: #f59e0b22;
    border: 1px solid #f59e0b44;
    color: #f59e0b;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Sample questions */
.sample-q {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    font-size: 0.83rem;
    color: #94a3b8;
    cursor: pointer;
    margin-bottom: 0.4rem;
    transition: all 0.2s;
}
.sample-q:hover { border-color: #a78bfa55; color: #c4b5fd; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""
if "rag" not in st.session_state:
    st.session_state.rag = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "result" not in st.session_state:
    st.session_state.result = None
if "question" not in st.session_state:
    st.session_state.question = ""

# ── CLOUD KEY DETECTION ────────────────────────────────────────────────────────
try:
    cloud_mode = "GROQ_API_KEY" in st.secrets
    if cloud_mode:
        st.session_state.groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    cloud_mode = False

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🧠 Self-Healing RAG Pipeline</div>
    <div class="hero-sub">AI that detects its own weak answers → re-queries → heals itself</div>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE BAR (dynamic) ─────────────────────────────────────────────────────
result = st.session_state.result
stage_classes = ["", "", "", "", ""]
if result:
    for i in range(len(stage_classes)):
        stage_classes[i] = "done"
    if result.get("healed"):
        stage_classes[3] = "done"

stages = ["📥 Ingest", "🔍 Retrieve", "🧮 Evaluate", "🔄 Heal", "✅ Answer"]
stage_html = ""
for i, (s, cls) in enumerate(zip(stages, stage_classes)):
    stage_html += f'<div class="stage {cls}">{s}</div>'
    if i < len(stages) - 1:
        stage_html += '<span class="arrow">›</span>'

st.markdown(f'<div class="pipeline-bar">{stage_html}</div>', unsafe_allow_html=True)

# ── API KEY ────────────────────────────────────────────────────────────────────
if not cloud_mode:
    with st.expander("⚙️ API Key", expanded=not bool(st.session_state.groq_key)):
        user_key = st.text_input("Groq API Key (free at console.groq.com)", type="password", placeholder="gsk_...")
        if user_key:
            st.session_state.groq_key = user_key
            st.success("✓ Key saved")

# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    # ── DOCUMENT UPLOAD ────────────────────────────────────────────────────────
    st.markdown('<div class="panel-title">📄 KNOWLEDGE BASE</div>', unsafe_allow_html=True)

    input_tab1, input_tab2 = st.tabs(["📋 Paste Text", "📁 Upload File"])
    doc_input = ""
    with input_tab1:
        doc_input = st.text_area(
            "Paste your documents / knowledge here",
            height=180,
            placeholder="Paste any text — product docs, research notes, articles, FAQs...",
            label_visibility="collapsed"
        )
    with input_tab2:
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"], label_visibility="collapsed")
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                doc_input = uploaded_file.read().decode("utf-8")
                st.success(f"✅ Loaded: {uploaded_file.name} ({len(doc_input)} chars)")
            elif uploaded_file.type == "application/pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(uploaded_file)
                    doc_input = "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                    st.success(f"✅ Loaded: {uploaded_file.name} ({len(reader.pages)} pages)")
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")
    # Sample docs button
    if st.button("📋 Load Sample Documents"):
        sample = """
# LangChain Framework Overview
LangChain is an open-source framework for building LLM-powered applications. It provides abstractions for chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, Google, and Groq. The framework uses LCEL (LangChain Expression Language) for composing pipelines declaratively.

# RAG (Retrieval Augmented Generation)
RAG is a technique that combines retrieval of relevant documents with LLM generation. It solves the problem of LLMs hallucinating by grounding answers in real documents. RAG pipelines typically have three stages: indexing, retrieval, and generation. ChromaDB and FAISS are popular vector stores for RAG. Embeddings convert text into numerical vectors that capture semantic meaning.

# Self-Healing RAG
Traditional RAG fails silently — it returns answers even when retrieved chunks are irrelevant. Self-healing RAG adds an evaluation step: the LLM scores its own confidence (0-10). If the score is below a threshold (e.g. 6), it re-queries using expanded keywords, semantic rephrasing, or a broader search. This reduces hallucination rates by 40-60% in production systems.

# Vector Databases
Vector databases store embeddings and enable semantic similarity search. Unlike traditional SQL search which matches exact keywords, vector search finds semantically similar content. ChromaDB is a lightweight open-source vector database that runs in-memory or on disk. Pinecone is a managed cloud vector database. FAISS (Facebook AI Similarity Search) is optimized for large-scale similarity search.

# Groq API
Groq offers ultra-fast LLM inference using custom LPU (Language Processing Unit) hardware. The free tier supports 14,400 requests per day with models like llama-3.1-8b-instant and mixtral-8x7b. Response times are 10-20x faster than traditional GPU-based inference. Groq is ideal for real-time applications requiring low latency.

# Agentic AI
Agentic AI refers to AI systems that can take autonomous actions to achieve goals. Agents use a Reason-Act loop (ReAct): they think about what to do, execute an action, observe the result, and repeat. LangGraph is a framework for building stateful multi-agent systems. AutoGen by Microsoft enables multi-agent collaboration where agents can debate and verify each other's answers.
        """
        st.session_state["sample_loaded"] = sample
        st.rerun()

    if "sample_loaded" in st.session_state and not doc_input.strip():
        doc_input = st.session_state["sample_loaded"]

    if st.button("🔨 Index Documents", disabled=not bool(st.session_state.groq_key)):
        if doc_input.strip():
            with st.spinner("Chunking and indexing documents..."):
                rag = SelfHealingRAG(api_key=st.session_state.groq_key)
                count = rag.ingest(doc_input)
                st.session_state.rag = rag
                st.session_state.documents_loaded = True
                st.session_state.result = None
                st.success(f"✅ Indexed {count} chunks into vector store")
        else:
            st.warning("Paste some documents first!")

    # ── QUESTION INPUT ─────────────────────────────────────────────────────────
    if st.session_state.documents_loaded:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">❓ ASK A QUESTION</div>', unsafe_allow_html=True)

        # Sample questions
        st.markdown("**Quick questions to try:**")
        samples = [
            "What is self-healing RAG and how does it work?",
            "How does vector search differ from keyword search?",
            "What hardware does Groq use?",
            "Tell me about quantum computing"  # intentionally off-topic to trigger healing
        ]
        for sq in samples:
            if st.button(sq, key=f"sq_{sq[:20]}"):
                st.session_state.question = sq
                st.rerun()

        question = st.text_input(
            "Your question",
            value=st.session_state.question,
            placeholder="Ask anything about your documents...",
            label_visibility="collapsed"
        )

        conf_threshold = st.slider("Confidence threshold (below this → heal)", 1, 9, 6)

        if st.button("🚀 Ask & Self-Heal", disabled=not question.strip()):
            with st.spinner("Pipeline running..."):
                result = st.session_state.rag.query(question, threshold=conf_threshold)
                st.session_state.result = result
                st.session_state.question = question
                st.rerun()

with right_col:
    st.markdown('<div class="panel-title">🔬 PIPELINE TRACE</div>', unsafe_allow_html=True)

    if not st.session_state.documents_loaded:
        st.markdown("""
        <div style="background:#0d0d1a; border:1px dashed #1e1e30; border-radius:12px; padding:3rem; text-align:center; color:#334155;">
            <div style="font-size:2rem; margin-bottom:0.8rem;">🧠</div>
            <div style="font-size:0.9rem;">Index documents on the left to begin.<br>The pipeline trace will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.result:
        st.markdown("""
        <div style="background:#0d0d1a; border:1px dashed #1e1e30; border-radius:12px; padding:3rem; text-align:center; color:#334155;">
            <div style="font-size:2rem; margin-bottom:0.8rem;">✅</div>
            <div style="font-size:0.9rem;">Documents indexed! Ask a question to see<br>the self-healing pipeline in action.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state.result

        # Step 1: Question received
        st.markdown(f"""
        <div class="thought-step think">
            <div class="step-icon">🧠</div>
            <div>
                <div class="step-label">QUESTION RECEIVED</div>
                <div class="step-text">{r['question']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 2: Retrieved chunks
        st.markdown(f"""
        <div class="thought-step search">
            <div class="step-icon">🔍</div>
            <div>
                <div class="step-label">RETRIEVED {len(r['initial_chunks'])} CHUNKS (semantic search)</div>
                <div class="step-text">Top chunks from vector store:</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for i, chunk in enumerate(r["initial_chunks"][:3]):
            score = chunk.get("score", 0)
            score_cls = "score-high" if score > 0.7 else "score-mid" if score > 0.4 else "score-low"
            st.markdown(f"""
            <div class="chunk-card">
                <span class="chunk-score {score_cls}">sim: {score:.2f}</span>
                {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}
            </div>
            """, unsafe_allow_html=True)

        # Step 3: Confidence evaluation
        score = r["initial_confidence"]
        score_cls = "score-high" if score >= 7 else "score-mid" if score >= 4 else "score-low"
        threshold = r["threshold"]

        st.markdown(f"""
        <div class="thought-step eval">
            <div class="step-icon">🧮</div>
            <div>
                <div class="step-label">CONFIDENCE EVALUATION</div>
                <div class="step-text">
                    Initial confidence: <span class="score-badge {score_cls}">{score}/10</span>
                    &nbsp; Threshold: <span class="score-badge score-mid">{threshold}/10</span>
                    &nbsp; {"✅ Above threshold — no healing needed" if score >= threshold else "⚠️ Below threshold — triggering self-heal"}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 4: Healing (if triggered)
        if r.get("healed"):
            st.markdown(f"""
            <div class="thought-step heal">
                <div class="step-icon">🔄</div>
                <div>
                    <div class="step-label">SELF-HEALING TRIGGERED ({r['heal_iterations']} iteration{'s' if r['heal_iterations']>1 else ''})</div>
                    <div class="step-text">
                        Strategy used: <strong>{r['heal_strategy']}</strong><br>
                        Re-query: <em>"{r['requery']}"</em><br>
                        New confidence: <span class="score-badge {'score-high' if r['final_confidence'] >= threshold else 'score-mid'}">{r['final_confidence']}/10</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Healed chunks
            for chunk in r["healed_chunks"][:2]:
                score = chunk.get("score", 0)
                score_cls = "score-high" if score > 0.7 else "score-mid" if score > 0.4 else "score-low"
                st.markdown(f"""
                <div class="chunk-card" style="border-color:#f59e0b33;">
                    <span class="chunk-score {score_cls}">sim: {score:.2f}</span>
                    {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="thought-step answer" style="border-color:#64748b;">
                <div class="step-icon">⏭️</div>
                <div>
                    <div class="step-label">HEALING SKIPPED</div>
                    <div class="step-text">Confidence was above threshold — answer accepted as-is.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Step 5: Final answer
        healed = r.get("healed", False)
        answer_cls = "" if not healed else ""
        st.markdown(f"""
        <div class="thought-step answer">
            <div class="step-icon">✅</div>
            <div>
                <div class="step-label">FINAL ANSWER {"(after healing)" if healed else "(first attempt)"}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="answer-box">{r["final_answer"]}</div>', unsafe_allow_html=True)

        # Copy button
        st.code(r["final_answer"], language=None)

        # Stats row
        elapsed = r.get("elapsed", 0)
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-top:0.8rem; flex-wrap:wrap;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#475569;">⏱ {elapsed:.1f}s</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#475569;">🔁 {r.get('heal_iterations',0)} heal iterations</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:{'#34d399' if not healed else '#f59e0b'};">{'✅ No healing needed' if not healed else '🔄 Self-healed successfully'}</span>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1.5rem; color:#334155; font-size:0.8rem; font-family:'JetBrains Mono',monospace; border-top:1px solid #1e1e30;">
    Built with Python · ChromaDB · Groq · Streamlit &nbsp;|&nbsp; AI Build Series by Vaishnavi Gahoi
</div>
""", unsafe_allow_html=True)
