import os
import re
import time
import json
import math
import requests
from typing import List, Dict, Any

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"


# ── SIMPLE IN-MEMORY VECTOR STORE (no external DB needed) ────────────────────
class SimpleVectorStore:
    """
    Lightweight vector store using TF-IDF-style cosine similarity.
    No ChromaDB or sentence-transformers needed — works 100% free on Streamlit Cloud.
    """

    def __init__(self):
        self.chunks: List[Dict] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # simple stopword removal
        stops = {'the','a','an','is','it','in','on','at','to','for','of','and','or','but','with','this','that','are','was','were','be','been','has','have','had','by','from','as','into'}
        return [t for t in tokens if t not in stops and len(t) > 1]

    def _build_index(self):
        """Build TF-IDF vectors for all chunks."""
        N = len(self.chunks)
        if N == 0:
            return

        # Build vocabulary
        all_tokens = []
        doc_tokens = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk['text'])
            doc_tokens.append(tokens)
            all_tokens.extend(set(tokens))

        vocab = sorted(set(all_tokens))
        self.vocab = {w: i for i, w in enumerate(vocab)}

        # Compute IDF
        self.idf = {}
        for word in vocab:
            df = sum(1 for tokens in doc_tokens if word in tokens)
            self.idf[word] = math.log((N + 1) / (df + 1)) + 1

        # Compute TF-IDF vectors
        self.tfidf_matrix = []
        for tokens in doc_tokens:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            max_tf = max(tf.values()) if tf else 1
            vec = [0.0] * len(vocab)
            for word, count in tf.items():
                if word in self.vocab:
                    tfidf = (count / max_tf) * self.idf.get(word, 1)
                    vec[self.vocab[word]] = tfidf
            # Normalize
            norm = math.sqrt(sum(v**2 for v in vec)) or 1
            self.tfidf_matrix.append([v/norm for v in vec])

    def _query_vector(self, query: str) -> List[float]:
        tokens = self._tokenize(query)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        max_tf = max(tf.values()) if tf else 1
        vec = [0.0] * len(self.vocab)
        for word, count in tf.items():
            if word in self.vocab:
                tfidf = (count / max_tf) * self.idf.get(word, 1)
                vec[self.vocab[word]] = tfidf
        norm = math.sqrt(sum(v**2 for v in vec)) or 1
        return [v/norm for v in vec]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    def add_chunks(self, chunks: List[Dict]):
        self.chunks.extend(chunks)
        self._build_index()

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.chunks:
            return []
        qvec = self._query_vector(query)
        scores = [(self._cosine(qvec, dvec), i) for i, dvec in enumerate(self.tfidf_matrix)]
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            results.append({**self.chunks[idx], "score": round(score, 3)})
        return results

    def clear(self):
        self.__init__()


# ── GROQ LLM CALL ─────────────────────────────────────────────────────────────
def call_groq(prompt: str, api_key: str, max_tokens: int = 800) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ── SELF-HEALING RAG ENGINE ───────────────────────────────────────────────────
class SelfHealingRAG:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.store = SimpleVectorStore()

    def ingest(self, text: str, chunk_size: int = 300, overlap: int = 50) -> int:
        """Split text into overlapping chunks and index them."""
        self.store.clear()
        words = text.split()
        chunks = []
        i = 0
        chunk_id = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if len(chunk_text) > 50:
                chunks.append({"id": chunk_id, "text": chunk_text})
                chunk_id += 1
            i += chunk_size - overlap
        self.store.add_chunks(chunks)
        return len(chunks)

    def _evaluate_confidence(self, question: str, chunks: List[Dict], draft_answer: str) -> int:
        """Ask the LLM to rate its own confidence in the answer (0-10)."""
        context = "\n\n".join(c["text"] for c in chunks[:3])
        prompt = f"""You are evaluating your own answer quality.

Question: {question}

Retrieved context:
{context}

Draft answer: {draft_answer}

On a scale of 0-10, how confident are you that this answer is accurate, complete, and well-supported by the context?
- 0-3: Context is irrelevant or answer is clearly wrong/hallucinated
- 4-6: Context is partially relevant but answer is incomplete or uncertain
- 7-10: Context is highly relevant and answer is accurate and complete

Return ONLY a single integer (0-10). Nothing else."""

        try:
            result = call_groq(prompt, self.api_key, max_tokens=10)
            score = int(re.search(r'\d+', result).group())
            return min(max(score, 0), 10)
        except Exception:
            return 5

    def _generate_answer(self, question: str, chunks: List[Dict]) -> str:
        """Generate an answer from retrieved chunks."""
        context = "\n\n".join(c["text"] for c in chunks[:4])
        prompt = f"""Answer the following question using only the provided context. 
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer (be concise and specific, 2-4 sentences):"""
        return call_groq(prompt, self.api_key, max_tokens=400)

    def _generate_requery(self, question: str, failed_chunks: List[Dict], strategy: str) -> str:
        """Generate an improved search query using a healing strategy."""
        context_preview = " ".join(c["text"][:100] for c in failed_chunks[:2])

        if strategy == "expand":
            prompt = f"""The search query "{question}" returned weak results.
Generate a broader, more general version of this query to find relevant information.
Return ONLY the new query string, nothing else."""

        elif strategy == "rephrase":
            prompt = f"""The search query "{question}" returned weak results.
Rephrase this question using different keywords and synonyms that might match better.
Return ONLY the rephrased query string, nothing else."""

        elif strategy == "decompose":
            prompt = f"""The search query "{question}" returned weak results.
Break this into the most important single concept to search for.
Return ONLY the key concept as a short phrase, nothing else."""

        else:
            prompt = f"""The search query "{question}" failed. Generate an alternative search query.
Return ONLY the new query string."""

        try:
            return call_groq(prompt, self.api_key, max_tokens=80).strip('"\'')
        except Exception:
            return question

    def query(self, question: str, threshold: int = 6, max_iterations: int = 2) -> Dict[str, Any]:
        """
        Full self-healing RAG query.
        Returns a dict with the full trace of the pipeline.
        """
        start = time.time()

        # Step 1: Initial retrieval
        initial_chunks = self.store.search(question, top_k=4)

        # Step 2: Generate initial answer
        initial_answer = self._generate_answer(question, initial_chunks)

        # Step 3: Evaluate confidence
        initial_confidence = self._evaluate_confidence(question, initial_chunks, initial_answer)

        result = {
            "question": question,
            "initial_chunks": initial_chunks,
            "initial_answer": initial_answer,
            "initial_confidence": initial_confidence,
            "threshold": threshold,
            "healed": False,
            "heal_iterations": 0,
            "heal_strategy": None,
            "requery": None,
            "healed_chunks": [],
            "final_answer": initial_answer,
            "final_confidence": initial_confidence,
            "elapsed": 0
        }

        # Step 4: Self-heal if below threshold
        if initial_confidence < threshold:
            strategies = ["rephrase", "expand", "decompose"]
            best_answer = initial_answer
            best_confidence = initial_confidence
            best_chunks = initial_chunks

            for i, strategy in enumerate(strategies[:max_iterations]):
                requery = self._generate_requery(question, initial_chunks, strategy)
                healed_chunks = self.store.search(requery, top_k=4)
                healed_answer = self._generate_answer(question, healed_chunks)
                healed_confidence = self._evaluate_confidence(question, healed_chunks, healed_answer)

                result["heal_iterations"] = i + 1
                result["heal_strategy"] = strategy
                result["requery"] = requery
                result["healed_chunks"] = healed_chunks

                if healed_confidence > best_confidence:
                    best_confidence = healed_confidence
                    best_answer = healed_answer
                    best_chunks = healed_chunks

                if healed_confidence >= threshold:
                    break

            result["healed"] = True
            result["final_answer"] = best_answer
            result["final_confidence"] = best_confidence

        result["elapsed"] = round(time.time() - start, 1)
        return result
