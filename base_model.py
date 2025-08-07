import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from openai import OpenAI
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client with explicit API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a given text using SentenceTransformer."""
    try:
        return embedder.encode([text])[0]
    except Exception as e:
        print(f"[ERROR] Failed to embed text: {e}")
        return np.zeros(embedder.get_sentence_embedding_dimension())

def compute_similarity(job_emb: np.ndarray, resume_embs: List[np.ndarray]) -> np.ndarray:
    """Compute cosine similarity between job embedding and each resume embedding."""
    job_vector = np.array(job_emb).reshape(1, -1)
    resume_matrix = np.vstack(resume_embs)
    return cosine_similarity(job_vector, resume_matrix)[0]

def generate_fit_summary(job_desc: str, resume_text: str) -> Tuple[str, str]:
    """Generate a 2-3 sentence AI summary explaining candidate fit."""
    prompt = f"""You are an AI recruiting assistant.

Here is a job description:

{job_desc}

Here is a candidate resume:

{resume_text[:1500]}

Write 2-3 sentences summarizing why this candidate may be a good fit for the job.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        return prompt, summary
    except Exception as e:
        return prompt, f"[ERROR] Summary generation failed: {str(e)}"

def rank_candidates(
    job_desc: str,
    resumes: List[Dict],
    top_k: int = 5,
    include_summary: bool = False,
    min_score: float = 0.5
) -> List[Dict]:
    """
    Rank candidates by cosine similarity to the job description.
    
    Args:
        job_desc: Job description text
        resumes: List of dicts with 'text' and 'file_name' keys
        top_k: Number of top matches to return
        include_summary: Whether to generate fit summaries
        min_score: Minimum similarity threshold to include

    Returns:
        A ranked list of candidates above the similarity threshold.
    """
    job_emb = embed_text(job_desc)
    resume_texts = [r["text"] for r in resumes]
    resume_embs = [embed_text(text) for text in resume_texts]
    similarities = compute_similarity(job_emb, resume_embs)

    results = []
    for i, sim in enumerate(similarities):
        file_name = resumes[i].get("file_name", f"Candidate {i+1}")
        print(f"[DEBUG] Similarity for {file_name}: {sim:.4f} (Min threshold: {min_score})")

        if sim >= min_score:
            prompt, summary = ("", "")
            if include_summary:
                prompt, summary = generate_fit_summary(job_desc, resume_texts[i])
            
            results.append({
                "file_name": file_name,
                "similarity_score": round(float(sim), 2),
                "summary": summary,
                "prompt": prompt,
                "embedding": resume_embs[i].tolist()
            })

    # Sort results by similarity score (descending) and return top_k
    return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
