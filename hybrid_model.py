from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt template for summary generation
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
    Given the following job description and candidate resume, generate a 3-sentence summary explaining why this candidate is a good fit.

    Job Description:
    {job}

    Resume:
    {resume}
    """
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Function to generate AI summary

def generate_fit_summary(job_desc: str, resume_text: str) -> tuple:
    prompt_text = SUMMARY_PROMPT.format_messages(job=job_desc, resume=resume_text[:1500])

    try:
        response = llm.invoke(prompt_text)
        return prompt_text[0].content, response.content
    except Exception as e:
        return prompt_text[0].content, f"[ERROR] Summary generation failed: {e}"

# Function to rank candidates

def query_top_k_candidates(job_desc, resumes, top_k=5, include_summary=False, min_score=0.5):
    results = []

    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    job_embedding = embedder.encode([job_desc])[0]

    # Preprocess resumes and compute similarity
    for resume in resumes:
        embedding = embedder.encode([resume['text']])[0]
        similarity = cosine_similarity([job_embedding], [embedding])[0][0]

        if similarity >= min_score:
            summary = ""
            prompt = ""
            if include_summary:
                prompt, summary = generate_fit_summary(job_desc, resume['text'])

            results.append({
                "file_name": resume.get('file_name', 'Unnamed Resume'),
                "similarity_score": round(similarity, 2),
                "summary": summary,
                "prompt": prompt,
                "embedding": embedding.tolist()
            })

    # Return top_k based on similarity
    return sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
