from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from dotenv import load_dotenv
import os

load_dotenv()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Prompt template
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Given the following job description and candidate resume, generate a 3-sentence summary explaining why this candidate is a good fit.\n\nJob Description:\n{job}\n\nResume:\n{resume}"
)

# LLM for summaries
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Utility to generate a fit summary and return prompt for transparency
def generate_fit_summary(job_desc: str, resume_text: str) -> tuple:
    try:
        # Generate list of messages (LangChain Message objects)
        messages = SUMMARY_PROMPT.format_messages(job=job_desc, resume=resume_text)
        
        # Pass messages to LLM
        response = llm(messages)

        print("=== PROMPT ===")
        print(messages[0].content)
        print("=== RESPONSE ===")
        print(response.content)
            
        # Return both the input and response for transparency
        return messages[0].content, response.content
    except Exception as e:
        return "Prompt error", f"Error generating summary: {str(e)}"

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
