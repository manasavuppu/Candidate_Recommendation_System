# Candidate Recommendation System

This repository contains an interactive Streamlit application that intelligently matches candidate resumes to job descriptions using semantic similarity techniques and optional AI-generated summaries. It also includes visualizations to help users understand the matching process.

---

## Project Overview

In today’s fast-paced hiring landscape, recruiters need intelligent tools to quickly identify top candidates. This system analyzes a given job description and compares it to multiple resumes using semantic similarity (not just keyword matching), returning a ranked list of candidates based on relevance.

---

## Key Features

- Upload and parse job descriptions (`.pdf`, `.docx`, or `.txt`)
- Upload multiple resumes
- Choose between two models:
  - **Base Model**: SentenceTransformers-based similarity with optional OpenAI-generated summary
  - **Hybrid Model**: Combines LlamaIndex with LangChain for enhanced retrieval and reasoning
- Adjustable parameters:
  - `Top K` candidates to return
  - Minimum similarity score threshold
- Optional AI-generated fit summaries using OpenAI GPT
- Visualize embedding vectors and similarity scores
- Modular code structure for scalability

---

## Model Design & Rationale

### 1. Base Model (SentenceTransformers + Optional OpenAI Summary)

- **Embedding Strategy**: Uses `all-MiniLM-L6-v2` from the SentenceTransformers library
- **Similarity Metric**: Cosine similarity between job and resume embeddings
- **AI Summary (Optional)**: GPT-generated explanation of candidate fit
- **Ideal For**: Fast local analysis or lightweight use cases

### 2. Hybrid Model (LlamaIndex + LangChain)

- **Embedding & Indexing**: Uses LlamaIndex to parse and embed resume chunks
- **Retrieval**: LangChain-powered retriever for reasoning-based queries
- **LLM Reasoning**: Suitable for more contextual, explainable matches
- **Ideal For**: Enterprise applications and in-depth resume screening

---

## Model Comparison

| Feature                  | Base Model                        | Hybrid Model                         |
|--------------------------|------------------------------------|--------------------------------------|
| Execution Speed          | Fast                               | Slower                               |
| Infrastructure Cost      | Minimal (OpenAI optional)          | Higher (LLMs + LangChain)            |
| Embedding Interpretability | Supports vector visualization    | Supports advanced prompt analysis    |
| Complexity Handling      | Basic                              | Advanced reasoning capabilities       |
| Deployment Simplicity    | Local/Streamlit compatible         | Requires additional setup            |

---

## Technology Stack

| Layer             | Tools and Libraries                      |
|-------------------|------------------------------------------|
| Application       | Streamlit                               |
| Embeddings        | SentenceTransformers, LlamaIndex         |
| LLMs              | OpenAI GPT (via API), LangChain          |
| Document Parsing  | PyMuPDF, python-docx                     |
| Visualization     | Matplotlib, NumPy                        |
| File Types        | PDF, DOCX, TXT                           |

---

## Running the App Locally

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/manasavuppu/Candidate_Recommendation_System.git
cd Candidate_Recommendation_System

# Create and activate virtual environment
python3 -m venv cand_rec_env
source cand_rec_env/bin/activate  # On Windows: cand_rec_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Environment Setup
Create a .env file in the project root directory with the following content:
OPENAI_API_KEY=your-openai-key-here
Ensure that .env is listed in your .gitignore file to avoid accidentally pushing secrets to GitHub.

Run the Application
streamlit run app.py
