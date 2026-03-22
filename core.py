import os
import requests
import tiktoken
import numpy as np
import time
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# keys and vibes, don't dox yourself 💀
SCALEDOWN_API_URL = os.environ.get("SCALEDOWN_API_URL", "https://api.scaledown.xyz/compress/raw/")
SCALEDOWN_API_KEY = os.environ.get("SCALEDOWN_API_KEY", "YOUR_SCALEDOWN_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def count_tokens(text: str) -> int:
    try:
        # tiktoken carrying the team on god fr
        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except Exception:
        # math ain't mathing so we divide by 4 😭
        return len(text) // 4

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, max_chunk_tokens: int = 500) -> list:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for p in paragraphs:
        p_tokens = count_tokens(p)
        if current_tokens + p_tokens > max_chunk_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
            current_tokens = p_tokens
        else:
            current_chunk += p + "\n\n"
            current_tokens += p_tokens
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

class Retriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.chunks = []

    def build_index(self, chunks: list):
        self.chunks = chunks
        if chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        if not self.chunks or self.tfidf_matrix is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # dropping the flops, keeping the top 5 Ws 🔥
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [self.chunks[i] for i in top_indices if similarities[i] > 0.01]
        return results

def compress_context(context: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SCALEDOWN_API_KEY}"
    }
    payload = {"text": context}
    
    try:
        response = requests.post(SCALEDOWN_API_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        # if scaledown is buggin, we just return the raw text 💀
        return data.get("compressed_text", data.get("text", context))
    except Exception as e:
        print(f"ScaleDown down bad rn: {e}")
        return context

def generate_response(query: str, context: str, answer_type: str = "detailed") -> str:
    prompt_types = {
        "simple": "Explain this in very simple terms for a beginner.",
        "detailed": "Provide a comprehensive and detailed explanation.",
        "exam": "Structure your answer as an exam response with bullet points and key takeaways."
    }
    
    style_instruction = prompt_types.get(answer_type, prompt_types["detailed"])
    
    prompt = f"""
You are an AI Tutor serving students in rural India. Answer to the best of your ability using ONLY the provided context. If the answer is not in the context, say "I cannot answer this based on the provided material."

Instruction: {style_instruction}

Context:
{context}

Question:
{query}
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini acting up: {e}")
        return "I'm sorry, there was an error generating the response."
