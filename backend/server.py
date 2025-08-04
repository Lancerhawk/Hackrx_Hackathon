from fastapi import FastAPI, APIRouter, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from groq import Groq
import logging
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any
import uuid
from datetime import datetime
import asyncio
import aiohttp
import tempfile
import io
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import openai
import tiktoken
import nltk
import re


def normalize_question(question: str) -> str:
    question = re.sub(r"[\"'`]", "", question)
    return question.strip().lower()  


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

groq_client = Groq(api_key=os.environ.get("API_KEY") ,)

embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    client.close()

app = FastAPI(lifespan=lifespan)

api_router = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class HackRXRequest(BaseModel):
    documents: str  
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}

class SemanticSearchEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the search engine"""
        self.chunks.extend(chunks)
        
        texts = [chunk.text for chunk in chunks]
        new_embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        logger.info(f"Added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Search for most relevant chunks using cosine similarity"""
        if len(self.embeddings) == 0:
            return []
        
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.similarity_score = similarities[idx]  
            relevant_chunks.append(chunk)
        
        relevant_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return relevant_chunks
    
    def search_progressive(self, query: str, start_k: int = 5, max_k: int = None, used_chunks: set = None) -> List[DocumentChunk]:
        """Search progressively with increasing chunk count, avoiding previously used chunks"""
        if len(self.embeddings) == 0:
            return []
        
        if max_k is None:
            max_k = len(self.chunks)
        
        if used_chunks is None:
            used_chunks = set()
        
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        all_indices = np.argsort(similarities)[::-1]
        
        available_indices = [idx for idx in all_indices if idx not in used_chunks]
        
        if not available_indices:
            return [], 0, max_k
        
        current_k = min(start_k, len(available_indices))
        new_indices = available_indices[:current_k]
        
        relevant_chunks = []
        for idx in new_indices:
            chunk = self.chunks[idx]
            chunk.similarity_score = similarities[idx]  
            relevant_chunks.append(chunk)
            used_chunks.add(idx) 
        
        relevant_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return relevant_chunks, current_k, max_k
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """Hybrid search combining semantic similarity with keyword matching"""
        if len(self.embeddings) == 0:
            return []
        
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.text.lower().split())
            keyword_matches = len(query_words.intersection(chunk_words))
            keyword_scores.append(keyword_matches)
        
        max_keyword = max(keyword_scores) if keyword_scores else 1
        keyword_scores = [score / max_keyword for score in keyword_scores]
        
        combined_scores = [0.7 * sim + 0.3 * kw for sim, kw in zip(similarities, keyword_scores)]
        
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.similarity_score = combined_scores[idx]  
            relevant_chunks.append(chunk)
        
        relevant_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return relevant_chunks

search_engine = SemanticSearchEngine()

async def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download PDF: {response.status}")
            return await response.read()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks with better strategy"""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            words = current_chunk.split()
            if len(words) > overlap:
                overlap_text = " ".join(words[-overlap:])
                current_chunk = overlap_text + " " + sentence
                current_length = overlap + sentence_length
            else:
                current_chunk = sentence
                current_length = sentence_length
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_length += sentence_length
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
    
    return chunks


async def generate_answer(question: str, relevant_chunks: List[DocumentChunk]) -> str:
    """Generate answer using llama-3.1-8b-instant"""
    try:
        context = "\n\n".join([chunk.text for chunk in relevant_chunks])
        
        encoding = tiktoken.get_encoding("cl100k_base") 
        context_tokens = encoding.encode(context)
        
        max_context_tokens = 4900
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = encoding.decode(context_tokens)
        
        prompt = f"""Based on the following document content, provide a direct and complete answer to the question in 1-4 sentence depending on the question and prioritize completeness over brevity.

Document Content:
{context}

Question: {question}

Instructions:
- Answer in 1-3 sentences only and no extra legal texts, but make it short and complete, don't remove the necessary information even in slightest
- Be direct and to the point
- No bullet points, no lists, no explanations
- If information is not available, simply say "This information is not available in the document"
- Use specific details from the document when available"""

        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on document content. Always provide accurate, direct answers based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        
        no_info_indicators = [
            "not available", "not found", "not mentioned", "not specified",
            "not provided", "not stated", "not included", "not covered",
            "no information", "no details", "no mention", "no data",
            "cannot find", "unable to find", "does not contain"
        ]
        
        answer_lower = answer.lower()
        if any(indicator in answer_lower for indicator in no_info_indicators):
            return answer, False 
        
        return answer, True 
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        if "6000" in str(e) or "tpm" in str(e).lower() or "rate limit" in str(e).lower():
            return "TPM_LIMIT_HIT", False 
        return f"Error generating answer: {str(e)}", False

async def summarize_answer(raw_answer: str, question: str) -> str:
    """Summarize verbose or legal-style answers into a clean, single-sentence form"""
    try:
        prompt = f"""Please rewrite the following answer to the question below into a single concise, human-friendly sentence that captures the key details but avoids legal jargon but prioritize completeness over brevity:

Question: {question}
Answer: {raw_answer}

Instruction:
- Make it short and to the point but make it short and complete, don't remove the necessary information even in slightest
- Remove legal/boilerplate language
- Keep essential facts like numbers, names, and conditions
- Do not say 'According to the document...' or similar prefixes"""

        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a summarizer that rewrites verbose legal answers into clean, human-friendly summaries, Formal and precise,  but don't remove the information even in slightest."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"Summarization failed: {str(e)}")
        return raw_answer 

async def generate_answer_with_retry(question: str, search_engine, start_k: int = 5) -> str:
    """Generate answer with hybrid search and smart retry logic"""
    max_attempts = 8
    used_chunks = set()
    
    logger.info("Starting with hybrid search for better initial results")
    hybrid_chunks = search_engine.hybrid_search(question, top_k=start_k)
    
    if hybrid_chunks:
        for chunk in hybrid_chunks:
            chunk_idx = search_engine.chunks.index(chunk)
            used_chunks.add(chunk_idx)
        
        logger.info(f"Hybrid search found {len(hybrid_chunks)} relevant chunks")
        answer, success = await generate_answer(question, hybrid_chunks)
        
        if answer == "TPM_LIMIT_HIT":
            logger.info("TPM limit hit on hybrid search, waiting 60 seconds...")
            await asyncio.sleep(60)
        elif success:
            logger.info("Successfully generated answer with hybrid search")
            summarized_answer = await summarize_answer(answer, question)
            return summarized_answer
    
    for attempt in range(max_attempts):
        logger.info(f"Progressive attempt {attempt + 1}: Searching for new chunks")
        
        new_chunks, current_k, max_k = search_engine.search_progressive(
            question, start_k=3, max_k=None, used_chunks=used_chunks
        )
        
        if not new_chunks:
            logger.info("No more new chunks available to search")
            break
        
        logger.info(f"Found {len(new_chunks)} new relevant chunks for attempt {attempt + 1}")
        
        answer, success = await generate_answer(question, new_chunks)
        
        if answer == "TPM_LIMIT_HIT":
            logger.info(f"TPM limit hit on attempt {attempt + 1}, waiting 60 seconds...")
            await asyncio.sleep(60)
            continue
        
        if success:
            logger.info(f"Successfully generated answer on attempt {attempt + 1}")
            # Summarize the answer to remove legal jargon
            summarized_answer = await summarize_answer(answer, question)
            return summarized_answer
        
        logger.info(f"No useful answer found on attempt {attempt + 1}")
        
        await asyncio.sleep(3)
    
    logger.info("Trying final comprehensive search with all remaining chunks")
    remaining_chunks = []
    for i, chunk in enumerate(search_engine.chunks):
        if i not in used_chunks:
            remaining_chunks.append(chunk)
    
    if remaining_chunks:
        logger.info(f"Final search with {len(remaining_chunks)} remaining chunks")
        answer, success = await generate_answer(question, remaining_chunks)
        
        if answer == "TPM_LIMIT_HIT":
            logger.info("TPM limit hit on final search, waiting 60 seconds...")
            await asyncio.sleep(60)
            answer, success = await generate_answer(question, remaining_chunks)
        
        if success:
            logger.info("Successfully generated answer on final search")
            summarized_answer = await summarize_answer(answer, question)
            return summarized_answer
    
    logger.info("No answer found after comprehensive search")
    return "This information is not available in the document"

@api_router.get("/")
async def root():
    return {"message": "Semantic Search Platform API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """Main endpoint for semantic search and question answering"""
    try:
        logger.info(f"Processing request with document: {request.documents}")
        logger.info(f"Questions: {len(request.questions)}")
        
        pdf_bytes = await download_pdf(request.documents)
        logger.info("PDF downloaded successfully")
        
        text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"Extracted text length: {len(text)} characters")
        
        chunks_text = chunk_text(text)
        logger.info(f"Created {len(chunks_text)} chunks")
        
        document_chunks = [DocumentChunk(text=chunk_text) for chunk_text in chunks_text]
        
        global search_engine
        search_engine = SemanticSearchEngine()  
        search_engine.add_chunks(document_chunks)
        
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")

            normalized_question = normalize_question(question)
            answer = await generate_answer_with_retry(question, search_engine, start_k=5)
            answers.append(answer)
            logger.info(f"Generated answer: {answer[:100]}...")
            
            if i < len(request.questions) - 1:
                logger.info("Waiting 5 seconds before processing next question...")
                await asyncio.sleep(5)
        
        logger.info("All questions processed successfully")
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in hackrx_run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"📚 API Documentation available at: http://{host}:{port}/docs")
    print(f"🔍 Health check available at: http://{host}:{port}/api/")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,  
        log_level="info"
    )