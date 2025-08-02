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
import openai
import tiktoken
import nltk
import re

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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        
        max_context_tokens = 131072 
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = encoding.decode(context_tokens)
        
        prompt = f"""Based on the following document content, provide a direct and concise answer to the question in one sentence.

Document Content:
{context}

Question: {question}

Instructions:
- Answer in ONE sentence only
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
            max_tokens=300,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

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
            
            relevant_chunks = search_engine.search(question, top_k=5)
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            answer = await generate_answer(question, relevant_chunks)
            answers.append(answer)
            logger.info(f"Generated answer: {answer[:100]}...")
            
            if i < len(request.questions) - 1:
                logger.info("Waiting 2 seconds before processing next question...")
                await asyncio.sleep(2)
        
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