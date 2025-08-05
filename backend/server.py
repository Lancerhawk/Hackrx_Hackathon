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
from typing import List, Dict, Any, Optional, Tuple
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
import tiktoken
import nltk
import re
import torch
import gc
from collections import OrderedDict
import time
import math
from typing import Generator, Iterator
import psutil
import hashlib
import json
import pickle

class RequestQueue:
    """Manages request queue for sequential processing"""
    
    def __init__(self):
        self.processing_lock = asyncio.Lock()
        self.current_request_id = None
        self.logger = logging.getLogger(__name__)
    
    async def process_request(self, request_id: str, request_func, *args, **kwargs):
        """Process a request with queue management"""
        async with self.processing_lock:
            self.current_request_id = request_id
            self.logger.info(f"🚀 Processing request {request_id}")
            
            try:
                result = await request_func(*args, **kwargs)
                self.logger.info(f"✅ Completed request {request_id}")
                return result
            except Exception as e:
                self.logger.error(f"❌ Error in request {request_id}: {e}")
                raise
            finally:
                self.current_request_id = None
    
    def get_queue_status(self):
        """Get current queue status"""
        return {
            'processing': self.processing_lock.locked(),
            'current_request': self.current_request_id
        }

request_queue = RequestQueue()

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pynvml not available. GPU memory monitoring disabled.")

def normalize_question(question: str) -> str:
    question = re.sub(r"[\"'`]", "", question)
    return question.strip().lower()

class HybridMemoryManager:
    """Manages GPU/CPU hybrid memory with dynamic offloading"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE and torch.cuda.is_available()
        self.gpu_memory_threshold = 0.85  
        self.gpu_offload_threshold = 0.75  
        self.cpu_cache = OrderedDict()
        self.gpu_cache = OrderedDict()  
        self.max_cpu_cache_size = 10000  
        self.max_gpu_cache_size = 1000   
        self.embedding_models = {}  
        self.logger = logging.getLogger(__name__)
        
        self.max_chunk_size = 500  
        self.max_batch_size = 32  
        self.memory_safety_margin = 0.1 
        
        self.small_pdf_threshold = 10 * 1024 * 1024 
        self.medium_pdf_threshold = 50 * 1024 * 1024 
        
        self.model_configs = {
            'heavy': {
                'name': 'nomic-ai/nomic-embed-text-v1',
                'dimension': 768,
                'speed': 'slow',
                'accuracy': 'high'
            },
            'medium': {
                'name': 'all-mpnet-base-v2',
                'dimension': 768,
                'speed': 'medium',
                'accuracy': 'high'
            },
            'light': {
                'name': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'speed': 'fast',
                'accuracy': 'good'
            }
        }
        
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as a percentage"""
        if not self.gpu_available:
            return 0.0
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / info.total
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory usage: {e}")
            return 0.0
    
    def select_model_for_pdf_size(self, file_size: int) -> str:
        """Select the appropriate model based on PDF size"""
        if file_size <= self.small_pdf_threshold:
            model_type = 'heavy'
            self.logger.info(f"Small PDF ({file_size / 1024 / 1024:.1f}MB) - using heavy model for high accuracy")
        elif file_size <= self.medium_pdf_threshold:
            model_type = 'medium'
            self.logger.info(f"Medium PDF ({file_size / 1024 / 1024:.1f}MB) - using medium model for balance")
        else:
            model_type = 'light'
            self.logger.info(f"Large PDF ({file_size / 1024 / 1024:.1f}MB) - using light model for speed")
        
        return model_type
    
    def get_embedding_model(self, model_type: str = 'heavy') -> SentenceTransformer:
        """Get or initialize the embedding model for the specified type"""
        if model_type not in self.embedding_models:
            model_config = self.model_configs[model_type]
            self.logger.info(f"Initializing {model_type} model: {model_config['name']}")
            
            try:
                if model_type == 'heavy':
                    model = SentenceTransformer(
                        model_config['name'], 
                        trust_remote_code=True
                    )
                else:
                    model = SentenceTransformer(model_config['name'])
                
                self.embedding_models[model_type] = model
                self.logger.info(f"Successfully loaded {model_type} model")
                
            except Exception as e:
                self.logger.error(f"Failed to load {model_type} model: {e}")
                if model_type != 'light':
                    self.logger.info("Falling back to light model")
                    return self.get_embedding_model('light')
                else:
                    raise e
        
        return self.embedding_models[model_type]
    
    def clear_model_cache(self):
        """Clear all model caches to free memory"""
        self.logger.info("Clearing model cache")
        for model_type, model in self.embedding_models.items():
            try:
                del model
                self.logger.info(f"Cleared {model_type} model")
            except Exception as e:
                self.logger.warning(f"Failed to clear {model_type} model: {e}")
        
        self.embedding_models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def should_offload_to_cpu(self) -> bool:
        """Check if we should offload embeddings to CPU"""
        if not self.gpu_available:
            return False
        
        gpu_usage = self.get_gpu_memory_usage()
        return gpu_usage > self.gpu_offload_threshold
    
    def should_use_cpu_only(self) -> bool:
        """Check if we should use CPU only (emergency fallback)"""
        if not self.gpu_available:
            return True
        
        gpu_usage = self.get_gpu_memory_usage()
        return gpu_usage > self.gpu_memory_threshold
    
    def offload_oldest_gpu_embeddings(self, count: int = 10):
        """Offload oldest GPU embeddings to CPU"""
        if not self.gpu_available or len(self.gpu_cache) == 0:
            return
        
        self.logger.info(f"Offloading {count} oldest GPU embeddings to CPU")
        
        oldest_keys = list(self.gpu_cache.keys())[:count]
        
        for key in oldest_keys:
            if key in self.gpu_cache:
                embedding = self.gpu_cache.pop(key)
                self.cpu_cache[key] = embedding
                
                if len(self.cpu_cache) > self.max_cpu_cache_size:
                    self.cpu_cache.popitem(last=False)  
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Offloaded {len(oldest_keys)} embeddings. GPU cache: {len(self.gpu_cache)}, CPU cache: {len(self.cpu_cache)}")
    
    def encode_texts_hybrid(self, texts: List[str], chunk_ids: List[str] = None, model_type: str = 'heavy') -> np.ndarray:
        """Encode texts using hybrid GPU/CPU approach with dynamic model selection"""
        if chunk_ids is None:
            chunk_ids = [str(i) for i in range(len(texts))]
        
        model = self.get_embedding_model(model_type)
        results = []
        
        if self.should_use_cpu_only():
            self.logger.warning("GPU memory critical, using CPU only for encoding")
            try:
                embeddings = model.encode(texts, convert_to_tensor=False, device='cpu')
                for i, chunk_id in enumerate(chunk_ids):
                    self.cpu_cache[chunk_id] = embeddings[i]
                return embeddings
            except Exception as e:
                self.logger.error(f"CPU encoding failed: {e}")
                raise e
        
        try:
            if self.should_offload_to_cpu():
                self.offload_oldest_gpu_embeddings(count=20)
            
            embeddings = model.encode(texts, convert_to_tensor=False, device='cuda')
            
            for i, chunk_id in enumerate(chunk_ids):
                self.gpu_cache[chunk_id] = embeddings[i]
                
                if len(self.gpu_cache) > self.max_gpu_cache_size:
                    self.gpu_cache.popitem(last=False) 
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embeddings
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                self.logger.warning(f"GPU OOM during encoding: {e}. Falling back to CPU")
                
                self.gpu_cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    embeddings = model.encode(texts, convert_to_tensor=False, device='cpu')
                    for i, chunk_id in enumerate(chunk_ids):
                        self.cpu_cache[chunk_id] = embeddings[i]
                    return embeddings
                except Exception as cpu_e:
                    self.logger.error(f"CPU fallback also failed: {cpu_e}")
                    raise cpu_e
            else:
                raise e
    
    def get_embeddings_for_search(self, chunk_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings for search, pulling from GPU/CPU as needed"""
        embeddings = []
        available_ids = []
        
        for chunk_id in chunk_ids:
            if chunk_id in self.gpu_cache:
                embeddings.append(self.gpu_cache[chunk_id])
                available_ids.append(chunk_id)
                self.gpu_cache.move_to_end(chunk_id)
            elif chunk_id in self.cpu_cache:
                embeddings.append(self.cpu_cache[chunk_id])
                available_ids.append(chunk_id)
                self.cpu_cache.move_to_end(chunk_id)
        
        if embeddings:
            return np.array(embeddings), available_ids
        else:
            return np.array([]), []
    
    def move_to_gpu_if_needed(self, chunk_ids: List[str]) -> List[str]:
        """Move embeddings from CPU to GPU if needed for processing"""
        if not self.gpu_available:
            return chunk_ids
        
        moved_ids = []
        
        for chunk_id in chunk_ids:
            if chunk_id in self.cpu_cache and chunk_id not in self.gpu_cache:
                if len(self.gpu_cache) < self.max_gpu_cache_size and not self.should_offload_to_cpu():
                    embedding = self.cpu_cache.pop(chunk_id)
                    self.gpu_cache[chunk_id] = embedding
                    moved_ids.append(chunk_id)
        
        if moved_ids:
            self.logger.info(f"Moved {len(moved_ids)} embeddings from CPU to GPU")
        
        return moved_ids
    
    def cleanup_memory(self):
        """Clean up memory and optimize cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        while len(self.cpu_cache) > self.max_cpu_cache_size:
            self.cpu_cache.popitem(last=False)
        
        while len(self.gpu_cache) > self.max_gpu_cache_size:
            self.gpu_cache.popitem(last=False)
        
        gc.collect()
        
        self.logger.info(f"Memory cleanup completed. GPU cache: {len(self.gpu_cache)}, CPU cache: {len(self.cpu_cache)}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        return {
            "gpu_available": self.gpu_available,
            "gpu_memory_usage": self.get_gpu_memory_usage(),
            "gpu_cache_size": len(self.gpu_cache),
            "cpu_cache_size": len(self.cpu_cache),
            "max_gpu_cache_size": self.max_gpu_cache_size,
            "max_cpu_cache_size": self.max_cpu_cache_size
        }
    
    def get_system_memory_usage(self) -> float:
        """Get system memory usage as a percentage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception as e:
            self.logger.warning(f"Failed to get system memory usage: {e}")
            return 0.0
    
    def should_use_piece_processing(self, file_size: int, text_length: int) -> bool:
        """Determine if piece-by-piece processing should be used"""
        large_file = file_size > 100 * 1024 * 1024  
        large_text = text_length > 500000  
        high_memory = self.get_system_memory_usage() > 0.8  
        high_gpu = self.get_gpu_memory_usage() > 0.7  
        
        return large_file or large_text or high_memory or high_gpu
    
    def encode_texts_piece_by_piece(self, texts: List[str], chunk_ids: List[str] = None, model_type: str = 'light') -> np.ndarray:
        """Encode texts in small pieces to prevent memory issues with dynamic model selection"""
        if chunk_ids is None:
            chunk_ids = [str(i) for i in range(len(texts))]
        
        model = self.get_embedding_model(model_type)
        all_embeddings = []
        
        piece_size = min(self.max_batch_size, max(1, len(texts) // 10)) 
        
        self.logger.info(f"Processing {len(texts)} texts in pieces of {piece_size} using {model_type} model")
        
        for i in range(0, len(texts), piece_size):
            piece_texts = texts[i:i + piece_size]
            piece_ids = chunk_ids[i:i + piece_size]
            
            if self.should_use_cpu_only():
                self.logger.info(f"Memory critical, processing piece {i//piece_size + 1} on CPU")
                try:
                    piece_embeddings = model.encode(piece_texts, convert_to_tensor=False, device='cpu')
                    for j, chunk_id in enumerate(piece_ids):
                        self.cpu_cache[chunk_id] = piece_embeddings[j]
                    all_embeddings.append(piece_embeddings)
                except Exception as e:
                    self.logger.error(f"CPU encoding failed for piece {i//piece_size + 1}: {e}")
                    raise e
            else:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    piece_embeddings = model.encode(piece_texts, convert_to_tensor=False, device='cuda')
                    
                    for j, chunk_id in enumerate(piece_ids):
                        self.gpu_cache[chunk_id] = piece_embeddings[j]
                    
                    all_embeddings.append(piece_embeddings)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.warning(f"GPU OOM on piece {i//piece_size + 1}, switching to CPU")
                        self.gpu_cache.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        try:
                            piece_embeddings = model.encode(piece_texts, convert_to_tensor=False, device='cpu')
                            for j, chunk_id in enumerate(piece_ids):
                                self.cpu_cache[chunk_id] = piece_embeddings[j]
                            all_embeddings.append(piece_embeddings)
                        except Exception as cpu_e:
                            self.logger.error(f"CPU fallback also failed: {cpu_e}")
                            raise cpu_e
                    else:
                        raise e
            
            gc.collect()
            
            self.logger.info(f"Completed piece {i//piece_size + 1}/{(len(texts) + piece_size - 1)//piece_size}")
        
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        else:
            return np.vstack(all_embeddings)

memory_manager = HybridMemoryManager()

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}

class DocumentCacheManager:
    """Manages caching of chunked documents to avoid reprocessing"""
    
    def __init__(self, cache_dir: str = "files"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Document cache initialized at: {self.cache_dir.absolute()}")
    
    def _get_document_hash(self, url: str) -> str:
        """Generate a hash for the document URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a document URL"""
        doc_hash = self._get_document_hash(url)
        return self.cache_dir / f"{doc_hash}.pkl"
    
    def _get_metadata_path(self, url: str) -> Path:
        """Get the metadata file path for a document URL"""
        doc_hash = self._get_document_hash(url)
        return self.cache_dir / f"{doc_hash}_metadata.json"
    
    def is_cached(self, url: str) -> bool:
        """Check if a document is already cached"""
        cache_path = self._get_cache_path(url)
        metadata_path = self._get_metadata_path(url)
        return cache_path.exists() and metadata_path.exists()
    
    def save_chunks(self, url: str, chunks: List[DocumentChunk], model_type: str, file_size: int):
        """Save chunked document to cache"""
        try:
            cache_path = self._get_cache_path(url)
            metadata_path = self._get_metadata_path(url)
            
            # Save chunks
            chunk_data = {
                'chunks': [{'text': chunk.text, 'metadata': chunk.metadata} for chunk in chunks],
                'model_type': model_type,
                'file_size': file_size,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(chunk_data, f)
            
            # Save metadata
            metadata = {
                'url': url,
                'model_type': model_type,
                'file_size': file_size,
                'chunk_count': len(chunks),
                'cached_at': datetime.utcnow().isoformat(),
                'hash': self._get_document_hash(url)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Cached {len(chunks)} chunks for document: {url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save chunks to cache: {e}")
            return False
    
    def load_chunks(self, url: str) -> Tuple[List[DocumentChunk], str, int]:
        """Load chunked document from cache"""
        try:
            cache_path = self._get_cache_path(url)
            metadata_path = self._get_metadata_path(url)
            
            if not cache_path.exists() or not metadata_path.exists():
                raise FileNotFoundError("Cache files not found")
            
            # Load chunks
            with open(cache_path, 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Reconstruct DocumentChunk objects
            chunks = []
            for chunk_info in chunk_data['chunks']:
                chunk = DocumentChunk(
                    text=chunk_info['text'],
                    metadata=chunk_info.get('metadata', {})
                )
                chunks.append(chunk)
            
            self.logger.info(f"Loaded {len(chunks)} chunks from cache for document: {url}")
            return chunks, chunk_data['model_type'], chunk_data['file_size']
            
        except Exception as e:
            self.logger.error(f"Failed to load chunks from cache: {e}")
            raise e
    
    def get_cache_info(self, url: str) -> Optional[Dict]:
        """Get cache information for a document"""
        try:
            metadata_path = self._get_metadata_path(url)
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get cache info: {e}")
            return None
    
    def clear_cache(self, url: str = None):
        """Clear cache for specific URL or all cache"""
        try:
            if url:
                cache_path = self._get_cache_path(url)
                metadata_path = self._get_metadata_path(url)
                
                if cache_path.exists():
                    cache_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                
                self.logger.info(f"Cleared cache for: {url}")
            else:
                # Clear all cache
                for file_path in self.cache_dir.glob("*.pkl"):
                    file_path.unlink()
                for file_path in self.cache_dir.glob("*_metadata.json"):
                    file_path.unlink()
                
                self.logger.info("Cleared all document cache")
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            metadata_files = list(self.cache_dir.glob("*_metadata.json"))
            
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_dir": str(self.cache_dir.absolute()),
                "cached_documents": len(cache_files),
                "total_cache_size_bytes": total_size,
                "total_cache_size_mb": total_size / (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}

document_cache_manager = DocumentCacheManager()

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

class SemanticSearchEngine:
    def __init__(self, model_type: str = 'heavy'):
        self.chunks = []
        self.chunk_ids = []  
        self.embeddings = []
        self.request_id = str(uuid.uuid4()) 
        self.model_type = model_type  
        
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the search engine with hybrid memory management"""
        start_idx = len(self.chunks)
        
        chunk_ids = [f"{self.request_id}_chunk_{start_idx + i}" for i in range(len(chunks))]
        
        self.chunks.extend(chunks)
        self.chunk_ids.extend(chunk_ids)
        
        texts = [chunk.text for chunk in chunks]
        
        new_embeddings = memory_manager.encode_texts_hybrid(texts, chunk_ids, self.model_type)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        logger.info(f"Added {len(chunks)} chunks with {self.model_type} model. Total chunks: {len(self.chunks)}")
    
    def add_chunks_piece_by_piece(self, chunks: List[DocumentChunk]):
        """Add document chunks piece by piece to prevent memory issues"""
        start_idx = len(self.chunks)
        
        chunk_ids = [f"{self.request_id}_chunk_{start_idx + i}" for i in range(len(chunks))]
        
        self.chunks.extend(chunks)
        self.chunk_ids.extend(chunk_ids)
        
        texts = [chunk.text for chunk in chunks]
        
        new_embeddings = memory_manager.encode_texts_piece_by_piece(texts, chunk_ids, self.model_type)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        logger.info(f"Added {len(chunks)} chunks with piece-by-piece processing using {self.model_type} model. Total chunks: {len(self.chunks)}")
    
    def clear_memory(self):
        """Clear all chunks and embeddings from memory"""
        for chunk_id in self.chunk_ids:
            if chunk_id in memory_manager.gpu_cache:
                memory_manager.gpu_cache.pop(chunk_id, None)
            if chunk_id in memory_manager.cpu_cache:
                memory_manager.cpu_cache.pop(chunk_id, None)
        
        self.chunks.clear()
        self.chunk_ids.clear()
        self.embeddings = np.array([])
        
        logger.info(f"Cleared memory for search engine (request_id: {self.request_id})")
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Search for most relevant chunks using hybrid memory management"""
        if len(self.chunks) == 0:
            return []
        
        query_embedding = memory_manager.encode_texts_hybrid([query], model_type=self.model_type)[0]
        
        embeddings, available_ids = memory_manager.get_embeddings_for_search(self.chunk_ids)
        
        if len(embeddings) == 0:
            logger.warning("No embeddings available for search")
            return []
        
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            if idx < len(available_ids):
                chunk_id = available_ids[idx]
                chunk_idx = self.chunk_ids.index(chunk_id)
                chunk = self.chunks[chunk_idx]
                chunk.similarity_score = similarities[idx]
                relevant_chunks.append(chunk)
        
        relevant_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return relevant_chunks
    
    def search_progressive(self, query: str, start_k: int = 5, max_k: int = None, used_chunks: set = None) -> List[DocumentChunk]:
        """Search progressively with increasing chunk count, avoiding previously used chunks"""
        if len(self.chunks) == 0:
            return []
        
        if max_k is None:
            max_k = len(self.chunks)
        
        if used_chunks is None:
            used_chunks = set()
        
        query_embedding = memory_manager.encode_texts_hybrid([query], model_type=self.model_type)[0]
        
        embeddings, available_ids = memory_manager.get_embeddings_for_search(self.chunk_ids)
        
        if len(embeddings) == 0:
            return [], 0, max_k
        
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        embedding_to_chunk = {i: self.chunk_ids.index(chunk_id) for i, chunk_id in enumerate(available_ids)}
        
        all_indices = np.argsort(similarities)[::-1]
        
        available_indices = [idx for idx in all_indices if embedding_to_chunk[idx] not in used_chunks]
        
        if not available_indices:
            return [], 0, max_k
        
        current_k = min(start_k, len(available_indices))
        new_indices = available_indices[:current_k]
        
        relevant_chunks = []
        for idx in new_indices:
            chunk_idx = embedding_to_chunk[idx]
            chunk = self.chunks[chunk_idx]
            chunk.similarity_score = similarities[idx]
            relevant_chunks.append(chunk)
            used_chunks.add(chunk_idx)
        
        relevant_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return relevant_chunks, current_k, max_k
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """Hybrid search combining semantic similarity with keyword matching"""
        if len(self.chunks) == 0:
            return []
        
        query_embedding = memory_manager.encode_texts_hybrid([query], model_type=self.model_type)[0]
        
        embeddings, available_ids = memory_manager.get_embeddings_for_search(self.chunk_ids)
        
        if len(embeddings) == 0:
            return []
        
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        embedding_to_chunk = {i: self.chunk_ids.index(chunk_id) for i, chunk_id in enumerate(available_ids)}
        
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for i, chunk_id in enumerate(available_ids):
            chunk_idx = embedding_to_chunk[i]
            chunk = self.chunks[chunk_idx]
            chunk_words = set(chunk.text.lower().split())
            keyword_matches = len(query_words.intersection(chunk_words))
            keyword_scores.append(keyword_matches)
        
        max_keyword = max(keyword_scores) if keyword_scores else 1
        keyword_scores = [score / max_keyword for score in keyword_scores]
        
        combined_scores = [0.7 * sim + 0.3 * kw for sim, kw in zip(similarities, keyword_scores)]
        
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            chunk_idx = embedding_to_chunk[idx]
            chunk = self.chunks[chunk_idx]
            chunk.similarity_score = combined_scores[idx]
            relevant_chunks.append(chunk)
        
        relevant_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return relevant_chunks

async def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download PDF: {response.status}")
            return await response.read()

def extract_text_from_pdf_streaming(pdf_bytes: bytes, max_pages_per_chunk: int = 10) -> Generator[str, None, None]:
    """Extract text from PDF in streaming fashion to prevent memory issues"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        total_pages = len(pdf_reader.pages)
        logger.info(f"Processing PDF with {total_pages} pages in streaming mode")
        
        for i in range(0, total_pages, max_pages_per_chunk):
            chunk_text = ""
            end_page = min(i + max_pages_per_chunk, total_pages)
            
            for page_num in range(i, end_page):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        chunk_text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            if chunk_text.strip():
                yield chunk_text.strip()
            
            gc.collect()
            
            logger.info(f"Processed pages {i+1}-{end_page}/{total_pages}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes (legacy method for small files)"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text_piece_by_piece(text: str, chunk_size: int = 300, overlap: int = 50) -> Generator[List[str], None, None]:
    """Split text into very small chunks for piece-by-piece processing"""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    max_chunks_per_piece = 20 
    
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
        
        if len(chunks) >= max_chunks_per_piece:
            valid_chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
            if valid_chunks:
                yield valid_chunks
            chunks = []
            gc.collect()  
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    if chunks:
        valid_chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
        if valid_chunks:
            yield valid_chunks

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks with better strategy (legacy method)"""
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

@api_router.get("/queue/status")
async def get_queue_status():
    """Get current queue status"""
    try:
        status = request_queue.get_queue_status()
        return {
            "queue_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@api_router.post("/queue/clear")
async def clear_queue():
    """Clear any stuck queue (for testing)"""
    try:
        if request_queue.processing_lock.locked():
            request_queue.processing_lock.release()
        
        return {
            "message": "Queue cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear queue: {str(e)}")

@api_router.post("/test/response")
async def test_response():
    """Test endpoint to verify response format works"""
    try:
        test_response = HackRXResponse(answers=["Test answer 1", "Test answer 2"])
        logger.info("Test response created successfully")
        return test_response
    except Exception as e:
        logger.error(f"Test response error: {e}")
        raise HTTPException(status_code=500, detail=f"Test response failed: {str(e)}")

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

@api_router.get("/memory/stats")
async def get_memory_stats():
    """Get hybrid memory management statistics"""
    try:
        stats = memory_manager.get_memory_stats()
        return {
            "memory_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")

@api_router.post("/memory/cleanup")
async def cleanup_memory():
    """Manually trigger memory cleanup"""
    try:
        memory_manager.cleanup_memory()
        return {
            "message": "Memory cleanup completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup memory: {str(e)}")

@api_router.get("/memory/status")
async def get_memory_status():
    """Get comprehensive memory status including piece-by-piece processing info"""
    try:
        memory_stats = memory_manager.get_memory_stats()
        system_memory = memory_manager.get_system_memory_usage()
        
        return {
            "memory_stats": memory_stats,
            "system_memory_usage": system_memory,
            "piece_processing_settings": {
                "max_chunk_size": memory_manager.max_chunk_size,
                "max_batch_size": memory_manager.max_batch_size,
                "memory_safety_margin": memory_manager.memory_safety_margin
            },
            "processing_thresholds": {
                "large_file_threshold_mb": 100,
                "large_text_threshold_chars": 500000,
                "high_memory_threshold": 0.8,
                "high_gpu_threshold": 0.7
            },
            "model_configurations": memory_manager.model_configs,
            "model_thresholds": {
                "small_pdf_threshold_mb": memory_manager.small_pdf_threshold / 1024 / 1024,
                "medium_pdf_threshold_mb": memory_manager.medium_pdf_threshold / 1024 / 1024
            },
            "loaded_models": list(memory_manager.embedding_models.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")

@api_router.get("/cache/stats")
async def get_cache_stats():
    """Get document cache statistics"""
    try:
        cache_stats = document_cache_manager.get_cache_stats()
        return {
            "cache_stats": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@api_router.get("/cache/info/{url:path}")
async def get_cache_info(url: str):
    """Get cache information for a specific document URL"""
    try:
        # Decode URL if it's encoded
        import urllib.parse
        decoded_url = urllib.parse.unquote(url)
        
        cache_info = document_cache_manager.get_cache_info(decoded_url)
        if cache_info is None:
            raise HTTPException(status_code=404, detail="Document not found in cache")
        
        return {
            "cache_info": cache_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache info: {str(e)}")

@api_router.delete("/cache/clear")
async def clear_cache(url: str = None):
    """Clear document cache - specific URL or all cache"""
    try:
        if url:
            import urllib.parse
            decoded_url = urllib.parse.unquote(url)
            document_cache_manager.clear_cache(decoded_url)
            message = f"Cache cleared for: {decoded_url}"
        else:
            document_cache_manager.clear_cache()
            message = "All document cache cleared"
        
        return {
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@api_router.get("/cache/list")
async def list_cached_documents():
    """List all cached documents"""
    try:
        cache_dir = document_cache_manager.cache_dir
        cached_docs = []
        
        for metadata_file in cache_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    cached_docs.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
                continue
        
        return {
            "cached_documents": cached_docs,
            "total_count": len(cached_docs),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing cached documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list cached documents: {str(e)}")

async def process_hackrx_request(request: HackRXRequest):
    """Internal function to process HackRX request with caching"""
    logger.info(f"Processing request with document: {request.documents}")
    logger.info(f"Questions: {len(request.questions)}")
    
    # Check if document is already cached
    if document_cache_manager.is_cached(request.documents):
        logger.info(f"Document found in cache: {request.documents}")
        try:
            # Load chunks from cache
            chunks, model_type, file_size = document_cache_manager.load_chunks(request.documents)
            logger.info(f"Loaded {len(chunks)} chunks from cache")
            
            # Create search engine with cached chunks
            search_engine = SemanticSearchEngine(model_type=model_type)
            search_engine.add_chunks(chunks)
            
            # Process questions
            answers = []
            for i, question in enumerate(request.questions):
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
                
                normalized_question = normalize_question(question)
                answer = await generate_answer_with_retry(question, search_engine, start_k=5)
                answers.append(answer)
                logger.info(f"Generated answer: {answer[:100]}...")
                
                if i % 3 == 0:
                    memory_manager.cleanup_memory()
                    logger.info("Periodic memory cleanup performed")
            
            logger.info("All questions processed successfully from cache")
            logger.info(f"Returning response with {len(answers)} answers")
            
            try:
                final_response = HackRXResponse(answers=answers)
                logger.info("Response object created successfully")
                logger.info(f"Response content: {final_response.answers}")
                return final_response
            except Exception as response_error:
                logger.error(f"Error creating response: {response_error}")
                return HackRXResponse(answers=answers)
                
        except Exception as cache_error:
            logger.error(f"Error loading from cache: {cache_error}")
            logger.info("Falling back to normal processing")
            # Continue with normal processing if cache loading fails
    
    # Normal processing path (not cached or cache failed)
    initial_stats = memory_manager.get_memory_stats()
    logger.info(f"Initial memory stats: {initial_stats}")
    
    pdf_bytes = await download_pdf(request.documents)
    file_size = len(pdf_bytes)
    logger.info(f"PDF downloaded successfully. File size: {file_size / 1024 / 1024:.1f}MB")
    
    use_piece_processing = memory_manager.should_use_piece_processing(file_size, 0)
    
    if use_piece_processing:
        logger.info("Using piece-by-piece processing for large document")
        response = await process_large_document_piece_by_piece(request, pdf_bytes, file_size)
    else:
        logger.info("Using standard processing for document")
        response = await process_document_standard(request, pdf_bytes, file_size)
    
    logger.info("All questions processed successfully")
    logger.info(f"Returning response with {len(response.answers)} answers")
    
    try:
        final_response = HackRXResponse(answers=response.answers)
        logger.info("Response object created successfully")
        logger.info(f"Response content: {final_response.answers}")
        return final_response
    except Exception as response_error:
        logger.error(f"Error creating response: {response_error}")
        return HackRXResponse(answers=response.answers)

@api_router.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """Main endpoint with request queue system"""
    request_id = str(uuid.uuid4())
    logger.info(f"📥 Received request {request_id}")
    
    try:
        result = await request_queue.process_request(request_id, process_hackrx_request, request)
        return result
    except Exception as e:
        logger.error(f"Error in hackrx_run: {str(e)}")
        try:
            memory_manager.cleanup_memory()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def process_large_document_piece_by_piece(request: HackRXRequest, pdf_bytes: bytes, file_size: int) -> HackRXResponse:
    """Process large documents piece by piece to prevent memory issues"""
    try:
        logger.info("Starting piece-by-piece document processing")
        
        model_type = memory_manager.select_model_for_pdf_size(file_size)
        
        search_engine = SemanticSearchEngine(model_type=model_type)
        
        total_chunks = 0
        piece_count = 0
        all_document_chunks = []  # Collect all chunks for caching
        
        for text_piece in extract_text_from_pdf_streaming(pdf_bytes, max_pages_per_chunk=5):
            piece_count += 1
            logger.info(f"Processing text piece {piece_count}, length: {len(text_piece)} characters")
            
            if len(text_piece) > 100000:  
                logger.info(f"Large text piece detected, using piece-by-piece chunking")
                
                for chunk_piece in chunk_text_piece_by_piece(text_piece, chunk_size=200, overlap=30):
                    document_chunks = [DocumentChunk(text=chunk_text) for chunk_text in chunk_piece]
                    search_engine.add_chunks_piece_by_piece(document_chunks)
                    all_document_chunks.extend(document_chunks)  # Collect for caching
                    total_chunks += len(chunk_piece)
                    
                    memory_manager.cleanup_memory()
                    gc.collect()
                    
                    logger.info(f"Added chunk piece with {len(chunk_piece)} chunks. Total chunks: {total_chunks}")
            else:
                chunks_text = chunk_text(text_piece, chunk_size=300, overlap=50)
                document_chunks = [DocumentChunk(text=chunk_text) for chunk_text in chunks_text]
                search_engine.add_chunks_piece_by_piece(document_chunks)
                all_document_chunks.extend(document_chunks)  # Collect for caching
                total_chunks += len(chunks_text)
                
                memory_manager.cleanup_memory()
                gc.collect()
                
                logger.info(f"Added text piece with {len(chunks_text)} chunks. Total chunks: {total_chunks}")
        
        logger.info(f"Document processing completed. Total chunks: {total_chunks}")
        
        # Save all chunks to cache for future use
        cache_success = document_cache_manager.save_chunks(
            request.documents, all_document_chunks, model_type, file_size
        )
        if cache_success:
            logger.info(f"Successfully cached {len(all_document_chunks)} chunks for future requests")
        else:
            logger.warning("Failed to cache chunks, but continuing with processing")
        
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            
            normalized_question = normalize_question(question)
            answer = await generate_answer_with_retry(question, search_engine, start_k=3)  
            answers.append(answer)
            logger.info(f"Generated answer: {answer[:100]}...")
            
            memory_manager.cleanup_memory()
            gc.collect()
            
            if i % 2 == 0:  
                logger.info("Aggressive memory cleanup performed")
        
        final_stats = memory_manager.get_memory_stats()
        logger.info(f"Final memory stats: {final_stats}")
        
        logger.info("Piece-by-piece processing completed successfully")
        
        response = HackRXResponse(answers=answers)
        logger.info("Response created, starting cleanup")
        
        try:
            memory_manager.cleanup_memory()
            search_engine.clear_memory()
            memory_manager.clear_model_cache()
            logger.info("Cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error (non-critical): {cleanup_error}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in piece-by-piece processing: {str(e)}")
        try:
            memory_manager.cleanup_memory()
            memory_manager.clear_model_cache()
        except:
            pass
        raise e

async def process_document_standard(request: HackRXRequest, pdf_bytes: bytes, file_size: int) -> HackRXResponse:
    """Process documents using standard method for smaller files"""
    try:
        text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"Extracted text length: {len(text)} characters")
        
        if file_size > 50 * 1024 * 1024:  
            chunk_size = 500
            overlap = 100
            logger.info(f"Large file detected, using smaller chunks: {chunk_size} words with {overlap} overlap")
        else:
            chunk_size = 1000
            overlap = 200
        
        chunks_text = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        logger.info(f"Created {len(chunks_text)} chunks")
        
        document_chunks = [DocumentChunk(text=chunk_text) for chunk_text in chunks_text]
        
        model_type = memory_manager.select_model_for_pdf_size(file_size)
        
        # Save chunks to cache for future use
        cache_success = document_cache_manager.save_chunks(
            request.documents, document_chunks, model_type, file_size
        )
        if cache_success:
            logger.info(f"Successfully cached {len(document_chunks)} chunks for future requests")
        else:
            logger.warning("Failed to cache chunks, but continuing with processing")
        
        search_engine = SemanticSearchEngine(model_type=model_type)
        search_engine.add_chunks(document_chunks)
        
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")

            normalized_question = normalize_question(question)
            answer = await generate_answer_with_retry(question, search_engine, start_k=5)
            answers.append(answer)
            logger.info(f"Generated answer: {answer[:100]}...")
            
            if i % 3 == 0:  
                memory_manager.cleanup_memory()
                logger.info("Periodic memory cleanup performed")
        
        final_stats = memory_manager.get_memory_stats()
        logger.info(f"Final memory stats: {final_stats}")
        
        logger.info("Standard processing completed successfully")
        
        response = HackRXResponse(answers=answers)
        logger.info("Response created, starting cleanup")
        
        try:
            memory_manager.cleanup_memory()
            search_engine.clear_memory()
            memory_manager.clear_model_cache()
            logger.info("Cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error (non-critical): {cleanup_error}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in standard processing: {str(e)}")
        try:
            memory_manager.cleanup_memory()
            memory_manager.clear_model_cache()
        except:
            pass
        raise e

async def process_simple_request(request: HackRXRequest):
    """Internal function to process simple HackRX request with caching"""
    logger.info(f"Processing simple request with document: {request.documents}")
    logger.info(f"Questions: {len(request.questions)}")
    
    # Check if document is already cached
    if document_cache_manager.is_cached(request.documents):
        logger.info(f"Document found in cache: {request.documents}")
        try:
            # Load chunks from cache
            chunks, model_type, file_size = document_cache_manager.load_chunks(request.documents)
            logger.info(f"Loaded {len(chunks)} chunks from cache")
            
            # Create search engine with cached chunks
            search_engine = SemanticSearchEngine(model_type=model_type)
            search_engine.add_chunks(chunks)
            
            # Process questions
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
            
            logger.info("All questions processed successfully from cache")
            logger.info(f"Returning response with {len(answers)} answers")
            return HackRXResponse(answers=answers)
            
        except Exception as cache_error:
            logger.error(f"Error loading from cache: {cache_error}")
            logger.info("Falling back to normal processing")
            # Continue with normal processing if cache loading fails
    
    # Normal processing path (not cached or cache failed)
    pdf_bytes = await download_pdf(request.documents)
    logger.info("PDF downloaded successfully")
    
    text = extract_text_from_pdf(pdf_bytes)
    logger.info(f"Extracted text length: {len(text)} characters")
    
    chunks_text = chunk_text(text)
    logger.info(f"Created {len(chunks_text)} chunks")
    
    document_chunks = [DocumentChunk(text=chunk_text) for chunk_text in chunks_text]
    
    # Save chunks to cache for future use
    file_size = len(pdf_bytes)
    cache_success = document_cache_manager.save_chunks(
        request.documents, document_chunks, 'heavy', file_size
    )
    if cache_success:
        logger.info(f"Successfully cached {len(document_chunks)} chunks for future requests")
    else:
        logger.warning("Failed to cache chunks, but continuing with processing")
    
    search_engine = SemanticSearchEngine(model_type='heavy')
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
    logger.info(f"Returning response with {len(answers)} answers")
    return HackRXResponse(answers=answers)

@api_router.post("/hackrx/run/simple", response_model=HackRXResponse)
async def hackrx_run_simple(request: HackRXRequest):
    """Simple endpoint with request queue system"""
    request_id = str(uuid.uuid4())
    logger.info(f"📥 Received simple request {request_id}")
    
    try:
        result = await request_queue.process_request(request_id, process_simple_request, request)
        return result
    except Exception as e:
        logger.error(f"Error in hackrx_run_simple: {str(e)}")
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