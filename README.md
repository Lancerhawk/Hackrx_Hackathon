# 🚀 HackRx AI - Advanced Document Processing & Question Answering Platform

## 📋 Overview

HackRx AI is a sophisticated document processing and question-answering platform that leverages advanced AI technologies to extract insights from PDF documents. The platform features intelligent memory management, request queuing, and robust processing capabilities for handling documents of any size.

## ✨ Key Features

### 🧠 **Intelligent Memory Management**
- **Hybrid GPU/CPU Processing**: Dynamically switches between GPU and CPU based on memory availability
- **Dynamic Model Selection**: Automatically chooses optimal embedding models based on document size
- **Memory Offloading**: Proactively moves data between GPU VRAM and CPU RAM to prevent OOM errors
- **Aggressive Cleanup**: Comprehensive memory management with automatic garbage collection

### 📊 **Advanced Document Processing**
- **Streaming PDF Extraction**: Processes large PDFs page-by-page to prevent memory issues
- **Piece-by-Piece Chunking**: Breaks down documents into manageable chunks for processing
- **Adaptive Processing Strategy**: Automatically selects processing method based on document size and system resources
- **Multiple Embedding Models**: Supports heavy, medium, and light models for different use cases

### 🔄 **Request Queue System**
- **Sequential Processing**: Ensures only one request processes at a time
- **Fair Queue Management**: First-come, first-served processing order
- **Resource Protection**: Prevents memory conflicts between concurrent requests
- **Queue Status Monitoring**: Real-time queue status and request tracking

### 🎯 **AI-Powered Question Answering**
- **Semantic Search**: Advanced vector-based document search using embeddings
- **Progressive Search**: Iteratively expands search scope for better answers
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Retry Logic**: Intelligent retry mechanism with TPM limit handling

## 🏗️ Architecture

### **Core Components**

#### **1. HybridMemoryManager**
```python
class HybridMemoryManager:
    - GPU/CPU memory management
    - Dynamic model selection
    - Memory offloading and cleanup
    - System resource monitoring
```

#### **2. RequestQueue**
```python
class RequestQueue:
    - Sequential request processing
    - Queue status management
    - Request tracking and logging
```

#### **3. SemanticSearchEngine**
```python
class SemanticSearchEngine:
    - Document chunking and embedding
    - Semantic search algorithms
    - Progressive search capabilities
    - Memory-efficient processing
```

### **Processing Pipeline**

1. **Document Reception** → Request queued if system busy
2. **PDF Download** → Streaming download with progress tracking
3. **Text Extraction** → Page-by-page extraction for large documents
4. **Chunking** → Intelligent text chunking with overlap
5. **Embedding Generation** → Model selection based on document size
6. **Question Processing** → Semantic search with progressive expansion
7. **Answer Generation** → AI-powered answer generation with retry logic
8. **Memory Cleanup** → Comprehensive resource cleanup

## 🛠️ Technical Stack

### **Backend Technologies**
- **FastAPI**: High-performance web framework
- **Motor**: Asynchronous MongoDB driver
- **Sentence Transformers**: State-of-the-art embedding models
- **PyTorch**: Deep learning framework for GPU acceleration
- **Groq API**: High-speed LLM inference (Llama 3.1-8b-instant)
- **PyPDF2**: PDF text extraction
- **NLTK**: Natural language processing
- **NumPy & scikit-learn**: Numerical computing and similarity calculations

### **AI Models**
- **Heavy Model**: `nomic-ai/nomic-embed-text-v1` (768d) - High accuracy for small documents
- **Medium Model**: `all-mpnet-base-v2` (768d) - Balanced performance for medium documents
- **Light Model**: `all-MiniLM-L6-v2` (384d) - Fast processing for large documents

### **Infrastructure**
- **MongoDB**: Document storage and caching
- **GPU Support**: NVIDIA GPU acceleration with fallback to CPU
- **Memory Monitoring**: Real-time GPU and system memory tracking
- **Queue Management**: Asynchronous request processing

## 📡 API Endpoints

### **Main Endpoints**

#### **1. Document Processing & Q&A**
```http
POST /api/hackrx/run
Content-Type: application/json

{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is this document about?", "What are the key points?"]
}
```

**Response:**
```json
{
  "answers": [
    "This document is about...",
    "The key points include..."
  ]
}
```

#### **2. Simple Processing**
```http
POST /api/hackrx/run/simple
```
Simplified processing without advanced features for testing.

### **Monitoring & Management**

#### **3. Queue Status**
```http
GET /api/queue/status
```
Returns current queue status and processing information.

#### **4. Memory Statistics**
```http
GET /api/memory/status
```
Comprehensive memory usage and system resource information.

#### **5. Memory Cleanup**
```http
POST /api/memory/cleanup
```
Manually trigger memory cleanup operations.

#### **6. Queue Management**
```http
POST /api/queue/clear
```
Clear stuck queue (for testing purposes).

### **Health & Testing**

#### **7. Health Check**
```http
GET /api/
```
Basic health check endpoint.

#### **8. Test Response**
```http
POST /api/test/response
```
Test endpoint for response format verification.

#### **9. Status Management**
```http
POST /api/status
GET /api/status
```
Client status tracking and monitoring.

## 🔧 Configuration

### **Environment Variables**
```bash
MONGO_URL=mongodb://localhost:27017
DB_NAME=hackrx_db
API_KEY=your_groq_api_key
PORT=8000
HOST=0.0.0.0
```

### **Model Configuration**
```python
model_configs = {
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
```

### **Processing Thresholds**
- **Small PDF**: ≤ 10MB → Heavy model
- **Medium PDF**: 10-50MB → Medium model
- **Large PDF**: > 50MB → Light model
- **Piece-by-Piece**: > 100MB or > 500k characters

## 🚀 Performance Features

### **Memory Optimization**
- **Dynamic Offloading**: Moves data between GPU and CPU based on usage
- **LRU Caching**: Efficient cache management for embeddings
- **Aggressive Cleanup**: Automatic memory cleanup after each request
- **Resource Monitoring**: Real-time GPU and system memory tracking

### **Processing Optimization**
- **Streaming Extraction**: Page-by-page PDF processing
- **Micro-batching**: Small batch processing to prevent OOM
- **Adaptive Chunking**: Dynamic chunk size based on document characteristics
- **Model Reuse**: Efficient model loading and caching

### **Queue Optimization**
- **Sequential Processing**: Prevents resource conflicts
- **Fair Scheduling**: First-come, first-served processing
- **Status Tracking**: Real-time queue and processing status
- **Error Recovery**: Graceful error handling and recovery

## 📊 Monitoring & Logging

### **Logging Levels**
- **INFO**: Processing steps and progress
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Critical errors and failures
- **DEBUG**: Detailed debugging information

### **Metrics Tracked**
- **Memory Usage**: GPU VRAM and system RAM
- **Processing Time**: Document processing duration
- **Queue Status**: Request queue length and processing status
- **Model Performance**: Embedding generation speed and accuracy
- **Error Rates**: Processing failures and retry attempts

## 🔒 Security & Reliability

### **Error Handling**
- **Graceful Degradation**: Fallback to CPU when GPU unavailable
- **Retry Logic**: Intelligent retry with exponential backoff
- **Resource Protection**: Memory limits and cleanup
- **Queue Management**: Prevents system overload

### **Data Management**
- **MongoDB Integration**: Persistent storage and caching
- **Memory Cleanup**: Automatic resource cleanup
- **Request Isolation**: Separate processing for each request
- **Error Recovery**: Comprehensive error handling

## 🧪 Testing

### **Test Endpoints**
- **Response Testing**: `/api/test/response`
- **Queue Testing**: `/api/queue/status`
- **Memory Testing**: `/api/memory/status`
- **Health Check**: `/api/`

### **Performance Testing**
- **Large Document Processing**: Test with 100MB+ PDFs
- **Concurrent Requests**: Test queue system with multiple requests
- **Memory Stress Testing**: Test memory management under load
- **Timeout Testing**: Test with long-running processes

## 📈 Scalability

### **Horizontal Scaling**
- **Stateless Design**: Easy to scale across multiple instances
- **Queue Management**: Handles multiple concurrent requests
- **Resource Isolation**: Each request processes independently
- **Load Balancing**: Compatible with load balancers

### **Vertical Scaling**
- **GPU Acceleration**: Leverages GPU for faster processing
- **Memory Management**: Efficient memory usage and cleanup
- **Model Optimization**: Multiple model options for different workloads
- **Resource Monitoring**: Real-time resource tracking

## 🎯 Use Cases

### **Document Analysis**
- **Legal Documents**: Contract analysis and clause extraction
- **Medical Reports**: Patient record analysis and information extraction
- **Financial Documents**: Report analysis and data extraction
- **Academic Papers**: Research paper analysis and summarization

### **Question Answering**
- **Customer Support**: Automated document-based support
- **Research**: Quick information extraction from large documents
- **Compliance**: Regulatory document analysis and compliance checking
- **Education**: Automated document-based learning and assessment

## 🔮 Future Enhancements

### **Planned Features**
- **Multi-language Support**: Support for multiple languages
- **Document Comparison**: Compare multiple documents
- **Advanced Caching**: Redis-based caching for better performance
- **API Rate Limiting**: Sophisticated rate limiting and throttling
- **WebSocket Support**: Real-time progress updates
- **Batch Processing**: Process multiple documents simultaneously

### **Performance Improvements**
- **Model Optimization**: Quantized models for faster inference
- **Distributed Processing**: Multi-GPU and multi-node processing
- **Advanced Caching**: Intelligent caching strategies
- **Streaming Responses**: Real-time response streaming

## 📞 Support

### **Documentation**
- **API Documentation**: Available at `/docs` when server is running
- **Code Comments**: Comprehensive inline documentation
- **Logging**: Detailed logging for debugging and monitoring

### **Troubleshooting**
- **Memory Issues**: Check `/api/memory/status`
- **Queue Issues**: Check `/api/queue/status`
- **Performance Issues**: Monitor logs and memory usage
- **Timeout Issues**: Check processing time and resource usage

## 🏆 Hackathon Features

This platform was specifically designed for hackathon requirements with:

- **Fixed API Endpoint**: `/api/hackrx/run` for submission
- **Robust Error Handling**: Prevents crashes during demos
- **Memory Management**: Handles large documents without OOM
- **Queue System**: Manages multiple concurrent requests
- **Performance Optimization**: Fast processing for live demos
- **Comprehensive Logging**: Easy debugging during presentations

---

**Built with ❤️ for HackRx Hackathon** 