import os
import logging
import time
from typing import List, Dict, Tuple, Optional, Any

# Vector DB imports
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
import qdrant_client

# Embedding model
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# LLM
from llama_index.llms.groq import Groq

# Document processing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline for document processing and retrieval."""
    
    def __init__(
        self, 
        qdrant_url: Optional[str] = None, 
        qdrant_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        collection_name: str = "study_materials",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: str = "llama-3.3-70b-versatile",
        storage_dir: str = "./storage",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.groq_api_key = groq_api_key
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.storage_dir = storage_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Ensure temp directory exists for document processing
        os.makedirs("./data", exist_ok=True)
        
        # Initialize node parser for text splitting
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_llm()
        
        # Set up vector store and other components
        self.qdrant_client = None
        self.vector_store = None
        self.index = None
        self.query_engine = None
        
        # Initialize vector store
        self._initialize_vector_store()

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.embed_model = FastEmbedEmbedding(model_name=self.embedding_model_name)
            Settings.embed_model = self.embed_model
            logger.info(f"Embedding model initialized: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            if not self.groq_api_key:
                logger.warning("Groq API key not provided. LLM not initialized.")
                self.llm = None
                return
                
            self.llm = Groq(model=self.llm_model_name, api_key=self.groq_api_key)
            Settings.llm = self.llm
            logger.info(f"Language model initialized: {self.llm_model_name}")
        except Exception as e:
            logger.error(f"Error initializing language model: {e}")
            raise

    def _initialize_vector_store(self):
        """Initialize the vector store."""
        if not self.qdrant_url or not self.qdrant_api_key:
            logger.warning("Qdrant URL or API key not provided. Vector store not initialized.")
            return
            
        try:
            self.qdrant_client = qdrant_client.QdrantClient(
                api_key=self.qdrant_api_key, 
                url=self.qdrant_url
            )
            
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client, 
                collection_name=self.collection_name
            )
            
            logger.info(f"Vector store initialized: Qdrant ({self.collection_name})")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
            
    def clear_collection(self) -> bool:
        """Clear the Qdrant collection to remove old data before indexing new files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.qdrant_client:
            logger.warning("Qdrant client not initialized. Cannot clear collection.")
            return False
            
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                # Delete collection
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                
                # Recreate collection with same settings
                logger.info(f"Recreating collection: {self.collection_name}")
                dimension = 768  # Default dimension for BAAI/bge-base-en-v1.5
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_client.http.models.VectorParams(
                        size=dimension,
                        distance=qdrant_client.http.models.Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} cleared and recreated successfully")
                
                # Reinitialize vector store with new collection
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client, 
                    collection_name=self.collection_name
                )
                return True
            else:
                logger.info(f"Collection {self.collection_name} does not exist yet, nothing to clear")
                return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise

    def extract_text_from_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint presentation."""
        try:
            ppt = Presentation(file_path)
            text = ""
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text += shape.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PPTX {file_path}: {e}")
            raise

    def process_uploaded_file(self, uploaded_file, temp_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Process an uploaded file and extract text."""
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        try:
            # Save uploaded file to disk temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text based on file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif file_extension in ['.pptx', '.ppt']:
                text = self.extract_text_from_pptx(file_path)
            else:
                return None, f"Unsupported file format: {file_extension}"
            
            # Remove the temporary file
            os.remove(file_path)
            
            # Check if we actually got content
            if not text or len(text.strip()) < 10:
                return None, f"Could not extract meaningful text from {uploaded_file.name}"
                
            return text, None
        except Exception as e:
            # Remove the temporary file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            return None, f"Error processing {uploaded_file.name}: {str(e)}"

    def process_documents(self, documents_dict: Dict[str, str]) -> List[Document]:
        """Convert extracted text to LlamaIndex documents with chunking."""
        all_nodes = []
        
        for filename, text in documents_dict.items():
            # Create a document
            doc = Document(
                text=text,
                metadata={"filename": filename, "source": filename}
            )
            
            # Parse the document into chunks
            nodes = self.node_parser.get_nodes_from_documents([doc])
            
            # Ensure each node has the source filename
            for node in nodes:
                if "filename" not in node.metadata:
                    node.metadata["filename"] = filename
            
            all_nodes.extend(nodes)
            
        logger.info(f"Created {len(all_nodes)} chunks from {len(documents_dict)} documents")
        return all_nodes

    def build_index(self, documents: List[Document], clear_database: bool = True) -> VectorStoreIndex:
        """Build vector index from documents.
        
        Args:
            documents: List of documents to index
            clear_database: If True, clear the Qdrant database before indexing
            
        Returns:
            VectorStoreIndex: The built index
        """
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized. Cannot build index.")
                raise ValueError("Vector store not initialized")
            
            # Clear the Qdrant collection before indexing if requested
            if clear_database:
                logger.info("Clearing Qdrant collection before building new index")
                success = self.clear_collection()
                if not success:
                    logger.error("Failed to clear Qdrant collection")
                    raise ValueError("Failed to clear Qdrant collection")
                
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            index = VectorStoreIndex(
                nodes=documents, 
                storage_context=storage_context,
                show_progress=True
            )
            
            self.index = index
            return index
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise

    def save_index(self, index_name: str = "study_materials_index"):
        """Save the index to disk."""
        if not self.index:
            logger.warning("No index to save")
            return False
            
        try:
            # Persist the index in the storage directory
            index_path = os.path.join(self.storage_dir, index_name)
            
            # Clean the existing directory if it exists to avoid conflicts
            if os.path.exists(index_path):
                # This helps ensure we don't have orphaned files from previous indices
                logger.info(f"Cleaning existing index directory: {index_path}")
                for filename in os.listdir(index_path):
                    file_path = os.path.join(index_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.warning(f"Error deleting {file_path}: {e}")
            
            # Make sure directory exists
            os.makedirs(index_path, exist_ok=True)
            
            # Persist the index
            self.index.storage_context.persist(persist_dir=index_path)
            
            logger.info(f"Index saved to {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def create_query_engine(self, similarity_top_k=7):
        """Create a query engine from the index with configurable parameters."""
        if not self.index:
            logger.warning("No index available for query engine")
            return None
            
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            llm=self.llm if self.llm else None
        )
        return self.query_engine

    def query(self, query_text: str, similarity_top_k=7) -> Dict[str, Any]:
        """Query the RAG system with configurable parameters."""
        start_time = time.time()
        
        if not self.index:
            return {
                "answer": "Sorry, no documents have been indexed yet. Please upload study materials first.",
                "sources": [],
                "elapsed_time": 0
            }
        
        try:
            # Create or update query engine with the specified parameters
            self.create_query_engine(similarity_top_k=similarity_top_k)
                
            # Execute query
            response = self.query_engine.query(query_text)
            
            # Extract sources and prepare response
            sources = []
            if hasattr(response, 'source_nodes'):
                sources = [
                    {
                        "filename": node.metadata.get("filename", "Unknown"),
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": node.score if hasattr(node, 'score') else None
                    }
                    for node in response.source_nodes
                ]
            
            elapsed_time = time.time() - start_time
            
            return {
                "answer": str(response),
                "sources": sources,
                "elapsed_time": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "elapsed_time": time.time() - start_time
            }

    def process_and_index_files(self, uploaded_files_info: List[Dict]) -> bool:
        """Process and index files from uploaded_files_info."""
        try:
            # Extract text content from file info
            documents_dict = {}
            for file_info in uploaded_files_info:
                if "content" in file_info and file_info["content"]:
                    documents_dict[file_info["name"]] = file_info["content"]
            
            if not documents_dict:
                logger.warning("No content found in uploaded files")
                return False
            
            # Convert to chunked LlamaIndex documents
            nodes = self.process_documents(documents_dict)
            
            if not nodes:
                logger.warning("No chunks were created from documents")
                return False
                
            # Build the index
            self.build_index(nodes)
            
            # Save the index
            self.save_index()
            
            # Create query engine
            self.create_query_engine()
            
            return True
        except Exception as e:
            logger.error(f"Error processing and indexing files: {e}")
            return False
            
    def test_pipeline(self) -> Dict[str, Any]:
        """Run a simple test to verify the pipeline is working."""
        if not self.index:
            return {
                "status": "error",
                "message": "No index available. Please upload and index documents first."
            }
            
        try:
            # Try a simple query
            test_query = "What is this document about?"
            result = self.query(test_query, similarity_top_k=1)
            
            if result and "answer" in result and result["answer"]:
                return {
                    "status": "success",
                    "message": "RAG pipeline is working correctly.",
                    "test_query": test_query,
                    "result": result
                }
            else:
                return {
                    "status": "warning",
                    "message": "Query returned empty results. Check document content."
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error testing pipeline: {str(e)}"
            }