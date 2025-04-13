from dotenv import load_dotenv
from llama_parse import LlamaParse
import os
import pickle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import qdrant_client


load_dotenv()

llamaparse_api_key = os.getenv("PARSE_API_KEY")

# Define a function to load parsed data if available, or parse if not
def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        #llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/uber_10q_march_2022.pdf")
        #llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/presentation.pptx")
        llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data(["./data/presentation.pptx", "./data/uber_10q_march_2022.pdf"])

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data


######## QDRANT ###########

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

######### FastEmbedEmbeddings #############

# by default llamaindex uses OpenAI models
from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")


######### Groq API ###########

from llama_index.llms.groq import Groq
groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
#llm = Groq(model="gemma-7b-it", api_key=groq_api_key)

from llama_index.core import Settings
#### Setting llm other than openAI ( by default used openAI's model)
Settings.llm = llm
Settings.embed_model = embed_model

# Function to clear existing Qdrant collection
def clear_qdrant_collection(client, collection_name):
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name in collection_names:
            # Delete collection
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            
            # Recreate collection with same settings
            print(f"Recreating collection: {collection_name}")
            dimension = 768  # Default dimension for BAAI/bge-base-en-v1.5
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=dimension,
                    distance=qdrant_client.http.models.Distance.COSINE
                )
            )
            print(f"Collection {collection_name} cleared and recreated successfully")
            return True
        else:
            print(f"Collection {collection_name} does not exist yet, nothing to clear")
            return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False

# Initialize Qdrant client
client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url)

# Clean existing collection before indexing
collection_name = 'qdrant_rag'
clear_qdrant_collection(client, collection_name)

# Load the documents
llama_parse_documents = load_or_parse_data()

# Create vector store and index
vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=llama_parse_documents, storage_context=storage_context, show_progress=True)

#### PERSIST INDEX #####
#index.storage_context.persist()

#storage_context = StorageContext.from_defaults(persist_dir="./storage")
#index = load_index_from_storage(storage_context)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine
#query = "what is the common stock balance as of Balance as of March 31, 2022?"
query = "what is the letter of credit As of December 31, 2021 "
response = query_engine.query(query)
print(response)