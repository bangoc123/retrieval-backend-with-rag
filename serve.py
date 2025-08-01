from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai
from flask_cors import CORS
from rag.core import RAG
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
import google.generativeai as genai
import openai
from reflection import Reflection
from re_rank import Reranker
from llms.llms import LLMs
import argparse
import warnings
from insert_data import load_csv_to_chromadb

# Load environment variables from .env file
load_dotenv()

# Add custom exception
class URLNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class APINotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class ValueNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

def main(args):

    # --- Semantic Router Setup --- #

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'

    sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=EmbeddingConfig(name=args.embedding_model))
    productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
    chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
    semanticRouter = SemanticRouter(sentenceTransformerEmbedding, routes=[productRoute, chitchatRoute])
    
    # --- End Semantic Router Setup --- #

    # --- Set up LLMs --- #

    if args.mode == "online" and args.model_name == "gemini":
        MODEL_API_KEY = os.getenv('GEMINI_API_KEY', None)  
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('GEMINI_API_KEY')

    elif args.mode == "online" and args.model_name == "openai":
        MODEL_API_KEY = os.getenv('OPENAI_API_KEY')
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('OPENAI_API_KEY')

    elif args.mode == "online" and args.model_name == "together":
        MODEL_API_KEY = os.getenv('TOGETHER_API_KEY', None)
        MODEL_BASE_URL = os.getenv("TOGETHER_BASE_URL", None)

        if not MODEL_API_KEY:
            raise APINotFoundError('TOGETHER_API_KEY')
        if not MODEL_BASE_URL:
            raise URLNotFoundError('TOGETHER_BASE_URL')

    elif args.mode == "offline" and args.model_engine == "ollama":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("OLLAMA_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("OLLAMA_BASE_URL")
    
    elif args.mode == "offline" and args.model_engine == "vllm":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("VLLM_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("VLLM_BASE_URL")

    elif args.mode == "offline" and args.model_engine == None:
        MODEL_API_KEY = None
        MODEL_BASE_URL = None
        if not MODEL_BASE_URL:
            raise URLNotFoundError("VLLM_BASE_URL or OLLAMA_BASE_URL")

    llm = LLMs(type=args.mode, model_version=args.model_version, model_name=args.model_name, engine=args.model_engine, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    # --- End Set up LLMs --- #

    # --- Relection Setup --- #

    gpt = openai.OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    reflection = Reflection(llm=gpt)

    # --- End Reflection Setup --- #

    app = Flask(__name__)
    CORS(app)

    # Initialize RAG
    if args.db == 'qdrant':
        QDRANT_API = os.getenv('QDRANT_API', None)
        QDRANT_URL = os.getenv('QDRANT_URL', None)
        if not QDRANT_API:
            raise APINotFoundError("QDRANT_API")
        if not QDRANT_URL:
            raise URLNotFoundError("QDRANT_URL")
        
        rag = RAG(
            type='qdrant',
            qdrant_api=QDRANT_API,
            qdrant_url=QDRANT_URL,
            embeddingName=args.embedding_model,
            llm=llm,
        )

    elif args.db == 'mongodb':
        MONGODB_URI = os.getenv('MONGODB_URI')
        MONGODB_NAME = os.getenv('MONGODB_NAME')
        MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')
        if not MONGODB_URI:
            raise URLNotFoundError("MONGODB_URI")
        if (not MONGODB_NAME) or (not MONGODB_COLLECTION):
            raise ValueNotFoundError(f"MONGODB_NAME and MONGODB_COLLECTION")
        
        rag = RAG(
            type='mongodb',
            mongodbUri=MONGODB_URI,
            dbName=MONGODB_NAME,
            dbCollection=MONGODB_COLLECTION,
            embeddingName=args.embedding_model,
            llm=llm,
        )
    else:

        def chromadb_collection_exists(collection_name: str, persist_dir: str = "./chroma_db") -> bool:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=persist_dir)
                collections = client.list_collections()
                return any(col.name == collection_name for col in collections)
            except Exception as e:
                print(f"Error checking ChromaDB collection: {e}")
                return False

        if not chromadb_collection_exists(collection_name=args.embedding_model.split('/')[-1]):
            print(f"The collection {args.embedding_model.split('/')[-1]} does not exist.\n")
            print("Starting to create new collection. Please make sure you have a valid CSV file in ./data.\n")
            load_csv_to_chromadb(csv_path="data/hoanghamobile.csv",persist_dir="./chroma_db", model_name=args.embedding_model)
            print("The data insert process is complete.")

        rag = RAG(
            type='chromadb',
            embeddingName=args.embedding_model,
            llm=llm
        )
    # Initialize ReRanker
    reranker = Reranker(model_name=args.reranker)

    def process_query(query):
        return query.lower()

    @app.route('/api/search', methods=['POST'])
    def handle_query():
        data = list(request.get_json())

        reflected_query = reflection(data)
        query = reflected_query

        guidedRoute = semanticRouter.guide(query)[1]

        if guidedRoute == PRODUCT_ROUTE_NAME:
            # Guide to RAG system
            print("Guide to RAGs")

            # Take relevant documents from RAG system
            passages = [passage['combined_information'] for passage in rag.vector_search(query)]
            
            # Rerannk retrieved documents
            scores, ranked_passages = reranker(query, passages)
            source_information = ""
            for i in range(len(ranked_passages)):
                source_information += f"{i+1} {ranked_passages[i]}\n"

            combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
            data.append({
                "role": "user",
                "content": combined_information
            })
            response = rag.generate_content(data)
        else:
            # Guide to LLMs
            print("Guide to LLMs")
            response = llm.generate_content(data)
        
        return jsonify({
            'content': response,
            'role': 'assistant'
            })
    app.run(host='0.0.0.0', port=5002, debug=True)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Arguments for serve.py")

    model_group = parser.add_argument_group("Model Option")
    model_group.add_argument('-m','--mode', type=str, choices=['online', 'offline'], default='online', help='Choose either online or offline mode system')
    model_group.add_argument('-n','--model_name', type=str, default='gemini', help='Define name of LLM model to use')
    model_group.add_argument('-e','--model_engine', type=str, default='ollama', help='Define model engine of LLM model (Optional)')
    model_group.add_argument('-v','--model_version', type=str, required=True, help='Define model version of LLM model (Optional)')

    feature_group = parser.add_argument_group("Feature Option")
    feature_group.add_argument('--db', type=str, choices=['qdrant', 'mongodb', 'chromadb'], default='qdrant', help='Choose type of vector store database')
    feature_group.add_argument('--embedding_model', type=str, default='Alibaba-NLP/gte-multilingual-base', help='Declare what embedding model to use for RAG')
    feature_group.add_argument('--reranker', type=str, default='Alibaba-NLP/gte-multilingual-reranker-base', help='Declare name of CrossEncoder ReRanker')

    args = parser.parse_args()
    main(args)