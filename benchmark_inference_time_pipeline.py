import time
import pandas as pd
from dotenv import load_dotenv
import os
import google.generativeai as genai
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
from insert_data import load_csv_to_chromadb

# Load environment variables from .env file
load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

class DataNotFoundError(Exception):
    def __init__(self):
        super().__init__(f"Please make sure you have valid CSV file in folder data")

def map_environment_variables(args):
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

    elif args.mode == "offline" and args.model_engine == "huggingface":
        MODEL_API_KEY = None
        MODEL_BASE_URL = None
    
    return MODEL_API_KEY, MODEL_BASE_URL

def setup_db_and_rag(args, llm):
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
        def csv_exists(folder_path: str) -> list:
            """
            Check if any CSV file exists in the given folder and return their file path.

            Args:
                folder_path (str): Path to the folder (e.g., "./data")

            Returns:
                list: list of csv files.    
            """
            if not os.path.isdir(folder_path):
                return []

            csv_paths = [
                os.path.abspath(os.path.join(folder_path, file))
                for file in os.listdir(folder_path)
                if file.lower().endswith(".csv") and os.path.isfile(os.path.join(folder_path, file))
            ]
            return csv_paths
        
        if not chromadb_collection_exists(collection_name=args.embedding_model.split('/')[-1]):
            csv_files = csv_exists(folder_path="data")
            if len(csv_files) == 0:
                raise DataNotFoundError
            else:
                print(f"The collection {args.embedding_model.split('/')[-1]} does not exist.\n")
                print("Starting to create new collection. Please make sure you have a valid CSV file in data folder.\n")
                print(f"Detected {len(csv_files)} csv files.\n")
                for i in  range(len(csv_files)):              
                    load_csv_to_chromadb(csv_path=csv_files[i], persist_dir="./chroma_db", model_name=args.embedding_model)
                    print(f"Processed {i+1} files.\n")  
                print("The data insert process is complete.")

        rag = RAG(
            type='chromadb',
            embeddingName=args.embedding_model,
            llm=llm
        )
    return rag 

def setup_pipeline(args):

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

    MODEL_API_KEY, MODEL_BASE_URL = map_environment_variables(args)

    llm = LLMs(type=args.mode, model_version=args.model_version, model_name=args.model_name, engine=args.model_engine, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    # --- End Set up LLMs --- #

    # --- Relection Setup --- #

    reflection = Reflection(llm=llm)

    # --- End Reflection Setup --- #

    # Initialize RAG
    rag = setup_db_and_rag(args, llm)

    # Initialize ReRanker 
    reranker = Reranker(model_name=args.reranker)

    return semanticRouter, llm, reflection, rag, reranker
def benchmark_from_csv(args):

    print("Load benchmark data from csv file.\n")
    try:
        df = pd.read_csv(args.csv_path)
    except:
        raise DataNotFoundError

    # Setup pipeline RAG
    semanticRouter, llm, reflection, rag, reranker = setup_pipeline(args)

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'

    results = []

    total_time = 0

    for idx, row in df.iterrows():
        data = {"role": "user", "content": row["query"]}
        query_id = row["_id"]


        print(f"\n[Query ID: {query_id}] Running inference...")

        start_time = time.time()

        # Reflection phase 
        reflected_query = reflection([data])
        query = reflected_query
        reflection_time = time.time()
        elapsed_reflection_time = reflection_time - start_time 

        # Routing time 
        guidedRoute = semanticRouter.guide(query)[1]
        routing_time = time.time()
        elapsed_routing_time = routing_time - reflection_time

        if guidedRoute == PRODUCT_ROUTE_NAME:
            # Guide to RAG system
            print("Guide to RAGs")

            # Take relevant documents from RAG system
            passages = [passage['combined_information'] for passage in rag.vector_search(query)]
            rag_time = time.time()
            elapsed_rag_time = rag_time - routing_time 

            # Rerannk retrieved documents
            scores, ranked_passages = reranker(query, passages)
            rerank_time = time.time()
            elapsed_rerank_time = rerank_time - rag_time 

            source_information = ""
            for i in range(len(ranked_passages)):
                source_information += f"{i+1} {ranked_passages[i]}\n"

            combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
            data = []
            data.append({
                "role": "user",
                "content": combined_information
            })
            response = rag.generate_content(data) 
            elapsed_llm_time = time.time() - rerank_time 
            total_response_time = time.time() - start_time
        else: 
            # Guide to LLMs
            print("Guide to LLMs")
            response = llm.generate_content([data])
            elapsed_llm_time = time.time() - routing_time 
            total_response_time = time.time() - start_time
            elapsed_rag_time = None 
            elapsed_rerank_time = None 

        print(f"Inference time: {total_response_time:.4f} seconds")

        results.append({
            "id": query_id,
            "query": row["query"],
            "response": response,
            "reflection_time": elapsed_reflection_time,
            "routing_time": elapsed_routing_time,
            "rag_time": elapsed_rag_time,
            "rerank_time": elapsed_rerank_time,
            "llm_time": elapsed_llm_time,
            "total_response_time": total_response_time
        })

        total_time += total_response_time
    
        # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"benchmark_results_{args.model_version.split('/')[-1]}_{args.model_engine}_pipeline_rag.csv", index=False)
    print(f"\n✅ Benchmarking completed. Results saved to benchmark_inference_time_{args.model_version.split('/')[-1]}_{args.model_engine}_pipeline_rag.csv.")
    print(f"Total time to run inference for {len(df)} queries in dataset: {total_time}")
    print(f"Average time to run inference for {len(df)} queries in dataset: {(total_time/len(df)):.4f}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Arguments for serve.py")

    parser.add_argument('-c','--csv_path', type=str, required=True, help='Paste your csv file absolute path.')
    parser.add_argument('-m','--mode', type=str, choices=['online', 'offline'], default='offline', help='Choose either online or offline mode system')
    parser.add_argument('-n','--model_name', type=str, default='gemini', help='Define name of LLM model to use')
    parser.add_argument('-e','--model_engine', type=str, default='huggingface', help='Define model engine of LLM model')
    parser.add_argument('-v','--model_version', type=str, required=True, help='Define model version of LLM model')

    parser.add_argument('--db', type=str, choices=['qdrant', 'mongodb', 'chromadb'], default='chromadb', help='Choose type of vector store database')
    parser.add_argument('--embedding_model', type=str, default='Alibaba-NLP/gte-multilingual-base', help='Declare what embedding model to use for RAG')
    parser.add_argument('--reranker', type=str, default='Alibaba-NLP/gte-multilingual-reranker-base', help='Declare name of CrossEncoder ReRanker')

    args = parser.parse_args()
    benchmark_from_csv(args)