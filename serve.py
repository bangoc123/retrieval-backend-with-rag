from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai
from flask_cors import CORS
from rag.core import RAG
from embeddings import OpenAIEmbedding
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
import google.generativeai as genai
import openai
from reflection import Reflection
from re_rank import Reranker
from llms.llms import LLMs
import argparse
import warnings

# Load environment variables from .env file
load_dotenv()

# Add custom exception
class URLNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} base URL in .env")

class APINotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} API key in .env")

def main(args):

    # --- Semantic Router Setup --- #

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'

    openAIEmbeding = OpenAIEmbedding(apiKey=os.getenv('OPEN_AI_KEY'), dimensions=1024, name=args.openai_embedding)
    productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
    chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
    semanticRouter = SemanticRouter(openAIEmbeding, routes=[productRoute, chitchatRoute])

    # --- End Semantic Router Setup --- #

    # --- Set up LLMs --- #

    if args.mode == "online" and args.model_name == "gemini":
        MODEL_API_KEY = os.getenv('GEMINI_API_KEY', None)  
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('GEMINI')

    elif args.mode == "online" and args.model_name == "openai":
        MODEL_API_KEY = os.getenv('OPENAI_API_KEY')
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('OPENAI')

    elif args.mode == "online" and args.model_name == "together":
        MODEL_API_KEY = os.getenv('TOGETHER_API_KEY', None)
        MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", None)

        if not MODEL_API_KEY:
            raise APINotFoundError('TogetherAI')
        if not MODEL_BASE_URL:
            raise URLNotFoundError('TogetherAI')

    elif args.mode == "offline" and (args.model_engine == "ollama" or args.model_engine == "vllm"):
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError(f"{args.model_engine}")

    elif args.mode == "offline" and args.model_engine == None:
        MODEL_API_KEY = None
        MODEL_BASE_URL = None

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
        rag = RAG(
            type='qdrant',
            qdrant_api=os.getenv('QDRANT_API'),
            qdrant_url=os.getenv('QDRANT_URL'),
            embeddingName=args.embedding_model,
            llm=llm,
        )
    elif args.db == 'mongodb':
        rag = RAG(
            mongodbUri=os.getenv('MONGODB_URI'),
            dbName=os.getenv('DB_NAME'),
            dbCollection=os.getenv('DB_COLLECTION'),
            embeddingName=args.embedding_model,
            llm=llm,
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
                source_information += f"{i+1} {ranked_passages}\n"

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
            'parts': [
                {
                'text': response
                }
            ],
            'role': 'model'
            })
    app.run(host='0.0.0.0', port=5002, debug=True)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Arguments for serve.py")

    parser.add_argument('--mode', type=str, choices=['online', 'offline'], default='online', help='Choose either online or offline mode system')
    parser.add_argument('--model_name', type=str, default='gemini', help='Define name of LLM model to use')
    parser.add_argument('--model_engine', type=str, default='ollama', help='Define model engine of LLM model (Optional)')
    parser.add_argument('--model_version', type=str, default='gemini-2.5-flash-lite', help='Define model version of LLM model (Optional)')
    parser.add_argument('--db', type=str, choices=['qdrant', 'mongodb'], default='qdrant', help='Choose type of vector store database')
    parser.add_argument('--embedding_model', type=str, default='Alibaba-NLP/gte-multilingual-base', help='Declare what embedding model to use for RAG')
    parser.add_argument('--reranker', type=str, default='BAAI/bge-reranker-v2-m3', help='Declare name of CrossEncoder ReRanker')
    parser.add_argument('--openai_embedding', type=str, default='text-embedding-3-small', help='Declare OpenAI Embedding model')

    args = parser.parse_args()
    main(args)