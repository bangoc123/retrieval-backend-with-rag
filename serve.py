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

# Load environment variables from .env file
load_dotenv()
# Access the key
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
LLM_KEY = os.getenv('GEMINI_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')
OPEN_AI_EMBEDDING_MODEL = os.getenv('OPEN_AI_EMBEDDING_MODEL') or 'text-embedding-3-small'
QDRANT_API = os.getenv('QDRANT_API')
QDRANT_URL = os.getenv('QDRANT_URL')
TOGETHER_AI = os.getenv('TOGETHER_AI')

OpenAIEmbedding(OPEN_AI_KEY)

MODEL_TYPE = os.getenv("MODEL_TYPE", "offline")
MODEL_NAME = os.getenv("MODEL_NAME", "NousResearch/Meta-Llama-3-8B-Instruct")
MODEL_ENGINE = os.getenv("MODEL_ENGINE", None)
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if MODEL_TYPE == "online" and MODEL_NAME == "gemini":
    MODEL_API_KEY = LLM_KEY
elif MODEL_TYPE == "online" and MODEL_NAME == "openai":
    MODEL_API_KEY = OPEN_AI_KEY
elif MODEL_TYPE == "online" and MODEL_NAME == "together":
    MODEL_API_KEY = TOGETHER_AI
else:
    MODEL_API_KEY = None

# --- embedding setup --- # 


# --- Semantic Router Setup --- #

PRODUCT_ROUTE_NAME = 'products' # define products route name
CHITCHAT_ROUTE_NAME = 'chitchat'

openAIEmbeding = OpenAIEmbedding(apiKey=OPEN_AI_KEY, dimensions=1024, name=OPEN_AI_EMBEDDING_MODEL)
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(openAIEmbeding, routes=[productRoute, chitchatRoute])

# --- End Semantic Router Setup --- #


# --- Set up LLMs --- #

genai.configure(api_key=LLM_KEY)
# llm = genai.GenerativeModel('gemini-1.5-pro')
llm = LLMs(type=MODEL_TYPE, model_name=MODEL_NAME, engine=MODEL_ENGINE, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY, model_version=MODEL_VERSION)


# --- End Set up LLMs --- #

# --- Relection Setup --- #

gpt = openai.OpenAI(api_key=OPEN_AI_KEY)
reflection = Reflection(llm=gpt)

# --- End Reflection Setup --- #

app = Flask(__name__)
CORS(app)


# Initialize RAG
rag = RAG(
    # mongodbUri=MONGODB_URI,
    # dbName=DB_NAME,
    # dbCollection=DB_COLLECTION,
    type='qdrant',
    qdrant_api=QDRANT_API,
    qdrant_url=QDRANT_URL,
    embeddingName='Alibaba-NLP/gte-multilingual-base',
    llm=llm,
)
reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
def process_query(query):
    return query.lower()

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = list(request.get_json())

    reflected_query = reflection(data)
    query = reflected_query
    # query = data[-1]["parts"][0]["text"]
    # query = process_query(query)

    if not query:
        return jsonify({'error': 'No query provided'}), 400
     
    guidedRoute = semanticRouter.guide(query)[1]

    if guidedRoute == PRODUCT_ROUTE_NAME:
        # Decide to get new info or use previous info
        # Guide to RAG system
        print("Guide to RAGs")

        # reflected_query = reflection(data)
        # query = reflected_query

        # source_information = rag.enhance_prompt(query).replace('<br>', '\n')
        passages = [passage['combined_information'] for passage in rag.vector_search(query)]
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

    # print('====data', data)
    
    return jsonify({
        'parts': [
            {
            'text': response,
            }
        ],
        'role': 'model'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
