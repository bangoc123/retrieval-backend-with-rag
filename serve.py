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


# Load environment variables from .env file
load_dotenv()
# Access the key
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
LLM_KEY = os.getenv('GEMINI_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
OPEN_AI_EMBEDDING_MODEL = os.getenv('OPEN_AI_EMBEDDING_MODEL') or 'text-embedding-3-small'

OpenAIEmbedding(OPEN_AI_KEY)

# --- Semantic Router Setup --- #

PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'

openAIEmbeding = OpenAIEmbedding(apiKey=OPEN_AI_KEY, dimensions=1024, name=OPEN_AI_EMBEDDING_MODEL)
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(openAIEmbeding, routes=[productRoute, chitchatRoute])

# --- End Semantic Router Setup --- #


# --- Set up LLMs --- #

genai.configure(api_key=LLM_KEY)
llm = genai.GenerativeModel('gemini-1.5-pro')

# --- End Set up LLMs --- #

app = Flask(__name__)
CORS(app)


# Initialize RAG
rag = RAG(
    mongodbUri=MONGODB_URI,
    dbName=DB_NAME,
    dbCollection=DB_COLLECTION,
    embeddingName='keepitreal/vietnamese-sbert',
    llm=llm,
)

def process_query(query):
    return query.lower()

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = process_query(data.get('content'))

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # get last message
    
    guidedRoute = semanticRouter.guide(query)[1]

    if guidedRoute == PRODUCT_ROUTE_NAME:
        # Guide to RAG system
        source_information = rag.enhance_prompt(query).replace('<br>', '\n')
        combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
        response = rag.generate_content(combined_information)
    else:
        # Guide to LLMs
        response = llm.generate_content(query)
    
    return jsonify({
        'content': response.text,
        'role': 'system'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
