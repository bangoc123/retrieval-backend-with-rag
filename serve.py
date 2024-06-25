from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai
from flask_cors import CORS
from rag.core import RAG

# Load environment variables from .env file
load_dotenv()
# Access the key
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
LLM_KEY = os.getenv('GEMINI_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
app = Flask(__name__)
CORS(app)


# Initialize RAG
rag = RAG(
    mongodbUri=MONGODB_URI,
    dbName=DB_NAME,
    dbCollection=DB_COLLECTION,
    llmName='gemini-1.5-pro',
    embeddingName='keepitreal/vietnamese-sbert',
    llmApiKey=LLM_KEY,
)

def process_query(query):
    return query.lower()

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = process_query(data.get('content'))

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Retrieve data from vector database
    source_information = rag.enhance_prompt(query).replace('<br>', '\n')
    combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."

    response = rag.generate_content(combined_information)
    
    return jsonify({
        'content': response.text,
        'role': 'system'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
