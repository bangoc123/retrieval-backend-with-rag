from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import pymongo
import google.generativeai as genai
from flask_cors import CORS

import pathlib
import textwrap

from IPython.display import display
from IPython.display import Markdown

# Load environment variables from .env file
load_dotenv()

# Access the key
MONGODB_URI = os.getenv('MONGODB_URI')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
GEMINI_KEY = os.getenv('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

client = pymongo.MongoClient(MONGODB_URI)
db = client[DB_NAME] 
collection = db[DB_COLLECTION]



app = Flask(__name__)
CORS(app)

from sentence_transformers import SentenceTransformer

# https://huggingface.co/thenlper/gte-large
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def vector_search(user_query, collection, limit=4):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": limit  # Return top k matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "title": 1,  # Include the title field
            "product_specs": 1,
            "color_options": 1,
            "current_price": 1,
            "product_promotion": 1,
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            }
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    # Execute the search
    results = collection.aggregate(pipeline)

    return list(results)

def get_search_result(query, collection):

    get_knowledge = vector_search(query, collection, 4)

    search_result = ""
    for i, result in enumerate(get_knowledge):
        print(result)
        # search_result += f"Title: {result.get('title', 'N/A')}, Context: {result.get('body', 'N/A')}\n"
        search_result = f"{i + 1}) Tên: {result.get('title', 'N/A')}"
        
        if result.get('current_price'):
            search_result += f", Giá: {result.get('current_price')}"
        else:
            # Mock up data
            # Retrieval model pricing from the internet.
            search_result += f", Giá: 1 000 000 VND"
        
        if result.get('product_promotion'):
            search_result += f", Ưu đãi: {result.get('product_promotion')}"
    
    return search_result

def get_embedding(text):
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('content')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Retrieve data from vector database

    source_information = get_search_result(query, collection)
    combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."

    chat = model.start_chat(history=[])

    response = chat.send_message(combined_information)

    for message in chat.history:
        display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))
    

    return jsonify({'response': response.text})


if __name__ == '__main__':
    app.run(debug=True)
