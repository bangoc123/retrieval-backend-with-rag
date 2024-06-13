from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import pymongo
import google.generativeai as genai


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


app = Flask(__name__)
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
            "body": 1,  # Include the plot field
            "title": 1,  # Include the title field
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

    get_knowledge = vector_search(query, collection, 1)

    search_result = ""
    for i, result in enumerate(get_knowledge):
        search_result += f"Title: {result.get('title', 'N/A')}, Context: {result.get('body', 'N/A')}\n"
        # search_result += f"{i + 1} Tên: {result.get('title', 'N/A')}"

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
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Example response based on the query
    clean_query = process_query(query)

    # Retrieve data from vector database

    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DB_NAME] 
    collection = db[DB_COLLECTION]

    source_information = get_search_result(query, collection)
    combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây:\n{source_information}."
    response = None
    try:
        response = model.generate_content(combined_information)
    except Exception as e:
        print('Error', e)
    return jsonify({'response': response.text})

def process_query(query):
    return query

if __name__ == '__main__':
    app.run(debug=True)
