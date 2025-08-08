# RAG Best Practice on Vietnamese

### Evaluation Framework

![Evaluation Framework](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/b905f980-57e1-11f0-84e4-0f8a7a754383-Screenshot_2025_07_03_144533.png)

Retrieval Benchmarks

![Retrieval Benchmarks](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/ffc9c650-5181-11f0-9c7a-bfa66305902b-output__2_.png)

ReRank Benchmarks

![Rerank Benchmarks](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/b5d82990-6c63-11f0-8395-0df7dffeba85-average_ndcg_reranker_models_horizontal_sorted_1.png)

LLM Answer Benchmarks

![LLM Answer Benchmarks](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/7920b290-5315-11f0-84e4-0f8a7a754383-output.png)

🔗 [Details](https://protonx.coursemind.io/courses/684d3a8bb224570012d03b22/topics/684f965f904b370012b6a553)

Groundedness Benchmarks

Groundedness measures how well a model’s responses are supported by the provided context or reliable sources, ensuring accuracy and reducing hallucinations.

![G-b](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/bc772d80-71e2-11f0-ae39-5b0a81678d54-Screenshot_2025-08-05_165745.png)

View details about this benchmarks [here](https://protonx.coursemind.io/courses/684d3a8bb224570012d03b22/topics/684f965f904b370012b6a553?activeAId=6866366ff7c7a147467c6140)


---

### Slides

📑 [Slide](https://drive.google.com/file/d/1HxTEHp4lV6i4C5F2ummqjFLXDnzPkaPX/view?usp=sharing)

![Slide Image](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/dd582970-3da7-11ef-bf69-71eafa46c86b-Screen_Shot_2024_07_09_at_11.00.59.png)

---

### Demo

▶️ [Video Demo](https://youtu.be/zzN3FEuzVt4)

---

#### Chatbot Architecture

![Architecture](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/dbbba200-6e88-11f0-9258-839289629457-Screenshot_2025-08-01_103702.png)



#### The chatbot can retrieve your product data and answer related questions:

![Product Q\&A](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/0e6926b0-2a05-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.04.23.png)

#### It can also handle casual conversations using Semantic Router:

![Chitchat](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/3efb6050-36ca-11ef-a9c5-539ef4fa11ba-Screen_Shot_2024_06_30_at_16.57.11.png)

### Opensource client via Docker


![](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/221bcae0-6e93-11f0-9258-839289629457-Screenshot_2025-08-01_115038.png)

We've open-sourced a chatbot client via Docker.

```bash
# Pull the Docker image
docker pull protonx/protonx-open-source:protonx-chat-client-v01
```

```bash
# Run the Docker image
docker run -p 3002:3000 -e RAG_BACKEND_URL=${YOUR_BACKEND_URL} protonx/protonx-open-source:protonx-chat-client-v01
```

If your local backend URL is `http://localhost:5002/api/search`, the command will be:

```bash
docker run -p 3002:3000 -e RAG_BACKEND_URL="http://localhost:5002/api/search" protonx/protonx-open-source:protonx-chat-client-v01
```

The backend should accept a `POST` request with the following request body:

```json
[
  {
    "role": "user",
    "content": "Tôi đang tham khảo redmi note 13 plus",
  }
]
```

And return a response in the following format:

```json
{
  "role": "assistant",
  "content": "Xin chào! Cảm ơn bạn đã quan tâm đến sản phẩm của chúng tôi. Điện thoại Redmi Note 13 Pro+ là một lựa chọn tuyệt vời..."
}
```


### Setup

#### 1. Installation

Requires Python >= 3.12

```bash
pip install -r requirements.txt
```

#### 2. Environment Variables

Create a `.env` file and add the following:

```env

# MongoDB vector database (leave blank if not used)
MONGODB_URI=
DB_NAME=
DB_COLLECTION=

# Qdrant vector database (leave blank if not used)
QDRANT_API=
QDRANT_URL=

# Gemini LLM (leave blank if not used)
GEMINI_API_KEY=

# OpenAI LLM (leave blank if not used)
OPENAI_API_KEY=

# Together AI LLM (leave blank if not used)
TOGETHER_API_KEY=
TOGETHER_BASE_URL=

# Ollama local LLM engine (leave blank if not used)
OLLAMA_BASE_URL=

# vLLM local LLM engine (leave blank if not used)
VLLM_BASE_URL=
```
---

#### 3. Data Preparation

Prepare your data as shown below:

![Data Format](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/36777950-2a04-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.10.39.png)

> Make sure to create a **Vector Search Index** in MongoDB Atlas. 🎥 [Watch how to do it](https://youtu.be/jZ4hN4evesg?si=ZbXAMlQ4dsBQU_oI&t=2076)

> Guide for Qdrant will be updated soon

---

#### 4. Customize Your Prompt

In `serve.py`, you can customize the LLM prompt like this:

```python
f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
```

Example full prompt:

```
Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: Samsung Galaxy Z Fold4 512GB
Trả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: 
1) Tên: điện thoại samsung galaxy z fold5 12gb/512gb - chính hãng, Giá: 30,990,000 ₫, Ưu đãi:
   - KM 1: Tặng gói Samsung care+ 6 tháng
   - KM 2: Trả góp tới 06 tháng không lãi suất, trả trước 0 đồng với Samsung Finance+.

2) Tên: điện thoại ai - samsung galaxy s24 - 8gb/512gb - chính hãng, Giá: 25,490,000 ₫, Ưu đãi:
   - KM 1: Trả góp tới 06 tháng không lãi suất, trả trước 0 đồng với Samsung Finance+.
   - KM 2: Giảm thêm 1.000.000đ cho khách hàng thân thiết (Chi tiết LH 1900 ****)

3) Tên: điện thoại samsung galaxy s23 ultra 12gb/512gb - chính hãng, Giá: 26,490,000 ₫, Ưu đãi:
   - KM 1: Trả góp tới 06 tháng không lãi suất, trả trước 0 đồng với Samsung Finance+.
```

---

#### 5. Run the Server

##### Use openai model online

```bash
python serve.py --mode online --model_name openai --model_version gpt-4o
```

##### Use Gemini model in online mode

```bash
python serve.py --mode online --model_name gemini --model_version gemini-2.0-flash
```

##### Run Ollama with the local model mistralai/Mistral-7B-Instruct-v0.2

```bash
python serve.py --mode offline --model_engine ollama --model_version mistralai/Mistral-7B-Instruct-v0.2
```

##### Run HuggingFace backend with the local model mistralai/Mistral-7B-Instruct-v0.2

```bash
python serve.py --mode offline --model_engine huggingface --model_version mistralai/Mistral-7B-Instruct-v0.2
```
##### Run ONNX backend with the local model

```bash
python serve.py --mode offline --model_name TinyLLama --model_engine onnx --model_version onnx-community/TinyLLama-v0-ONNX

```

---

#### 6. Test the API

![Test API](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/709ea4a0-298c-11ef-8393-319a26aa84a3-Screen_Shot_2024_06_13_at_20.54.17.png)

Try the chatbot UI here:
🔗 [GitHub: protonx-ai-app-UI](https://github.com/bangoc123/protonx-ai-app-UI)

---

#### 7. Run Evaluation

Run **all evaluation tests**:

```bash
python -m unittest discover -s ./test/integrationTest -p "test*.py" -v
```

Run a **specific test**:

```bash
python ./test/integrationTest/llm-answer/test_bleu.py
```

Current evaluation
- Integration Test
   - LLM Answer
      - BLEU test
      - ROUGE test
- Unit Test
   - Test vector search
     - Retrieval
      - Hit@K
   - Rerank
      - nCDG
   - Test Reflection