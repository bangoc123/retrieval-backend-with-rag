# RAG Best Practice on Vietnamese

### Evaluation Framework

![Evaluation Framework](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/b905f980-57e1-11f0-84e4-0f8a7a754383-Screenshot_2025_07_03_144533.png)

🔗 [Details](https://protonx.coursemind.io/courses/684d3a8bb224570012d03b22/topics/684f965f904b370012b6a553)

---

### Slides

📑 [Slide](https://drive.google.com/file/d/1HxTEHp4lV6i4C5F2ummqjFLXDnzPkaPX/view?usp=sharing)

![Slide Image](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/dd582970-3da7-11ef-bf69-71eafa46c86b-Screen_Shot_2024_07_09_at_11.00.59.png)

---

### Demo

▶️ [Video Demo](https://youtu.be/zzN3FEuzVt4)

---

#### Chatbot Architecture

![Architecture](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/f8928780-3da7-11ef-a9c5-539ef4fa11ba-Screen_Shot_2024_07_09_at_11.01.45.png)

#### The chatbot can retrieve your product data and answer related questions:

![Product Q\&A](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/0e6926b0-2a05-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.04.23.png)

#### It can also handle casual conversations using Semantic Router:

![Chitchat](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/3efb6050-36ca-11ef-a9c5-539ef4fa11ba-Screen_Shot_2024_06_30_at_16.57.11.png)

---

### Setup

#### 1. Installation

Requires Python >= 3.9

```bash
pip install -r requirements.txt
```

#### 2. Environment Variables

Create a `.env` file and add the following:

```env
MONGODB_URI=
EMBEDDING_MODEL=
DB_NAME=
DB_COLLECTION=
GEMINI_KEY=
```

* `MONGODB_URI`: Your MongoDB Atlas connection string.
* `EMBEDDING_MODEL`: The name of the embedding model.
* `DB_NAME`: Database name in MongoDB.
* `DB_COLLECTION`: Collection name within the database.
* `GEMINI_KEY`: API key for Gemini.

---

#### 3. Data Preparation

Prepare your data as shown below:

![Data Format](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/36777950-2a04-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.10.39.png)

> Make sure to create a **Vector Search Index** in MongoDB Atlas.
> 🎥 [Watch how to do it](https://youtu.be/jZ4hN4evesg?si=ZbXAMlQ4dsBQU_oI&t=2076)

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

```bash
python serve.py
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
