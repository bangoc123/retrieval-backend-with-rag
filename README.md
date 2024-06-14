# Vietnamese Retrieval Backend: RAG + MongoDB + Gemini 1.5 Pro


This demo will be presented at Google I/O Extended HCMC 2024.

[Slide](https://drive.google.com/file/d/1S4yVEKePiGQpvynEDuYHjY9hCyfkXikZ/view?usp=sharing)


![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/3a194420-2933-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_11_at_10.08.04_1200x990.png)

![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/ea30c3b0-2933-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_13_at_10.20.36.png)

![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/0e6926b0-2a05-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.04.23.png)

### Set up

#### 1. Installation
This code requires Python >= 3.9.


```
pip install -r requirements.txt
```

#### 2. Environment Variables

Create a file named .env and add the following lines, replacing placeholders with your actual values:

```
MONGODB_URI=
EMBEDDING_MODEL=
DB_NAME=
DB_COLLECTION=
GEMINI_KEY=
```

- MONGODB_URI: URI of your MongoDB Atlas instance.
- EMBEDDING_MODEL: Name of the embedding model you're using for text embedding.
- DB_NAME: Name of the database in your MongoDB Atlas.
- DB_COLLECTION: Name of the collection within the database.
- GEMINI_KEY: Your key to access the Gemini API.

#### 3. Data

Prepare your data following the format below:

![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/36777950-2a04-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.10.39.png)

For this project, we are using MongoDB Atlas for Vector Search.

Make sure you create a Vector Search Index. [Follow this video](https://youtu.be/jZ4hN4evesg?si=ZbXAMlQ4dsBQU_oI&t=2076).

#### 4. Edit your Prompt in serve.py

In the serve.py file, you can see that we used the prompt like this. This prompt was enhanced by adding information about your products to it.

```
f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
```

- query: Query from the user.
- source_information: Information we get from our database.

The full prompt will look like this:

```
Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: Samsung Galaxy Z Fold4 512GB
Trả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: 
 1) Tên: điện thoại samsung galaxy z fold5 12gb/512gb - chính hãng, Giá: 30,990,000 ₫, Ưu đãi: - KM 1
- Tặng gói Samsung care+ 6 tháng
- KM 2
- Trả góp tới 06 tháng không lãi suất, trả trước 0 đồng với Samsung Finance+.

 2) Tên: điện thoại ai - samsung galaxy s24 - 8gb/512gb - chính hãng, Giá: 25,490,000 ₫, Ưu đãi: - KM 1
- Trả góp tới 06 tháng không lãi suất, trả trước 0 đồng với Samsung Finance+.
- KM 2
- Giảm thêm 1.000.000đ cho khách hàng thân thiết (Chi tiết LH 1900 ****)

 3) Tên: điện thoại samsung galaxy s23 ultra 12gb/512gb - chính hãng, Giá: 26,490,000 ₫, Ưu đãi: - KM 1
- Trả góp tới 06 tháng không lãi suất, trả trước 0 đồng với Samsung Finance+.

```

The prompt is then fed to LLMs.

#### 5. Run server

```
python serve.py
```

#### 6. Testing API

![](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/709ea4a0-298c-11ef-8393-319a26aa84a3-Screen_Shot_2024_06_13_at_20.54.17.png)

Testing on web-app. [Link](https://github.com/bangoc123/protonx-ai-app-UI)

![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/0e6926b0-2a05-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_14_at_11.04.23.png)