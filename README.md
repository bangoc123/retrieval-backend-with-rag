# Vietnamese Retrieval Backend: RAG + MongoDB + Gemini 1.5 Pro


This demo will be presented at Google I/O Extended HCMC 2024.

[Slide](https://drive.google.com/file/d/1S4yVEKePiGQpvynEDuYHjY9hCyfkXikZ/view?usp=sharing)


![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/3a194420-2933-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_11_at_10.08.04_1200x990.png)

![](https://storage.googleapis.com/mle-courses-prod/users/61b869ca9c3c5e00292bb42d/private-files/ea30c3b0-2933-11ef-bde4-3b0f2c27b69f-Screen_Shot_2024_06_13_at_10.20.36.png)


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

#### 3. Run server

```
python server.py
```

#### 4. Testing API

![](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/709ea4a0-298c-11ef-8393-319a26aa84a3-Screen_Shot_2024_06_13_at_20.54.17.png)