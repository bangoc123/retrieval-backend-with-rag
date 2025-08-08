import requests
import json

url = "http://localhost:5002/api/search"
payload = [
    {
        "role": "user",
        "content": "tôi tên Ngọc"
    },
    {
        "content": "Cảm ơn bạn! Tôi là AI, và tôi rất sẵn sàng hỗ trợ bạn với nhiều công việc như trả lời câu hỏi, cung cấp thông tin, hoặc tạo nội dung. Bạn có muốn làm gì? 😊",
        "role": "assistant"
    },
    {
        "role": "user",
        "content": "đánh vần tên tôi"
    }
]

try:
    # Đặt timeout=None => chờ vô hạn, hoặc dùng số giây lớn như 300
    response = requests.post(url, json=payload, timeout=None)
    response.raise_for_status()
    print("Status:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), ensure_ascii=False, indent=2))
except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
