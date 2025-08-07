import unittest
import pandas as pd
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Path setup
current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Debug imports
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

try:
    from llms.llms import LLMs  # Assumes Reranker has __call__(query, passages) method
    from insert_data import load_csv_to_chromadb
    print("✓ LLMS imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise

class HuggingFaceBackendTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.llm = LLMs(type="offline", model_version="mistralai/Mistral-7B-Instruct-v0.2", engine="huggingface")
    
    def setUp(self):
        return super().setUp()

    def test_inference(self):
        # Test inference 
        try:
            print("Test 1: Guide to RAG")
            data = []
            user_query = "Giá Iphone 15 là bao nhiêu"
            source_information = """
                                    1 _id: 666baeb69793e149fe739413, url: https://hoanghamobile.com/dien-thoai-di-dong/apple-iphone-15-128gb-chinh-hang-vn-a, title: điện thoại iphone 15 (128gb) - chính hãng vn/a, product_promotion: - KM 1<br>- Giảm thêm 100.000đ khi khách hàng thanh toán bằng hình thức chuyển khoản ngân hàng khi mua iPhone 15 Series.<br>- KM 2<br>- Ưu đãi trả góp 0% qua thẻ tín dụng<br>, product_specs: Công nghệ màn hình:
                                    Màn hình Super Retina XDR, Tấm nền OLED, Dynamic Island, Màn hình HDR, Tỷ lệ tương phản 2.000.000:1 , Màn hình có dải màu rộng (P3), Haptic Touch<br> Độ phân giải:
                                    1179 x 2556, Chính: 48MP, khẩu độ ƒ/1.6, Ultra Wide: 12MP, khẩu độ ƒ/2.4, Camera trước TrueDepth 12MP, khẩu độ ƒ/1.9<br> Kích thước màn hình:        
                                    6.1 inch<br> Hệ điều hành:
                                    iOS 17<br> Vi xử lý:
                                    A16 Bionic<br> Bộ nhớ trong:
                                    128GB<br> RAM:
                                    6GB<br> Mạng di động:
                                    2G, 3G, 4G, 5G<br> Số khe SIM:
                                    SIM kép (nano-SIM và eSIM), Hỗ trợ hai eSIM<br>, current_price: 18,790,000 ₫, color_options: ['Xanh Lá', 'Xanh Dương', 'Màu Đen', 'Hồng', 'Màu Vàng']
                                """
            combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {user_query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}." 
            data.append({
                "role": "user",
                "content": combined_information
            })
            response = self.llm.generate_content(data)
            print(f"Response Test1: {response}")
            self.assertIsInstance(response, str)

            print("Test 2: Guide to Chat")
            data = []
            user_query = "Thủ đô của Việt Nam ở đâu?"
            data.append({
                "role": "user",
                "content": user_query
            })
            response = self.llm.generate_content(data)
            print(f"Response Test2: {response}")
            self.assertIsInstance(response, str)
            
            print("Test Passed")

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    def tearDown(self):
        return super().tearDown()
    
    @classmethod
    def tearDownClass(cls):
        print("Finished HuggingFace backend test")

if __name__ == "__main__":
    unittest.main()