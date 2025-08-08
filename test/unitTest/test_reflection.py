import unittest
import pandas as pd
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Path setup
current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Debug imports
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

try:
    from llms.llms import LLMs
    from reflection import Reflection
    print("✓ Reflection imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise

GEMINI_API_KEY= os.getenv('GEMINI_API_KEY', None)
OPENAI_API_KEY= os.getenv('OPENAI_API_KEY', None)
OLLAMA_BASE_URL= os.getenv('OLLAMA_BASE_URL', None)
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', None)
TOGETHER_BASE_URL= os.getenv('TOGETHER_BASE_URL', None)
TOGETHER_API_KEY= os.getenv('TOGETHER_API_KEY', None)

class ReflectionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        return

    def setUp(self):
        return

    def test_reflection(self):
        """Test reflection with different LLMs"""
        tests = [{'mode':'online',
                'model_version':'gemini-2.5-flash',
                'model_name':'gemini',
                'model_engine':None,
                'base_url':None,
                'api_key':GEMINI_API_KEY},

                {'mode':'online',
                'model_version':'gpt-4o',
                'model_name':'openai',
                'model_engine':None,
                'base_url':None,
                'api_key':OPENAI_API_KEY},

                {'mode':'online',
                'model_version':'mistralai/Mistral-7B-Instruct-v0.2',
                'model_name':'together',
                'model_engine':None,
                'base_url': TOGETHER_BASE_URL,
                'api_key': TOGETHER_API_KEY},

                ]
        k = 0
        for i in range(len(tests)):
            print(f"Test {i+1}:")
            print(f"*Model engine: {tests[i]['model_engine']}")
            print(f"*Model name: {tests[i]['model_version']}")
            try:
                llm = LLMs(type=tests[i]['mode'], 
                        model_version=tests[i]['model_version'], 
                        model_name=tests[i]['model_name'], 
                        engine=tests[i]['model_engine'],
                        api_key=tests[i]['api_key'],
                        base_url=tests[i]['base_url'])
                print(tests[i]['mode'])
                reflection = Reflection(llm=llm)

                data = {
                    "role": "user",
                    "content": "Chào bạn, tôi đang tìm một chiếc điện thoại để dùng, hãy cho tôi xin giá thành hiện tại của Iphone 15 trong cửa hàng bạn."
                }
                reflected_query = reflection([data])
                print(f"Response reflection Test {i+1}: {reflected_query}")
                print(type(reflected_query))
                if isinstance(reflected_query, str):
                    print(f"Result: passed.")
                    k += 1
                else:
                    print(f"Result: failed.")
            except Exception as e:
                print(f"Result: failed")
                print(f"Error: {e}")
        print(f"Passed {k}/{len(tests)} tests.")

    @classmethod
    def tearDownClass(cls):
        print("Finished Reflection Test\n")

if __name__ == "__main__":
    unittest.main() 