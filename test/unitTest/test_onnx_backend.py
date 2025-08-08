import unittest
import os
import sys

# Path setup
current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

try:
    from llms.llms import LLMs
    print("✓ LLMS imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise


class ONNXLLMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llm = LLMs(
            engine="onnx",
            model_version="onnx-community/TinyLLama-v0-ONNX",
            type="offline",
            local_dir="./test_onnx_models",
            max_tokens=100
        )

    def test_basic_inference(self):
        print("--- Running basic ONNX inference test ---")
        response = self.llm.generate_content([{"role": "user", "content": "Hello, how are you?"}])
        print("Model response:", response)
        self.assertIsInstance(response, str, "Model output is not a string")


if __name__ == "__main__":
    unittest.main()
