import requests
import re
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class LocalLLMs:
    def __init__(self, engine: str, model_version: str, base_url: str = None):
        """ Initialize the LocalLLMs class 
            Args:
            engine (str): "ollama" or "vllm".
            model_version (str): name of model ("llama3","meta-llama/Llama-2-7b-chat-hf").
            base_url (str, optional): BASE URL of server engine.
        """
        self.engine = engine
        self.model_version = model_version
        self.client = None
        self.max_tokens = 4096  # Default max tokens
        if engine == "ollama":
            self.base_url = base_url 
            self._initialize_ollama_model(model_version)
        elif engine == "vllm":
            self.base_url = base_url
            self._initialize_vllm_model(model_version)
        elif engine == "huggingface":
            self._initialize_huggingface_model(model_version)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def _initialize_ollama_model(self, model_version: str):
        """Pull the specified model from the Ollama server."""
        try:
            response = requests.get(self.base_url, timeout=5)
            response.raise_for_status()
            print("Kết nối đến máy chủ Ollama thành công.")
            self.client = requests.Session()
            self._pull_ollama_model(model_version)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Không thể kết nối đến máy chủ Ollama tại {self.base_url}. Vui lòng đảm bảo Ollama đang chạy. Lỗi: {e}")

    def _pull_ollama_model(self, model_version: str):
        """Pull model from Ollama if not exist."""
        try:
            # 1. Check if the model already exists
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_exists = any(model_version in m["name"] for m in models)

            # 2. If the model does not exist, pull it
            if not model_exists:
                print(f"Model '{model_version}' chưa tồn tại. Bắt đầu tải...")
                pull_data = {"name": model_version}
                pull_response = self.client.post(f"{self.base_url}/api/pull", json=pull_data)
                pull_response.raise_for_status()
                print(f"Tải model '{model_version}' thành công.")
            else:
                print(f"Model '{model_version}' đã có sẵn.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Lỗi khi giao tiếp với API của Ollama. Lỗi: {e}")

    def _initialize_vllm_model(self, model_version: str):
        """Initialize the vLLM model with the specified name and parameters."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            matched_model = next((m for m in models if m["id"] == self.model_version), None)

            if matched_model:
                self.max_tokens = matched_model.get("max_model_len", 4096)
                print(f"Model '{self.model_version}' đã được tìm thấy với max_tokens: {self.max_tokens}.")
            else:
                print(f"Không tìm thấy model '{self.model_version}' trong danh sách model của vLLM. Dùng giá trị mặc định 4096.")

            print("Kết nối đến vLLM server thành công.")
            self.client = requests.Session()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Không thể kết nối đến vLLM tại {self.base_url}. Lỗi: {e}")
    
    def _initialize_huggingface_model(self, model_version: str):
        """Initialize the Huggingface model with the specified name and parameters"""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
                        model_version,
                        trust_remote_code=True,
                        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.client = AutoModelForCausalLM.from_pretrained(
            model_version,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code = True
        ).to(device)
        self.client.eval()

    def remove_think_blocks(self,text):
        """Remove <think> blocks and their content from text"""
        # Pattern to match <think>...</think> blocks (including multiline)
        pattern = r'<think>.*?</think>'
        # Remove the think blocks using re.sub with DOTALL flag for multiline matching
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        # Clean up any extra whitespace that might be left
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
        return cleaned_text
    
    def generate_content(self, prompt: List[Dict[str,str]]) -> str:
        """Generate content using the local LLM based on the provided prompt.
            input: prompt (str): The prompt to generate content for.
            output: str: The generated content.
        """
        if not self.client:
            raise RuntimeError("Client chưa được khởi tạo. Vui lòng kiểm tra lại cấu hình.")

        print(f"Đang tạo nội dung với engine '{self.engine}' và model '{self.model_version}'...")

        try:
            if self.engine == 'ollama':
                payload = {
                    "model": self.model_version,
                    "messages": prompt,
                    "stream": False
                }
                response = self.client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                response_data = response.json()["message"]["content"].strip()
                return self.remove_think_blocks(response_data)

            elif self.engine == 'vllm':
                payload = {
                    "model": self.model_version,
                    "messages": prompt,
                    # "max_tokens": self.max_tokens,
                    # "temperature": 0.7
                }
                response = self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                response.raise_for_status()
                response_data = response.json()["choices"][0]["message"]["content"].strip()
                return self.remove_think_blocks(response_data)
            elif self.engine == 'huggingface':
                # messages = [
                #     {"role": "user", "content": prompt}
                # ]
                import torch 
                messages = prompt
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # thinking mode unabled
                )

                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

                # conduct text completion
                with torch.no_grad():

                    generated_ids = self.client.generate(
                        **model_inputs,
                        max_new_tokens=1024,  #16384, 8192
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

                response_data = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                return self.remove_think_blocks(response_data)

        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình tạo nội dung: {e}")
            raise
        