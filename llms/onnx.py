import requests
import re
import os
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import time

class ONNXModel:
    """ONNX model wrapper for efficient inference"""
    
    def __init__(self, model_version: str, local_dir: str = "./onnx_models"):
        """Initialize ONNX model
        
        Args:
            model_version (str): HuggingFace model repository ID
            local_dir (str): Local directory to store downloaded models
        """
        self.model_version = model_version
        self.local_dir = local_dir
        self.onnx_session = None
        self.tokenizer = None
        self.input_names = []
        self.output_names = []
        self.num_layers = 28
        self.num_heads = 8
        self.head_dim = 128
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ONNX model by downloading and loading it"""
        print("🚀 Initializing ONNX model...")
        
        # Download the pre-converted ONNX model
        print(f"📥 Downloading ONNX model: {self.model_version}")
        model_path = snapshot_download(
            repo_id=self.model_version,
            local_dir=self.local_dir
        )
        print(f"✅ Downloaded ONNX model to: {model_path}")
        
        # Load ONNX session
        onnx_model_path = os.path.join(model_path, "onnx", "model.onnx")
        if not os.path.exists(onnx_model_path):
            # Try alternative paths
            alternative_paths = [
                os.path.join(model_path, "model.onnx"),
                os.path.join(model_path, "onnx", "model_fp16.onnx"),
                os.path.join(model_path, "model_fp16.onnx")
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    onnx_model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"ONNX model file not found in {model_path}")
        
        print(f"🔧 Loading ONNX Runtime session from: {onnx_model_path}")
        self.onnx_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model architecture info
        self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]
        
        # Auto-detect model architecture
        self._detect_model_architecture()
        
        print(f"✅ ONNX Model loaded successfully!")
        print(f"   - Input tensors: {len(self.input_names)}")
        print(f"   - Output tensors: {len(self.output_names)}")
        print(f"   - Layers: {self.num_layers}, Heads: {self.num_heads}, Head dim: {self.head_dim}")

    def _detect_model_architecture(self):
        """Auto-detect or set model architecture parameters"""
        # Model architecture configurations
        model_configs = {}
        
        # Try to get configuration from predefined configs
        if self.model_version in model_configs:
            config = model_configs[self.model_version]
            self.num_layers = config["num_layers"]
            self.num_heads = config["num_heads"]
            self.head_dim = config["head_dim"]
            print(f"🔍 Using predefined config for {self.model_version}")
        else:
            # Try to load from tokenizer config if available
            try:
                config = AutoConfig.from_pretrained(self.model_version)
                
                # Map common config names to our attributes
                self.num_layers = getattr(config, 'num_hidden_layers', 
                                        getattr(config, 'n_layer', 
                                              getattr(config, 'num_layers', 28)))
                
                self.num_heads = getattr(config, 'num_attention_heads', 
                                       getattr(config, 'n_head', 
                                             getattr(config, 'num_heads', 8)))
                
                # Calculate head_dim from hidden_size and num_heads
                hidden_size = getattr(config, 'hidden_size', 
                                    getattr(config, 'd_model', 1024))
                self.head_dim = hidden_size // self.num_heads
                
                print(f"🔍 Auto-detected config from model config")
                
            except Exception as e:
                print(f"⚠️ Could not auto-detect model config: {e}")
                print("🔧 Using default Qwen3-0.6B configuration")
                # Default fallback to Qwen3-0.6B specs
                self.num_layers = 28
                self.num_heads = 8  
                self.head_dim = 128

    def set_architecture(self, num_layers: int, num_heads: int, head_dim: int):
        """Manually set model architecture parameters
        
        Args:
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            head_dim (int): Dimension of each attention head
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        print(f"🔧 Manual config set - Layers: {self.num_layers}, Heads: {self.num_heads}, Head dim: {self.head_dim}")

    def prepare_inputs(self, input_ids: np.ndarray, 
                      attention_mask: Optional[np.ndarray] = None,
                      past_key_values: Optional[Dict[str, np.ndarray]] = None,
                      position_ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Prepare inputs for ONNX inference"""
        batch_size, seq_length = input_ids.shape

        inputs = {
            'input_ids': input_ids.astype(np.int64)
        }

        # Attention mask
        if attention_mask is None:
            attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
        inputs['attention_mask'] = attention_mask.astype(np.int64)

        # Position IDs
        if position_ids is None:
            if past_key_values is None:
                position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
                position_ids = np.repeat(position_ids, batch_size, axis=0)
            else:
                past_length = past_key_values[f'past_key_values.0.key'].shape[2]
                position_ids = np.array([[past_length]], dtype=np.int64)
                position_ids = np.repeat(position_ids, batch_size, axis=0)
        inputs['position_ids'] = position_ids

        # KV cache preparation
        if past_key_values is None:
            past_seq_length = 0
            total_seq_length = seq_length
        else:
            past_seq_length = past_key_values[f'past_key_values.0.key'].shape[2]
            total_seq_length = past_seq_length + seq_length

        if past_key_values is not None:
            full_attention_mask = np.ones((batch_size, total_seq_length), dtype=np.int64)
            inputs['attention_mask'] = full_attention_mask

        # Add KV cache tensors
        for layer_idx in range(self.num_layers):
            key_name = f'past_key_values.{layer_idx}.key'
            value_name = f'past_key_values.{layer_idx}.value'

            if past_key_values is None:
                cache_shape = (batch_size, self.num_heads, 0, self.head_dim)
                inputs[key_name] = np.zeros(cache_shape, dtype=np.float32)
                inputs[value_name] = np.zeros(cache_shape, dtype=np.float32)
            else:
                inputs[key_name] = past_key_values[key_name]
                inputs[value_name] = past_key_values[value_name]

        return inputs

    def extract_kv_cache(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract KV cache from ONNX outputs"""
        kv_cache = {}
        for i, output_name in enumerate(self.output_names[1:], 1):
            if output_name.startswith('present.'):
                layer_idx = output_name.split('.')[1]
                cache_type = output_name.split('.')[2]
                past_name = f'past_key_values.{layer_idx}.{cache_type}'
                kv_cache[past_name] = outputs[i]
        return kv_cache

    def generate_token(self, input_ids: np.ndarray, 
                      past_key_values: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, Dict[str, np.ndarray]]:
        """Generate single token using ONNX model"""
        # Prepare inputs
        inputs = self.prepare_inputs(input_ids, past_key_values=past_key_values)
        
        # Run inference
        outputs = self.onnx_session.run(None, inputs)
        
        # Get next token
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]
        next_token_id = int(np.argmax(next_token_logits))
        
        # Extract KV cache
        new_kv_cache = self.extract_kv_cache(outputs)
        
        return next_token_id, new_kv_cache

    def generate(self, prompt: str, max_new_tokens: int = 4096, 
                temperature: float = 1.0, do_sample: bool = False) -> str:
        """Generate text using ONNX model
        
        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (not implemented yet)
            do_sample (bool): Whether to use sampling (not implemented yet)
            
        Returns:
            str: Generated text
        """
        print(f"🚀 Generating with ONNX: '{prompt[:50]}...' (max_tokens: {max_new_tokens})")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        prompt_tokens = input_ids.shape[1]
        
        generated_tokens = []
        kv_cache = None

        t0 = time.perf_counter()
        
        # Generation loop
        for step in range(max_new_tokens):
            if step == 0:
                current_input = input_ids
            else:
                current_input = np.array([[generated_tokens[-1]]], dtype=np.int64)
            
            try:
                next_token_id, kv_cache = self.generate_token(
                    current_input, 
                    past_key_values=kv_cache
                )
                
                generated_tokens.append(next_token_id)
                
                # Check for EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                    
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                break

        t1 = time.perf_counter()
        elapsed = max(t1 - t0, 1e-9)  # guard div-by-zero

        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


        full_text = prompt + generated_text
        completion_tokens = len(generated_tokens)
        total_tokens = prompt_tokens + completion_tokens
        tps = completion_tokens / elapsed

        print(f"✅ Generated {completion_tokens} tokens in {elapsed:.3f}s "
          f"({tps:.2f} tok/s) | prompt={prompt_tokens}, total={total_tokens}")

        return generated_text

    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, return_tensors="np")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_version": self.model_version,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.tokenizer.vocab_size,
            "input_names": self.input_names,
            "output_names": self.output_names
        }