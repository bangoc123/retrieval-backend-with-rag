import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVModelForCausalLM
import time
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OpenVINO:
    """OpenVINO model wrapper for efficient inference with KV cache"""

    def __init__(self, 
                 model_version: str, 
                 local_dir: str = "./openvino_model", 
                 device: str = "cpu", 
                 use_kv_cache: bool = True,
                 batch_size: int = 1,
                 max_new_tokens: int=512,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 use_cache = True):
        """Initialize OpenVINO model
        
        Args:
            model_version (str): HuggingFace model repository ID
            local_dir (str): Local directory to store downloaded models
        """
        self.model_version = model_version
        self.local_dir = local_dir 
        self.model_dir = os.path.join(self.local_dir, self.model_version.split('/')[-1])
        self.use_kv_cache = use_kv_cache
        self.batch_size = batch_size
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p 
        self.do_sample = do_sample 
        self.use_cache = use_cache
        
        # Model architecture parameters (will be auto-detected or set manually)
        self.num_layers = 28
        self.num_heads = 8
        self.head_dim = 128
        
        self._initialize_model()

    def _hf_to_openvino(self):
        # Load and export model
        model = OVModelForCausalLM.from_pretrained(self.model_version, export=True)
        model.save_pretrained(self.model_dir)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        tokenizer.save_pretrained(self.model_dir)

        print(f"Model exported to OpenVINO format at {self.model_dir}")

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
            logger.info(f"Using predefined config for {self.model_version}")
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
                
                logger.info(f"Auto-detected config from model config")
                
            except Exception as e:
                logger.warning(f"Could not auto-detect model config: {e}")
                logger.info("Using default Qwen3-0.6B configuration")
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
        logger.info(f"Manual config set - Layers: {self.num_layers}, Heads: {self.num_heads}, Head dim: {self.head_dim}")

    def _initialize_model(self):
        """Initialize OpenVINO model"""
        try:
            if not os.path.exists(self.model_dir):
                self._hf_to_openvino()
            
            logger.info(f"Loading OVModelForCausalLM from {self.model_dir} on {self.device}")
            # Load OpenVINO model
            self.model = OVModelForCausalLM.from_pretrained(self.model_dir, use_cache=True, task='text-generation-with-past')

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_version, trust_remote_code=True)  
            if self.tokenizer.pad_token is None:
                # set pad token to eos if missing
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Auto-detect model architecture
            self._detect_model_architecture()
            
        except Exception as e:
            self.model = None
            logger.warning("openvino model failed to load: %s", e)

        if self.model is None:
            raise RuntimeError(
                "OVModelForCausalLM could not be loaded. Ensure `optimum[intel]` is installed and the model_dir is valid."
            )
        
        # Configure cache implementation if supported
        try:
            if self.use_kv_cache and hasattr(self.model, "set_cache_implementation"):
                # static is usually best for repeated single-request low-latency generation
                self.model.set_cache_implementation("static")
                logger.info("Set OpenVINO cache implementation to static")
            # Ensure config.use_cache = True if available
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = True
        except Exception as e:
            logger.warning("Could not configure cache implementation: %s", e)

        # Internal KV cache placeholder. The exact format depends on the exported model.
        # For Optimum OpenVINO, forward returns a dict-like object with 'past_key_values' when caching is enabled.
        self.past_key_values = None

        # If optimum wrapper supports .generate and returns numpy/tensors, we can use it directly
        self._has_generate = hasattr(self.model, "generate")
        logger.info("OpenVINO model loaded. has_generate=%s", self._has_generate)
        print(f"DEBUG model.input_names: {self.model.input_names}")
        print(f"DEBUG model.output_names: {self.model.output_names}")

    def prepare_inputs(self, 
                      input_ids, 
                      past_key_values: Optional = None,
                      attention_mask: Optional = None,
                      position_ids: Optional = None
                      ):
        if not self.use_kv_cache:
            return {"input_ids": input_ids}
        return {"input_ids": input_ids, "past_key_values": past_key_values} if past_key_values is not None else {"input_ids": input_ids}

        #=======================================================

    def extract_kv_cache(self, outputs) -> Optional[Dict]:
        if hasattr(outputs, "past_key_values"):
            return outputs.past_key_values
        if isinstance(outputs, dict) and "past_key_values" in outputs:
            return outputs["past_key_values"]
        return None

    def generate_token(self, input_ids, past_key_values=None) -> Tuple[int, Optional[Dict]]:

        input_ids = torch.from_numpy(input_ids)
        # print(f"DEBUG input_ids: {input_ids.type}")
        # print(f"DEBUG past_key_values: {past_key_values}")
        inputs = self.prepare_inputs(input_ids, past_key_values)
        # print(f"DEBUG generate_token: input: {inputs}\n")
        outputs = self.model(**inputs, use_cache=True)
        # print(f"DEBUG outputs = {len(outputs)}")
        # print(f"DEBUG outputs = {outputs[0].shape}")
        # print(f"DEBUG outputs = {outputs[1]}")
        # print(f"DEBUG generate_token: output: {outputs}\n")

        # print(f"DEBUG past_key_values: outputs.past_key_values {outputs.past_key_values}")
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_logits = logits[:, -1, :]
        # print(f"DEBUG generate_token: next_token_logits: {next_token_logits}\n")
        next_token_id = int(np.argmax(next_token_logits, axis=-1)[0])
        new_kv_cache = self.extract_kv_cache(outputs)
        # print(f"DEBUG generate_token: next_token_id: {next_token_id}\n")
        # print(f"DEBUG generate_token: new_kv_cache: {new_kv_cache}\n")
        return next_token_id, new_kv_cache

    def generate_content(self, prompt: str | list[dict[str, str]]) -> str:
        """
        Generate text with KV cache, supporting:
        - str: plain prompt
        - list[dict[str, str]]: chat-style messages
        """
        # If the input is chat-style, convert to prompt text
        if isinstance(prompt, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        elif isinstance(prompt, str):
            prompt_text = prompt
        else:
            raise ValueError("Prompt must be a string or a list of {role, content} dicts.")

        logger.info(f"Generating with OpenVINO: '{prompt_text[:50]}...'")

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt_text, return_tensors="np")
        # print(f"DEBUG: input_ids {input_ids}")
        prompt_tokens = input_ids.shape[1]
        generated_tokens = []
        kv_cache = None

        t0 = time.perf_counter()
        for step in range(self.max_new_tokens):
            step_input = (
                input_ids if step == 0 
                else np.array([[generated_tokens[-1]]], dtype=np.int64)
            )
            # print(f"DEBUG: step input {step_input}")
            next_token_id, kv_cache = self.generate_token(step_input, kv_cache)
            # print(f"DEBUG: next token id {next_token_id}, kv_cache {kv_cache}")
            generated_tokens.append(next_token_id)

            if next_token_id == self.tokenizer.eos_token_id:
                break
        t1 = time.perf_counter()

        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"DEBUG generated_text {generated_text}")
        completion_tokens = len(generated_tokens)
        total_tokens = prompt_tokens + completion_tokens
        tps = completion_tokens / (t1 - t0 + 1e-9)

        logger.info(
            f"Generated {completion_tokens} tokens in {(t1-t0):.3f}s "
            f"({tps:.2f} tok/s) | prompt={prompt_tokens}, total={total_tokens}"
        )
        return generated_text

    def warmup(self, steps: int = 3):
        example = {"role":"user", "content":"Hello"}
        for _ in range(steps):
            try:
                _ = self.generate_content([example])
                logger.info(f"Warmup: {_}")
            except Exception as e:
                logger.warning("warmup failed: %s", e)

    def shutdown(self):
        # Try to release model resources
        try:
            del self.model
        except Exception:
            pass