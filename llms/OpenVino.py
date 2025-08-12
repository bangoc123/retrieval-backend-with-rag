import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVModelForCausalLM
import time
import logging


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
            self.model = OVModelForCausalLM.from_pretrained(self.model_dir).to(self.device)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)  
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

    def prepare_inputs(self, input_ids, 
                      attention_mask: Optional = None,
                      past_key_values: Optional[Dict] = None,
                      position_ids: Optional = None):
        """Prepare inputs for OpenVINO inference"""
        batch_size, seq_length = input_ids.shape

        inputs = {
            'input_ids': input_ids
        }

        # Attention mask
        if attention_mask is None:
            attention_mask = input_ids.new_ones((batch_size, seq_length))
        
        # Position IDs
        if position_ids is None:
            if past_key_values is None:
                position_ids = input_ids.new_zeros((batch_size, seq_length))
                for i in range(batch_size):
                    position_ids[i] = input_ids.new_tensor(range(seq_length))
            else:
                # Get past length from KV cache
                past_length = past_key_values[0][0].shape[2] if past_key_values else 0
                position_ids = input_ids.new_tensor([[past_length] * seq_length] * batch_size)

        # Handle KV cache and attention mask extension
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            total_seq_length = past_length + seq_length
            
            # Extend attention mask to include past
            full_attention_mask = input_ids.new_ones((batch_size, total_seq_length))
            inputs['attention_mask'] = full_attention_mask
            inputs['past_key_values'] = past_key_values
        else:
            inputs['attention_mask'] = attention_mask

        inputs['position_ids'] = position_ids
        return inputs

    def generate_token(self, input_ids, past_key_values: Optional[Dict] = None):
        """Generate single token using OpenVINO model with KV cache"""
        # Prepare inputs
        inputs = self.prepare_inputs(input_ids, past_key_values=past_key_values)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get next token
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        
        if self.do_sample:
            # Apply temperature and top_p sampling
            next_token_logits = next_token_logits / self.temperature
            
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            # Greedy decoding
            next_token_id = torch.argmax(next_token_logits).item()
        
        # Extract new KV cache
        new_kv_cache = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
        
        return next_token_id, new_kv_cache

    def generate_with_kv_cache(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate text using OpenVINO model with manual KV cache management
        
        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated text
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        logger.info(f"Generating with OpenVINO KV cache: '{prompt[:50]}...' (max_tokens: {max_new_tokens})")
        
        # Import torch here to avoid dependency issues
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for manual KV cache generation")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = input_ids.shape[1]
        
        generated_tokens = []
        kv_cache = None
        
        t0 = time.perf_counter()
        
        # Generation loop
        for step in range(max_new_tokens):
            if step == 0:
                current_input = input_ids
            else:
                current_input = torch.tensor([[generated_tokens[-1]]], dtype=torch.long)
            
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
                logger.error(f"Error at step {step}: {e}")
                break

        t1 = time.perf_counter()
        elapsed = max(t1 - t0, 1e-9)  # guard div-by-zero
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        completion_tokens = len(generated_tokens)
        total_tokens = prompt_tokens + completion_tokens
        tps = completion_tokens / elapsed

        logger.info(f"Generated {completion_tokens} tokens in {elapsed:.3f}s "
                   f"({tps:.2f} tok/s) | prompt={prompt_tokens}, total={total_tokens}")

        return generated_text

    # ------------------ Original Generation Methods ------------------

    def run_batch_generate(self, texts):
        # texts: list[str]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)  # pad to max in batch
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()      # original token lengths per item
        
        # generate (works with batched inputs)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.use_cache
        )
        generated_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
        # produced shape: (batch_size, seq_len_total)
        responses = []
        for i, seq in enumerate(generated):
            orig_len = int(input_lens[i])
            new_tokens = seq[orig_len:].tolist()   # slice new tokens for this sample
            responses.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return responses, generated_tokens
    

    def generate_content(
        self,
        prompts: list[list[dict[str, str]]] | list[dict[str, str]],
        use_manual_kv_cache: bool = False
    ) -> List[str]:
        """Generate outputs for a batch of prompts.

        Args:
            prompts: list of prompt strings or list of conversation message-lists/dicts
            use_manual_kv_cache: whether to use manual KV cache implementation
        Returns:
            list of generated strings (one per prompt)
        """
        
        if use_manual_kv_cache:
            # Use manual KV cache for single prompt generation
            if isinstance(prompts, list) and len(prompts) == 1:
                if isinstance(prompts[0], list):
                    text = self.tokenizer.apply_chat_template(
                        prompts[0],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                else:
                    text = self.tokenizer.apply_chat_template(
                        prompts,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                generated_text = self.generate_with_kv_cache(text)
                return [generated_text]

        batch_texts = []
        total_time = 0.0
        results = []
        total_generated_tokens = 0
        
        if prompts and isinstance(prompts[0], list):
            for prompt in prompts:
                batch_texts.append(self.tokenizer.apply_chat_template(
                                    prompt,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False
                                    ))
                if len(batch_texts) >= self.batch_size:
                    start = time.time()
                    responses, generated_tokens = self.run_batch_generate(batch_texts)
                    elapsed = time.time() - start
                    total_time += elapsed
                    total_generated_tokens += generated_tokens

                    for resp in responses:
                        results.append(resp)
                    batch_texts = []
            if batch_texts:
                start = time.time()
                responses, generated_tokens = self.run_batch_generate(batch_texts)
                elapsed = time.time() - start
                total_time += elapsed
                total_generated_tokens += generated_tokens
                for resp in responses:
                    results.append(resp)
            self.tok_per_sec = total_generated_tokens / total_time
        else:
            text = self.tokenizer.apply_chat_template(
                    prompts,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )
            start = time.time()
            responses, generated_tokens = self.run_batch_generate([text])
            elapsed = time.time() - start
            total_time += elapsed
            total_generated_tokens += generated_tokens

            for resp in responses:
                results.append(resp)
            self.tok_per_sec = total_generated_tokens / total_time

        return results

    def get_tokens_per_sec(self):
        return getattr(self, 'tok_per_sec', 0)

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_version": self.model_version,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.tokenizer.vocab_size,
            "device": self.device,
            "use_kv_cache": self.use_kv_cache
        }

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