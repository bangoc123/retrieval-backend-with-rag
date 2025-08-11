import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import time
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OpenVINO:
    """OpenVINO model wrapper for efficient inference"""

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
        self._initialize_model()

    def _hf_to_openvino(self):
        
        # Load and export model
        model = OVModelForCausalLM.from_pretrained(self.model_version, export=True)
        model.save_pretrained(self.model_dir)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        tokenizer.save_pretrained(self.model_dir)

        print(f"Model exported to OpenVINO format at {self.model_dir}")
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
        
    # ------------------ Generation ------------------

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
    ) -> List[str]:
        """Generate outputs for a batch of prompts.

        Args:
            prompts: list of prompt strings or list of conversation message-lists/dicts (compatible with your earlier API)
            max_new_tokens: tokens to generate
            batch_size: override batch size
        Returns:
            list of generated strings (one per prompt)
        """

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
                    batch_texts, batch_ids = [], []
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
            responses, generated_tokens = self.run_batch_generate(text)
            elapsed = time.time() - start
            total_time += elapsed
            total_generated_tokens += generated_tokens

            for resp in responses:
                results.append(resp)
            self.tok_per_sec = total_generated_tokens / total_time

        return results

    def get_tokens_per_sec(self):
        return self.tok_per_sec

    def warmup(self, steps: int = 3):
        example = {"role":"user", "content":"Hello"}
        for _ in range(steps):
            try:
                _ = self.generate_content([example])
                print(f"Warmup: {_}")
            except Exception as e:
                logger.warning("warmup failed: %s", e)

    def shutdown(self):
        # Try to release model resources
        try:
            del self.model
        except Exception:
            pass

    