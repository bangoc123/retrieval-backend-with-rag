class LocalLLMs:
    def __init__(self, engine: str, model_name: str, base_url: str = None, **kwargs):
        """ Initialize the LocalLLMs class 
            Args:
            engine (str): "ollama" or "vllm".
            model_name (str): name of model ("llama3","meta-llama/Llama-2-7b-chat-hf").
            base_url (str, optional): BASE URL of server engine.
            **kwargs: arguments for vLLM.
        """

    def _initialize_ollama_model(self, model_name: str):
        """Pull the specified model from the Ollama server."""
    
    def _initialize_vllm_model(self, model_name: str, **kwargs):
        """Initialize the vLLM model with the specified name and parameters."""
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using the local LLM based on the provided prompt.
            input: prompt (str): The prompt to generate content for.
            output: str: The generated content.
        """
        