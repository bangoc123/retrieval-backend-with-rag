from llms.localLlms import LocalLLMs
from llms.onlinesLlms import OnLineLLMs

class LLMs:
    def __init__(self, type : str, model_name : str, engine : str = None, api_key : str = None, model_version: str = None, base_url: str = None):
        """Initialize object LocalLLMs or OnlineLLMs with the provided configuration.
            not return 
        """
        if type == "offline":
            self.llm = LocalLLMs(engine=engine, model_name=model_name,base_url=base_url)
        elif type == "online":
            self.llm = OnLineLLMs(model_name=model_name, api_key=api_key, model_version=model_version, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM type: {type}")

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the LLM based on the provided prompt.
        input: prompt (str): The prompt to generate content for.
        output: str: The generated content.
        """
        return self.llm.generate_content(prompt)

       