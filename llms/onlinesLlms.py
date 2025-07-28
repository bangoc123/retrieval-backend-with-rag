import google.generativeai as genai
import openai

class OnLineLLMs:
    def __init__(self, model_name: str, api_key: str, model_version: str):
        """Initialize model with the specified name, API key, and model version."""
        self.model_name = model_name.lower()
        self.model_version = model_version

        if self.model_name == "gemini" and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=model_version)
        elif self.model_name == "openai" and api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError("Unsupported model name or missing API key.")


    def generate_content(self, prompt: str) -> str:
        """Generate content using the online LLM based on the provided prompt.
            input: prompt (str): The prompt to generate content for.
            output: str: The generated content.
        """
        if self.model_name == "gemini":
            gemini_messages = [
                {"role": msg["role"], "parts": [msg["content"]]} for msg in prompt
            ]
            response = self.model.generate_content(gemini_messages)
            try:
                return response.text 
            except:
                return response.candidates[0].content.parts[0].text

        elif self.model_name == "openai":
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=prompt
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported model name: {self.name}")
