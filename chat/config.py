from dotenv import load_dotenv
from groq import Groq
import os


class Configuration:
    """
    This class creates a connection between the LLM and the Agent,
    so that the user can communicate with the model
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model_name = os.getenv('MODEL')
        if not self.api_key:
            raise ValueError("API Key not found in env")

        if not self.model_name:
            raise ValueError("Model not found in env")

        self.client = Groq(api_key=self.api_key)
