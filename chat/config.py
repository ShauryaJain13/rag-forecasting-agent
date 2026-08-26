# from dotenv import load_dotenv
# from groq import Groq
# from FlagEmbedding import BGEM3FlagModel
# import os


# class Configuration:
#     """
#     This class creates a connection between the LLM and the Agent,
#     so that the user can communicate with the model
#     """

#     def __init__(self):
#         load_dotenv()
#         self.api_key = os.getenv('GROQ_API_KEY')
#         self.model_name = os.getenv('MODEL')
#         self.embedding_model = os.getenv('EMBEDDING_MODEL')
#         if not self.api_key:
#             raise ValueError("API Key not found in env")

#         if not self.model_name:
#             raise ValueError("Model not found in env")

#         self.client = Groq(api_key=self.api_key)
#         self.embedding_model = BGEM3FlagModel

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
        self.embedding_model = os.getenv('EMBEDDING_MODEL')

        if not self.api_key:
            raise ValueError("API Key not found in env")

        if not self.model_name:
            raise ValueError("Model not found in env")

        self.client = Groq(api_key=self.api_key)
        # CHANGED: removed `self.embedding_model = BGEM3FlagModel` and the
        # now-unused `from FlagEmbedding import BGEM3FlagModel` import.
        # That line overwrote the env-var string fetched two lines above
        # with a reference to the *class itself* (not an instance), and
        # nothing ever read Configuration.embedding_model anyway --
        # rag/embeddings.py's Embedder loads BGEM3FlagModel independently
        # via its own os.getenv('EMBEDDING_MODEL') call. This just
        # removed dead code and an unnecessary heavy import.