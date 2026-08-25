from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
import os


class Embedder:
    """
    This class covers the embedding of the chunks in the context
    """

    def __init__(self):  # , model):
        load_dotenv()
        self.model = BGEM3FlagModel(os.getenv('EMBEDDING_MODEL'),
                                    use_fp16=True)

    def embed_documents(self, documents):
        """
        This function embeds the documents to be understood by the LLM
        """
        texts = [document.text for document in documents]
        # for document in documents:
        #     texts.append(document.text)

        embeddings = self.model.encode(texts, batch_size=12, max_length=8192)
        return embeddings["dense_vecs"]

    def embed_query(self, query):
        """
        This function embeds the queries of the user
        """
        embedding = self.model.encode([query], batch_size=1, max_length=8192)
        return embedding["dense_vecs"][0]
