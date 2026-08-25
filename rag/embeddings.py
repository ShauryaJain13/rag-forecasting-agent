# from FlagEmbedding import BGEM3FlagModel


class Embedder:
    """
    This class covers the embedding of the chunks in the context
    """

    def __init__(self, model):
        self.model = model

    def embed_documents(self, documents):
        """
        This function embeds the documents to be understood by the LLM
        """
        texts = []
        for document in documents:
            texts.append(document.text)

        embeddings = self.model.encode(texts)
        return embeddings

    def embed_query(self, query):
        """
        This function embeds the queries of the user
        """
        embedding = self.model.encode(query)
        return embedding
