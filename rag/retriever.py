class Retriever:
    """
    This class retrieves the data that has been vectorized
    """
    def __init__(self, embedder, vector_store, top_k):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query):
        