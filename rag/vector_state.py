from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    This class vectorizes the inputted documents and stores them
    """

    def __init__(self):
        self.vector_storage = []
        self.document_storage = []

    def add_documents(self, documents, embeddings):
        """
        This method vectorizes the chunks of the documents and embeds them
        """
        try:
            for document, embedding in (documents, embeddings):
                self.document_storage.append(document)
                self.vector_storage.append(embedding)
            return True
        except Exception as e:
            raise f"An exception {str(e)} was experienced"

    def search(self, query_embedding, top_k):
        """
        Searching for top k queries in the embedding
        """
        similarities = []
        for embedding in self.vector_storage:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append(similarity)
        similarities.sort(reverse=True)

        return similarities[:top_k]
