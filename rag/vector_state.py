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
        if len(documents) != len(embeddings):
            raise ValueError("Number of document chunks must match number of"
                             "vectors")

        # try:
        for document, embedding in zip(documents, embeddings):
            self.document_storage.append(document)
            self.vector_storage.append(embedding)
        return True
        # except Exception as e:
        # raise RuntimeError(f"An exception {str(e)} was experienced")

    def search(self, query_embedding, top_k):
        """
        Searching for top k queries in the embedding
        """
        if not self.vector_storage:
            return []

        similarities = []
        for document, embedding in zip(self.document_storage,
                                       self.vector_storage):
            similarity = cosine_similarity([query_embedding],
                                           [embedding])[0][0]
            similarities.append({"document": document,
                                 "score": float(similarity)})
        similarities.sort(key=lambda x: x["score"], reverse=True)

        return similarities[:top_k]
