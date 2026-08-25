# from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """
    This class retrieves the data that has been vectorized
    """
    def __init__(self, embedder, vector_store, top_k=5, minimum_threshold=0.6):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.minimum_threshold = minimum_threshold

    def retrieve(self, query):
        """
        Retrieves the appropriate data for the query
        """
        embedded_query = self.embedder.embed_query(query)
        results = self.vector_store.search(embedded_query, self.top_k)
        filtered_results = []
        for result in results:
            if result["score"] >= self.minimum_threshold:
                filtered_results.append(result)

        return filtered_results

        # while cosine_similarity(results[-1],
        #                         embedded_query) < self.minimum_threshold:
        #     results = results[:-2]
        # return results
