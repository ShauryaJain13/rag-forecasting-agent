class RAGPipeline:
    """
    This class coordinates the retrieval of rag information
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, query):
        """
        Retrieves the answer of the query
        """
        return self.retriever.retrieve(query)


class KnowledgeBase:
    """
    This class serves as an abstraction layer
    """

    def __init__(self, loader, chunker, embedder, vector_storage):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_storage = vector_storage

    def index_file(self, filepath):
        """
        Creates an index for the file
        """
        # try:
        documents = self.loader.load(filepath)
        chunks = self.chunker.chunk_documents(documents)
        embeddings = self.embedder.embed_documents(chunks)
        self.vector_storage.add_documents(chunks, embeddings)
        return True
        # except Exception as e:
        #     return f"Ran into error {str(e)}"
