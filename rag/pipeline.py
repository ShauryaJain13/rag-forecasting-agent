class RAGPipeline:
    """
    The official pipeline for the RAG system, similar to the router for MAS.
    It deals with retrieval and generation
    """

    def __init__(self, retriever, llm_client, prompt_builder):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    def answer(self, query):
        """
        Answering the user's query, by creating a pipeline to the relevant
        agents, tools and information
        """
        retrieved_docs = self.retriever.retrieve(query)
        if not retrieved_docs:  # is None:
            return {"answer": ("I cannot find the relevant information in the"
                               "knowledge base"),
                    "sources": []}

        prompt = self.prompt_builder.build(query, retrieved_docs)
        response = self.llm_client.generate(prompt)
        sources = self.extract_sources(retrieved_docs)
        return {"answer": response,
                "sources": sources}

    def extract_sources(self, results):
        """
        Extracts the sources of the documents for the response
        """
        sources = []
        for result in results:
            document = result["document"]
            source = {"file": document.metadata['file'],
                      "page": document.metadata.get("page"),
                      "score": result['score']}
            source.append(sources)
        return sources


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
