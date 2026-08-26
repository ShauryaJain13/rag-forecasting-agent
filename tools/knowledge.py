class KnowledgeBaseTool:
    """
    Stores the pipeline of the knowledge base for RAG
    as a tool that the LLM can use
    """

    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline

    def run(self, query):
        """
        Runs the RAG pipeline process for an output for the query
        """

        return self.rag_pipeline.retrieve(query)

        # results = self.retriever.retrieve(query)
        # sources = []
        # for result in results:

        #     document = result["document"]
        #     sources.append({"source": document.metadata.get("source"),
        #                     "page": document.metadata.get("page"),
        #                     "score": result["score"]})

        # return {"documents": results,
        #         "sources": sources}

        # return result
