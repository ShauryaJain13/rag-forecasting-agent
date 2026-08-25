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
        result = self.rag_pipeline.answer(query)
        return result
