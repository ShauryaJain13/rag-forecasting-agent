from agent.base_agent import BaseAgent


class RAGAgent(BaseAgent):
    """
    This agent deals with RAG and provides any necessary additional context
    for the LLM to know before forecasting
    """

    def __init__(self, name, llm, tools, system_prompt, prompt_builder,
                 memory, knowledge_base_tool, max_iterations=10):
        super().__init__(name, llm, tools, system_prompt, prompt_builder,
                         memory, max_iterations)
        self.knowledge_base_tool = knowledge_base_tool

    def run(self, task, state):
        """
        Retrieve relevant information from the knowledge base
        and store it in shared AgentState.
        """

        state.set_current_agent(self.name)

        try:
            query = task.strip() if task else state.user_request
            state.retrieval_query = query
            results = self.knowledge_base_tool.run(query)
            if not results:
                state.retrieved_documents = []
                state.sources = []
                state.rag_context = None
                state.mark_agent_complete(self.name)
                return state

            state.retrieved_documents = self._format_documents(results)
            state.sources = self._extract_sources(results)
            state.rag_context = self._build_rag_context(
                state.retrieved_documents)
            state.mark_agent_complete(self.name)
            return state

        except Exception as e:
            state.add_error({"agent": self.name,
                             "error": str(e)})
            state.current_agent = None
            return state

    def _format_documents(self, results):
        """
        Formats retrieval queries to fit JSON standards
        """
        documents = []
        for result in results:
            document = result["document"]
            documents.append({"text": document.text,
                              "source": document.metadata.get('source'),
                              "page": document.metadata.get('page'),
                              "score": result['score']})
        return documents

    def _extract_sources(self, results):
        """
        Extracts the sources of the documents for the response
        """
        sources = []
        for result in results:
            document = result["document"]
            source = {"file": document.metadata.get("source"),
                      "page": document.metadata.get("page"),
                      "score": result['score']}
            sources.append(source)
        return sources

    def _build_rag_context(self, documents):
        """
        This function builds up the context for the RAG model agent.
        """
        if not documents:
            return None

        context_parts = []
        for document in documents:
            source = document.get("source", "unknown")
            page = document.get("page")
            text = document.get("text", "")
            context_parts.append(f"""
SOURCE: {source},
PAGE: {page},
{text}
""")
        return "\n---\n".join(context_parts)
