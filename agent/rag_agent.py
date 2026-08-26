# from agent.base_agent import BaseAgent


# class RAGAgent(BaseAgent):
#     """
#     This agent deals with RAG and provides any necessary additional context
#     for the LLM to know before forecasting
#     """

#     def __init__(self, name, llm, tools, system_prompt, prompt_builder,
#                  memory, knowledge_base_tool, max_iterations=10):
#         super().__init__(name, llm, tools, system_prompt, prompt_builder,
#                          memory, max_iterations)
#         self.knowledge_base_tool = knowledge_base_tool

#     def run(self, state):
#         """
#         Running the query for RAG
#         """
#         state.set_current_agent(self.name)
#         try:
#             user_query = state.user_request
#             state.retrieval_query = user_query
#             result = self.knowledge_base_tool.run(user_query)

#             if not result:
#                 state.retrieved_documents = []
#                 state.sources = None
#                 state.rag_context = None
#                 state.mark_agent_complete(self.name)
#                 return state

#             # retrieved_documents = result.get("documents", [])
#             # results = result  # ["documents"]

#             state.retrieved_documents = self._format_documents(
#                 result)
#             # retrieved_documents)
#             state.sources = self._extract_sources(result)
#             # result["sources"]  # result.get("sources", [])
#             state.rag_context = self._build_context(  # result._build_context(
#                 state.retrieved_documents)
#             state.mark_agent_complete(self.name)
#             return state

#         except Exception as e:
#             state.add_error(f"{self.name} failed: {str(e)}")
#             state.current_agent = None
#             return state

#     def _format_documents(self, results):
#         """
#         Formats retrieval queries to fit JSON standards
#         """
#         documents = []
#         for result in results:
#             document = result["document"]
#             documents.append({"text": document.text,
#                               "source": document.metadata.get('source'),
#                               "page": document.metadata.get('page'),
#                               "score": result['score']})
#         return documents

#     def _extract_sources(self, results):
#         """
#         Extracts the sources of the documents for the response
#         """
#         sources = []
#         for result in results:
#             document = result["document"]
#             source = {"file": document.metadata['file'],
#                       "page": document.metadata.get("page"),
#                       "score": result['score']}
#             source.append(sources)
#         return sources

#     def _build_context(self, documents):
#         """
#         This function builds up the context for the RAG model agent
#         """
#         if not documents:
#             return None

#         context_parts = []
#         for document in documents:
#             source = document.get("source", "unknown")
#             page = document.get("page")
#             text = document.get("text", "")
#             context_parts.append(f"""
# SOURCE: {source},
# PAGE: {page},
# {text}
# """)
#         return "\n---\n".join(context_parts)


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
        Running the query for RAG.

        CHANGED: signature is now (task, state) instead of (state).
        Controller dispatches every agent uniformly as
        agent.run(task, state) -- with the old (state)-only signature,
        selecting rag_agent would TypeError immediately.
        """
        state.set_current_agent(self.name)
        try:
            # CHANGED: prefer the specific task the Router crafted for
            # this step (e.g. "find info about holiday effects") over
            # the raw original user_request, falling back to
            # user_request if the router didn't give a task.
            user_query = task or state.user_request
            state.retrieval_query = user_query
            result = self.knowledge_base_tool.run(user_query)

            if not result:
                state.retrieved_documents = []
                state.sources = None
                state.rag_context = None
                state.mark_agent_complete(self.name)
                return state

            state.retrieved_documents = self._format_documents(result)
            state.sources = self._extract_sources(result)
            state.rag_context = self._build_rag_context(
                state.retrieved_documents)
            state.mark_agent_complete(self.name)
            return state

        except Exception as e:
            # CHANGED: use the same {"agent": ..., "error": ...} shape
            # every other agent uses instead of a bare string, since
            # state.errors is otherwise a mix of dicts and strings.
            state.add_error({"agent": self.name, "error": str(e)})
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
            # CHANGED: document metadata uses the key "source" (see
            # rag/document.py's DocumentLoader), not "file" -- the old
            # code did document.metadata['file'], which raised a
            # KeyError every single time this ran.
            source = {"file": document.metadata.get("source"),
                      "page": document.metadata.get("page"),
                      "score": result['score']}
            # CHANGED: was `source.append(sources)` -- source is a dict
            # (no .append method), and the call was backwards anyway.
            # This raised AttributeError before a single source was
            # ever collected.
            sources.append(source)
        return sources

    def _build_rag_context(self, documents):
        """
        This function builds up the context for the RAG model agent.

        CHANGED: renamed from _build_context to _build_rag_context.
        BaseAgent already defines _build_context(self, task, state) for
        its ReAct loop; this subclass was silently shadowing it with an
        incompatible (documents-only) signature. RAGAgent doesn't
        currently call super().run(), so it never surfaced, but it's a
        landmine for anyone who later makes RAGAgent use the inherited
        tool-calling loop.
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
