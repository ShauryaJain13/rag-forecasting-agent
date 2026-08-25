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

    def run(self, state):
        """
        Running the query for RAG
        """
        user_query = state.user_query
