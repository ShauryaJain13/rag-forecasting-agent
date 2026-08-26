from agent.base_agent import BaseAgent


class DataAgent(BaseAgent):
    """
    This agent class is specifically meant for the Data of the files
    """

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(name="data_agent", llm=llm, tools=tools,
                         prompt_builder=prompt_builder, memory=memory,
                         system_prompt="""
You are the Data Agent.

Load and inspect the dataset.

1. Call read_csv to load the CSV.
2. Call analyze_dataset after loading it.
3. If the user specified a target column, pass that exact
   column name to analyze_dataset.
4. If no target was specified, let analyze_dataset determine it.
5. Do not answer with JSON.
6. After the tools finish, briefly confirm completion.
""")

    def run(self, task, state):
        """
        This function runs the agent and updates the state as necessary
        """
        result = super().run(task, state)
        state.mark_agent_complete(self.name)
        return result
