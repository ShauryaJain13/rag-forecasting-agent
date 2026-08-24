from agent.base_agent import BaseAgent


class DataAgent(BaseAgent):
    """
    This agent class is specifically meant for the Data of the files
    """

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(name="Data Agent", llm=llm, tools=tools,
                         prompt_builder=prompt_builder, memory=memory,
                         system_prompt="You are responsible for understanding"
                         "the data set. You conduct an analysis of the data,"
                         "such as the columns, rows, statistical analysis,"
                         "missing values and time-series characteristics")

    def run(self, task, state):
        """
        This function runs the agent and updates the state as necessary
        """
        result = super().run(task, state)
        state.mark_agent_complete(self.name)
        return result
        # result = super().run(task, state)
        # data = state.data
        # if data is None:
        #     raise ValueError("No data provided")

        # summary = self.analyze_dataset(data)  # result)
        # state.data_summary = summary
        # state.mark_agent_complete(self.name)
        # return summary  # result

    # def analyze_dataset(self, data_frame):
    #     """
    #     Conducting analysis on the data
    #     """
    #     summary = {
    #         "rows": len(data_frame),
    #         "columns": list(data_frame.columns),
    #         "data_types": data_frame.dtypes.astype(str).to_dict(),
    #         "missing_values": data_frame.isnull().sum().to_dict(),
    #         "duplicated_rows": int(data_frame.duplicated().sum())
    #     }
    #     return summary
