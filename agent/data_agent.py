# from agent.base_agent import BaseAgent


# class DataAgent(BaseAgent):
#     """
#     This agent class is specifically meant for the Data of the files
#     """

#     def __init__(self, llm, tools, prompt_builder, memory):
#         super().__init__(name="Data Agent", llm=llm, tools=tools,
#                          prompt_builder=prompt_builder, memory=memory,
#                          system_prompt="You're responsible for understanding"
#                          "the data set. You conduct an analysis of the data,"
#                          "such as the columns, rows, statistical analysis,"
#                          "missing values and time-series characteristics")

#     def run(self, task, state):
#         """
#         This function runs the agent and updates the state as necessary
#         """
#         result = super().run(task, state)
#         state.mark_agent_complete(self.name)
#         return result
#         # result = super().run(task, state)
#         # data = state.data
#         # if data is None:
#         #     raise ValueError("No data provided")

#         # summary = self.analyze_dataset(data)  # result)
#         # state.data_summary = summary
#         # state.mark_agent_complete(self.name)
#         # return summary  # result

#     # def analyze_dataset(self, data_frame):
#     #     """
#     #     Conducting analysis on the data
#     #     """
#     #     summary = {
#     #         "rows": len(data_frame),
#     #         "columns": list(data_frame.columns),
#     #         "data_types": data_frame.dtypes.astype(str).to_dict(),
#     #         "missing_values": data_frame.isnull().sum().to_dict(),
#     #         "duplicated_rows": int(data_frame.duplicated().sum())
#     #     }
#     #     return summary

from agent.base_agent import BaseAgent
# import json


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
        # CHANGED: name was "Data Agent" (Title Case). state.completed_agents
        # stores whatever self.name is, but the Router's vocabulary and
        # Controller's agent dict use snake_case ("data_agent"). With the
        # old name, the Router could never match "Data Agent" in
        # completed_agents to its own "data_agent" identifier, so it had
        # no reliable way to know this agent already ran.

    def run(self, task, state):
        """
        This function runs the agent and updates the state as necessary
        """
        result = super().run(task, state)

        # if isinstance(result, str):
        #     try:
        #         data_info = json.loads(result)

        #         target = data_info.get("target_column")
        #         date_column = data_info.get("date_column")

        #         # Validate target against actual dataset
        #         if target and target not in state.data.columns:
        #             raise ValueError(
        #                 f"Target column '{target}' not found in dataset."
        #             )

        #         state.target_column = target
        #         state.date_column = date_column
        #         state.data_summary = data_info.get("data_summary")

        #     except json.JSONDecodeError:
        #         state.add_error({
        #             "agent": self.name,
        #             "error": "DataAgent returned invalid JSON."
        #         })

        state.mark_agent_complete(self.name)
        return result
