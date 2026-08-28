# from agent.base_agent import BaseAgent
# import json


# class DataAgent(BaseAgent):
#     """
#     This agent class is specifically meant for the Data of the files
#     """

#     def __init__(self, llm, tools, prompt_builder, memory):
#         super().__init__(name="data_agent", llm=llm, tools=tools,
#                          prompt_builder=prompt_builder, memory=memory,
#                          system_prompt="""
# You are the Data Agent.

# Inspect the dataset. Identify:
# - target column
# - date/time column
# - useful covariates

# Use the user's explicitly stated target.
# If none is given, use the only numerical column; otherwise return null.

# Return ONLY JSON:
# {
#  "target_column": "...",
#  "date_column": "...",
#  "covariates": ["..."],
#  "data_summary": "..."
# }
# """)

#     def run(self, task, state):
#         """
#         This function runs the agent and updates the state as necessary
#         """
#         # result = super().run(task, state)
#         # state.mark_agent_complete(self.name)
#         # return result

#         result = super().run(task, state)

#         if not result:
#             return result

#         try:
#             if isinstance(result, str):
#                 result = json.loads(result)

#             state.target_column = result.get("target_column")
#             state.date_column = result.get("date_column")
#             state.covariates = result.get("covariates", [])
#             state.data_summary = result.get("data_summary")

#         except (json.JSONDecodeError, AttributeError):
#             state.add_error({
#                 "agent": self.name,
#                 "error": "DataAgent returned invalid JSON."
#             })

#         state.mark_agent_complete(self.name)
#         return result

from agent.base_agent import BaseAgent


class DataAgent(BaseAgent):

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(
            name="data_agent",
            llm=llm,
            tools=tools,
            prompt_builder=prompt_builder,
            memory=memory,
            system_prompt=""
        )

    def run(self, task, state):

        state.set_current_agent(self.name)

        try:
            # Load the dataset
            read_csv = self.tools.get("read_csv")

            if read_csv is None:
                raise ValueError("read_csv tool is not registered.")

            # The filename is currently supplied in the user's request.
            # For your current project, extract it from the task/request.
            file_path = self._extract_file_path(
                task,
                state.user_request
            )

            read_csv.execute({
                "file_path": file_path
            })

            if state.data is None:
                raise ValueError("Dataset could not be loaded.")

            # Target was explicitly supplied by the user.
            target = self._extract_target(
                task,
                state.user_request,
                state.data.columns
            )

            if target is not None:
                state.target_column = target

            # Basic dataset summary
            analyzer = self.tools.get("analyze_dataset")

            if analyzer is not None:
                state.data_summary = analyzer.execute({})

            state.mark_agent_complete(self.name)

            return state

        except Exception as e:
            state.add_error({
                "agent": self.name,
                "error": str(e)
            })
            state.current_agent = None
            return state

    def _extract_target(self, task, request, columns):
        """
        Extract target column from the user's request.
        """

        text = f"{request} {task}".lower()

        for column in columns:
            if column.lower() in text:
                if "target" in text:
                    return column

        return None

    def _extract_file_path(self, task, request):
        """
        Extract CSV filename from the user's request.
        """

        text = f"{request} {task}"

        start = text.find('"')
        end = text.find('"', start + 1)

        if start != -1 and end != -1:
            return text[start + 1:end]

        # fallback
        for word in text.split():
            if word.lower().endswith(".csv"):
                return word.strip("'\"., ")

        raise ValueError("Could not determine the CSV file path.")
