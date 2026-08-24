from agent.base_agent import BaseAgent


class AnomalyAgent(BaseAgent):
    """
    Agent responsible for detecting and interpreting anomalies
    in the dataset.

    The agent uses the LLM to determine what anomaly analysis
    should be performed and uses anomaly-detection tools to
    perform the actual numerical calculations.
    """

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(
            name="Anomaly Agent",
            llm=llm,
            tools=tools,
            prompt_builder=prompt_builder,
            memory=memory,
            system_prompt=(
                "You are an Anomaly Detection Agent in a "
                "multi-agent data forecasting system.\n\n"

                "Your responsibility is to determine whether "
                "the dataset contains unusual observations "
                "that could affect forecasting.\n\n"

                "You should:\n"
                "1. Examine the dataset characteristics "
                "provided in the shared state.\n"
                "2. Select an appropriate anomaly detection "
                "approach.\n"
                "3. Use the available anomaly detection tools "
                "to perform the analysis.\n"
                "4. Interpret the results.\n"
                "5. Store relevant findings in shared state.\n\n"

                "Do not blindly assume that every statistical "
                "outlier is an error. An unusual observation "
                "may represent a legitimate event.\n\n"

                "Your findings will be used by the Forecasting "
                "Agent to determine whether anomalies could "
                "affect the forecast."
            )
        )

    def run(self, task, state):
        """
        Run the Anomaly Agent's reasoning and tool-use loop.
        """
        result = super().run(task, state)
        self._update_state(state, result)
        state.mark_agent_complete(self.name)
        return result

    def _update_state(self, state, result):
        """
        Store the agent's interpretation in shared state.
        """
        state.anomaly_analysis = result
