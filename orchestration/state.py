class AgentState:
    """
    This class monitors the 'state' of the model
    """

    def __init__(self, user_request):
        self.user_request = user_request
        self.data = None
        self.data_summary = None
        self.target_column = None  # CHANGED: explicit default so
    # getattr/hasattr checks in
    # anomalies.py / forecasting_agent.py
    # behave predictably instead of
    # relying on the attribute simply
    # never existing
        self.forecast = None
        self.forecast_metrics = None
        self.anomalies = []
        self.anomaly_analysis = None
        self.current_agent = None
        self.completed_agents = []
        self.errors = []

    def add_error(self, error):
        """
        Adds the error to the current state
        """
        self.errors.append(error)  # CHANGED: removed the `return` --
                                    # list.append() always returns None,
                                    # so this had no effect either way,
                                    # just cleaner without it

    def mark_agent_complete(self, agent):
        """
        Marking agent after it finished its task
        """
        if agent not in self.completed_agents:
            self.completed_agents.append(agent)

        self.current_agent = None

    def set_current_agent(self, agent):
        """
        Setting current agent
        """
        self.current_agent = agent

    def _data_preview(self):
        """
        CHANGED (new method): build a small, JSON-friendly summary of the
        loaded dataset instead of embedding the full DataFrame.

        Previously to_dict() returned {"data": self.data} directly. Every
        single agent call (DataAgent, AnomalyAgent, ForecastingAgent, the
        Router, and generate_final_response) rebuilds its prompt from
        state.to_dict(), which gets JSON-dumped in prompts.py. For
        anything beyond a toy CSV, that means the entire dataset gets
        stringified into every LLM call, every iteration -- expensive and
        pointless, since state.data_summary already carries the useful
        structural info (rows, columns, dtypes, missing values).
        """
        if self.data is None:
            return None

        preview = {}
        if hasattr(self.data, "shape"):
            preview["shape"] = list(self.data.shape)
        if hasattr(self.data, "columns"):
            preview["columns"] = list(self.data.columns)
        return preview

    def to_dict(self):
        """
        Converts the state to a dictionary for easy readability.
        This also allows the state to be included in the context in an easier
        manner
        """
        return {"user_request": self.user_request,
                "data": self._data_preview(),  # CHANGED: was `self.data`
                "data_summary": self.data_summary,
                "target_column": self.target_column,  # CHANGED: now included
                "forecast": self.forecast,
                "forecast_metrics": self.forecast_metrics,
                "anomalies": self.anomalies,
                "anomaly_analysis": self.anomaly_analysis,
                "current_agent": self.current_agent,
                "completed_agents": self.completed_agents,
                "errors": self.errors}
