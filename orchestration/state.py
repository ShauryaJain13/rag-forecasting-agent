class AgentState:
    """
    This class monitors the 'state' of the model
    """

    def __init__(self, user_request):
        self.user_request = user_request
        self.data = None
        self.data_summary = None
        self.target_column = None
        self.date_column = None

        self.forecast = None
        self.forecast_metrics = None
        # CHANGED: added -- ForecastingAgent sets state.selected_model,
        # but it was never initialized here nor included in to_dict(),
        # so the Router/final-response generator could never see which
        # model was actually used.
        self.selected_model = None

        self.anomalies = []
        self.anomaly_analysis = None

        self.current_agent = None
        self.completed_agents = []
        self.agent_plan = []

        self.final_response = None
        self.errors = []

        self.retrieved_documents = []
        self.retrieval_query = None
        self.sources = []
        self.rag_context = None

    def add_error(self, error):
        """
        Adds the error to the current state
        """
        self.errors.append(error)

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
        Build a small, JSON-friendly summary of the loaded dataset
        instead of embedding the full DataFrame.
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
        """
        return {"user_request": self.user_request,
                "data": self._data_preview(),
                "data_summary": self.data_summary,
                "target_column": self.target_column,
                "date_column": self.date_column,

                "forecast": self.forecast,
                "forecast_metrics": self.forecast_metrics,
                # CHANGED: added, see __init__ note above.
                "selected_model": self.selected_model,

                "anomalies": self.anomalies,
                "anomaly_analysis": self.anomaly_analysis,

                "current_agent": self.current_agent,
                "completed_agents": self.completed_agents,
                "agent_plan": self.agent_plan,

                "errors": self.errors,

                "retrieved_documents": self.retrieved_documents,
                "retrieval_query": self.retrieval_query,
                "rag_context": self.rag_context,
                "sources": self.sources,
                "final_response": self.final_response}