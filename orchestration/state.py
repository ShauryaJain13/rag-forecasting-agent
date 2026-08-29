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
        self.data_frequency = None
        self.seasonal_period = None

        self.forecast_covariates = []
        self.covariates = []
        self.ignore_columns = []

        self.forecast = None
        self.forecast_metrics = None
        self.forecast_horizon = 7
        self.selected_model = None

        self.anomalies = []
        self.anomaly_analysis = None

        self.current_agent = None
        self.completed_agents = []
        self.agent_plan = []

        self.final_response = None
        self.errors = []
        self.warnings = []

        self.retrieved_documents = []
        self.retrieval_query = None
        self.sources = []
        self.rag_context = None

    def add_error(self, error):
        """
        Adds the error to the current state
        """
        self.errors.append(error)

    def add_warning(self, warning):
        """
        Adds a warning to the current state- a place where the system
        made an assumption rather than failed
        """
        self.warnings.append(warning)

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
                "data_frequency": self.data_frequency,
                "seasonal_period": self.seasonal_period,

                "forecast_covariates": self.forecast_covariates,
                "covariates": self.covariates,
                "ignore_columns": self.ignore_columns,

                "forecast": self.forecast,
                "forecast_metrics": self.forecast_metrics,
                "forecast_horizon": self.forecast_horizon,
                "selected_model": self.selected_model,

                "anomalies": self.anomalies,
                "anomaly_analysis": self.anomaly_analysis,

                "current_agent": self.current_agent,
                "completed_agents": self.completed_agents,
                "agent_plan": self.agent_plan,

                "errors": self.errors,
                "warnings": self.warnings,

                "retrieved_documents": self.retrieved_documents,
                "retrieval_query": self.retrieval_query,
                "rag_context": self.rag_context,
                "sources": self.sources,
                "final_response": self.final_response}
