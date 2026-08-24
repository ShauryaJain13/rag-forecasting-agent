class ForecastTool:
    """
    Tool that performs a forecast using the selected
    forecasting model.

    NOTE: as of the current forecasting_agent.py, this tool is not
    actually wired up anywhere. ForecastingAgent.run() bypasses the LLM
    tool-calling loop entirely (it never calls super().run()) and instead
    deterministically evaluates/fits/predicts using
    forecasting.evaluation + forecasting.models directly. This class is
    fixed below for correctness, but it stays unregistered/unused unless
    you decide to make forecasting LLM-driven through tool calls instead.
    """

    name = "forecast"

    description = ("Generate a forecast for the loaded time-series data. "
                   # CHANGED: added space
                   "Requires a forecasting model and forecast horizon.")

    def __init__(self, state, forecasting_engine):
        self.state = state
        self.forecasting_engine = forecasting_engine

    def schema(self):
        """
        The official schema of the tool

        CHANGED: wrapped in {"type": "function", "function": {...}}.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": ("The forecasting model to use. "
                                            "For example: naive, "
                                            "holt_winters or xgboost.")
                        },
                        "horizon": {
                            "type": "integer",
                            "description": ("Number of future observations "
                                            "to forecast.")
                        }
                    },
                    "required": ["model", "horizon"]
                }
            }
        }

    def execute(self, arguments):
        """
        Executes the tool call to forecast
        """
        model_name = arguments.get("model")
        horizon = arguments.get("horizon")

        if not model_name:
            raise ValueError("A forecasting model is required.")

        if not horizon:
            raise ValueError("A forecast horizon is required.")

        if horizon <= 0:
            raise ValueError("Forecast horizon must be greater than zero.")

        if self.state.data is None:
            raise ValueError("No dataset is loaded.")

        forecast = self.forecasting_engine.forecast(data=self.state.data,
                                                    model_name=model_name,
                                                    horizon=horizon)
        self.state.forecast = forecast
        self.state.selected_model = model_name

        return {"model": model_name,
                "horizon": horizon,
                "forecast": forecast.tolist()}
