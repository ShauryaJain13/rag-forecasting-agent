from agent.base_agent import BaseAgent
from forecasting.evaluation import best_model_walk_forward
from forecasting.models import NaiveModel, HoltWinters, XGBoost, SARIMAModel


class ForecastingAgent(BaseAgent):
    """
    Agent responsible for selecting forecasting models,
    evaluating them, and generating forecasts.
    """

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(
            name="forecasting_agent", llm=llm, tools=tools,
            # system_prompt=(
            #     "You are a forecasting agent. "
            #     "You are responsible for selecting "
            #     "appropriate forecasting models, "
            #     "evaluating them, generating forecasts, "
            #     "and interpreting the results."),
            system_prompt=("""
You are the Forecasting Agent.
Forecast the requested target using the dataset and relevant RAG context.

Consider:
- target and date columns
- relevant dataset columns as covariates
- available future covariates
- forecast horizon
- relevant information from RAG

Use covariates when their future values are available. Do not invent
future values.

Use RAG information such as documented trends, seasonality, or domain
knowledge when relevant. Do not blindly override the historical data.

Select an appropriate forecasting model and produce the forecast.

Return:
- selected model
- forecast values
- covariates used
- brief explanation of how RAG context affected the forecast
"""),
            prompt_builder=prompt_builder, memory=memory)

    # def run(self, task, state):
    #     """
    #     Run the forecasting process.
    #     """
    #     if state.data is None:
    #         raise ValueError("Forecasting cannot begin because no data"
    #                          "is available.")

    #     data = self._get_series(state)
    #     horizon = getattr(state, "forecast_horizon", 7)
    #     best_result, results = self.select_model(data, state)
    #     state.forecast_metrics = results
    #     state.selected_model = (best_result["model"])

    #     model_class = (best_result["model_class"])
    #     model = model_class()
    #     prediction = self.forecast(model, data, horizon)

    #     state.forecast = prediction
    #     state.mark_agent_complete(self.name)
    #     return prediction

    def run(self, task, state):
        """
        Running the forecasting agent
        """
        if state.data is None:
            raise ValueError("No dataset available.")

        if state.target_column is None:
            raise ValueError("No target column specified.")

        horizon = state.forecast_horizon

        target = state.target_column
        data = state.data
        y = data[target]
        covariates = self._get_covariates(state)

        best_result, results = self.select_model(y, state, covariates)

        state.forecast_metrics = results
        state.selected_model = best_result["model"]

        prediction = self.forecast(best_result, y, covariates, state, horizon)

        state.forecast = prediction
        state.mark_agent_complete(self.name)

        return prediction

    def select_model(self, data, state, covariates=None):
        """
        Evaluate candidate forecasting models using
        walk-forward validation and select the best one.
        """
        # models = self._get_potential_models(state.data_summary)
        # train_size = int(0.8 * len(data))
        # horizon = getattr(state, "forecast_horizon", 7)

        # step = horizon
        # best_result, results = (best_model_walk_forward(models, data,
        #                                                 train_size, horizon,
        #                                                 step))

        # return best_result, results

        models = self._get_potential_models(state.data_summary)

        train_size = int(0.8 * len(data))
        horizon = state.forecast_horizon

        best_result, results = best_model_walk_forward(models, data,
                                                       covariates, train_size,
                                                       horizon)

        return best_result, results

    def forecast(self, result, data, covariates, state, horizon):
        """
        Fit the selected model on all available data
        and generate the requested forecast.
        """
        # model.fit(data)
        # prediction = model.predict(horizon)
        # return prediction
        model = result["model_class"]()

        if result["model"] == "XGBoost":
            model.fit(data, covariates=covariates)
            prediction = model.predict(horizon, covariates=covariates)
        else:
            model.fit(data)
            prediction = model.predict(horizon)

        return prediction

    def _get_potential_models(self, summary):
        """
        Return the forecasting model classes available
        to the agent.
        """
        models = [NaiveModel, HoltWinters, XGBoost, SARIMAModel]
        return models

    def _get_series(self, state):
        """
        Extract the target time series from shared state.
        """
        data = state.data
        if hasattr(state, "target_column"):
            target = state.target_column
            if target is not None:
                return data[target]

        if hasattr(data, "name"):
            return data

        numerical_columns = (data.select_dtypes(include="number").columns)

        if len(numerical_columns) == 1:
            return data[numerical_columns[0]]

        raise ValueError("Unable to determine the target forecasting column.")

    def _get_covariates(self, state):
        """
        Retrieves the covariates present of the data
        """
        data = state.data
        target = state.target_column

        covariates = state.covariates

        if not covariates:
            return None

        valid_covariates = [
            column for column in covariates
            if column in data.columns and column != target
        ]

        if not valid_covariates:
            return None

        return data[valid_covariates]
