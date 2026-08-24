from agent.base_agent import BaseAgent

from forecasting.evaluation import (
    best_model_walk_forward
)

from forecasting.models import (
    NaiveModel,
    HoltWinters,
    XGBoost
)


class ForecastingAgent(BaseAgent):
    """
    Agent responsible for selecting forecasting models,
    evaluating them, and generating forecasts.
    """

    def __init__(
        self,
        llm,
        tools,
        prompt_builder,
        memory
    ):
        super().__init__(
            name="Forecasting Agent",
            llm=llm,
            tools=tools,
            system_prompt=(
                "You are a forecasting agent. "
                "You are responsible for selecting "
                "appropriate forecasting models, "
                "evaluating them, generating forecasts, "
                "and interpreting the results."
            ),
            prompt_builder=prompt_builder,
            memory=memory
        )

    def run(self, task, state):
        """
        Run the forecasting process.
        """

        if state.data is None:
            raise ValueError(
                "Forecasting cannot begin because "
                "no data is available."
            )

        # --------------------------------
        # Get forecasting series
        # --------------------------------

        data = self._get_series(state)

        # --------------------------------
        # Determine forecast horizon
        # --------------------------------

        horizon = getattr(
            state,
            "forecast_horizon",
            7
        )

        # --------------------------------
        # Select and evaluate model
        # --------------------------------

        best_result, results = self.select_model(
            data,
            state
        )

        # Store evaluation results
        state.forecast_metrics = results

        # Store selected model
        state.selected_model = (
            best_result["model"]
        )

        # --------------------------------
        # Create final model
        # --------------------------------

        model_class = (
            best_result["model_class"]
        )

        model = model_class()

        # --------------------------------
        # Generate forecast
        # --------------------------------

        prediction = self.forecast(
            model,
            data,
            horizon
        )

        state.forecast = prediction

        # --------------------------------
        # Mark completion
        # --------------------------------

        state.mark_agent_complete(
            self.name
        )

        return prediction

    def select_model(self, data, state):
        """
        Evaluate candidate forecasting models using
        walk-forward validation and select the best one.
        """

        models = self._get_potential_models(
            state.data_summary
        )

        # Use 80% of the data as the initial training set
        train_size = int(
            0.8 * len(data)
        )

        horizon = getattr(
            state,
            "forecast_horizon",
            7
        )

        # Move forward by one forecast horizon
        step = horizon

        best_result, results = (
            best_model_walk_forward(
                models,
                data,
                train_size,
                horizon,
                step
            )
        )

        return best_result, results

    def forecast(
        self,
        model,
        data,
        horizon
    ):
        """
        Fit the selected model on all available data
        and generate the requested forecast.
        """

        model.fit(data)

        prediction = model.predict(
            horizon
        )

        return prediction

    def _get_potential_models(self, summary):
        """
        Return the forecasting model classes available
        to the agent.
        """

        models = [
            NaiveModel,
            HoltWinters,
            XGBoost
        ]

        return models

    def _get_series(self, state):
        """
        Extract the target time series from shared state.
        """

        data = state.data

        # If DataAgent already identified the target
        if hasattr(state, "target_column"):
            target = state.target_column

            if target is not None:
                return data[target]

        # If state.data is already a Series
        if hasattr(data, "name"):
            return data

        # If there is only one numerical column
        numerical_columns = (
            data.select_dtypes(
                include="number"
            ).columns
        )

        if len(numerical_columns) == 1:
            return data[
                numerical_columns[0]
            ]

        raise ValueError(
            "Unable to determine the target "
            "forecasting column."
        )