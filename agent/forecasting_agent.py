# import pandas as pd
# from agent.base_agent import BaseAgent
# from forecasting.evaluation import best_model_walk_forward
# from forecasting.models import NaiveModel, HoltWinters, XGBoost, SARIMAModel


# class ForecastingAgent(BaseAgent):
#     """
#     Agent responsible for selecting forecasting models,
#     evaluating them, and generating forecasts.
#     """

#     def __init__(self, llm, tools, prompt_builder, memory):
#         super().__init__(
#             name="forecasting_agent", llm=llm, tools=tools,
#             system_prompt=("""
# You are the Forecasting Agent.
# Forecast the requested target using the dataset and relevant RAG context.

# Consider:
# - target and date columns
# - relevant dataset columns as covariates
# - available future covariates
# - forecast horizon
# - relevant information from RAG

# Use covariates when their future values are available. Do not invent
# future values.

# Use RAG information such as documented trends, seasonality, or domain
# knowledge when relevant. Do not blindly override the historical data.

# Select an appropriate forecasting model and produce the forecast.

# Return:
# - selected model
# - forecast values
# - covariates used
# - brief explanation of how RAG context affected the forecast
# """),
#             prompt_builder=prompt_builder, memory=memory)

#     def run(self, task, state):
#         """
#         Running the forecasting agent
#         """
#         if state.data is None:
#             raise ValueError("No dataset available.")

#         if state.target_column is None:
#             raise ValueError("No target column specified.")

#         horizon = state.forecast_horizon

#         target = state.target_column
#         data = state.data
#         y = data[target]
#         covariates = self._get_covariates(state)

#         best_result, results = self.select_model(y, state, covariates)

#         state.forecast_metrics = results
#         state.selected_model = best_result["model"]

#         prediction = self.forecast(best_result, y, covariates, state,
# horizon)

#         state.forecast = prediction
#         state.mark_agent_complete(self.name)

#         return prediction

#     def select_model(self, data, state, covariates=None):
#         """
#         Evaluate candidate forecasting models using
#         walk-forward validation and select the best one.
#         """
#         models = self._get_potential_models(state.data_summary)

#         train_size = int(0.8 * len(data))
#         horizon = state.forecast_horizon

#         best_result, results, failures = best_model_walk_forward(
#             models, data, covariates, train_size, horizon)

#         if failures:
#             state.add_warning({"component": "forecasting_agent",
#                                "warning": ("Some models could not be
# evaluated"
#                                            f"and were skipped: {failures}")})

#         return best_result, results

#     def forecast(self, result, data, covariates, state, horizon):
#         """
#         Fit the selected model on all available data
#         and generate the requested forecast.
#         """
#         model = result["model_class"]()

#         future_covariates = None
#         if covariates is not None:
#             future_covariates = self._get_future_covariates(
#                 covariates, horizon, state)

#         model.fit(data, covariates=covariates)
#         prediction = model.predict(horizon, covariates=future_covariates)

#         return prediction

#     def _get_potential_models(self, summary):
#         """
#         Return the forecasting model classes available
#         to the agent.
#         """
#         return [NaiveModel, HoltWinters, XGBoost, SARIMAModel]

#     def _get_series(self, state):
#         """
#         Extract the target time series from shared state.
#         """
#         data = state.data
#         if hasattr(state, "target_column"):
#             target = state.target_column
#             if target is not None:
#                 return data[target]

#         if hasattr(data, "name"):
#             return data

#         numerical_columns = (data.select_dtypes(include="number").columns)

#         if len(numerical_columns) == 1:
#             return data[numerical_columns[0]]

#         raise ValueError("Unable to determine the target forecasting
# column.")

#     def _get_covariates(self, state):
#         """
#         Retrieves the *historical* covariate data for model fitting.

#         CHANGED: was reading state.forecast_covariates (a misnomer --
#         that field was actually holding the identified historical
#         covariate column *names*). Historical covariate column names
#         belong in state.covariates; state.forecast_covariates is
#         reserved for actual future covariate *values*, used in
#         _get_future_covariates below.
#         """
#         data = state.data
#         target = state.target_column
#         covariates = state.covariates
#         if not covariates:
#             return None

#         valid_covariates = [column for column in covariates
#                             if column in data.columns and column != target]

#         if not valid_covariates:
#             return None
#         return data[valid_covariates]

#     def _get_future_covariates(self, covariates, horizon, state):
#         """
#         Determine covariate values to use during prediction.

#         CHANGED: now checks state.forecast_covariates first -- if
#         future covariate values have actually been supplied (e.g. by
#         the user, or a future upstream agent), those are used
#         directly. Only when nothing has been supplied does this fall
#         back to repeating the last observed historical row for the
#         entire forecast horizon, which is a real modeling assumption
#         (that covariates stay constant into the future) rather than a
#         neutral default -- still recorded as a warning either way.
#         """
#         future = state.forecast_covariates

#         if future is not None and len(future) > 0:
#             future_df = (future if isinstance(future, pd.DataFrame)
#                          else pd.DataFrame(future))

#             if len(future_df) < horizon:
#                 state.add_warning({
#                     "component": "forecasting_agent",
#                     "warning": (
#                         f"Only {len(future_df)} future covariate rows "
#                         f"were provided for a {horizon}-step forecast; "
#                         "the remaining steps repeat the last provided "
#                         "row.")})
#                 last_row = future_df.iloc[[-1]]
#                 padding = pd.concat(
#                     [last_row] * (horizon - len(future_df)),
#                     ignore_index=True)
#                 future_df = pd.concat([future_df, padding],
#                                       ignore_index=True)
#             elif len(future_df) > horizon:
#                 future_df = future_df.iloc[:horizon].reset_index(drop=True)

#             missing_columns = [c for c in covariates.columns
#                                if c not in future_df.columns]
#             if missing_columns:
#                 state.add_warning({
#                     "component": "forecasting_agent",
#                     "warning": (
#                         f"Future values were not provided for "
#                         f"{missing_columns}; the forecast assumes "
#                         "these stay constant at their last observed "
#                         "value for the entire forecast horizon.")})
#                 last_row = covariates.iloc[[-1]][missing_columns]
#                 fallback = pd.concat([last_row] * horizon,
#                                      ignore_index=True)
#                 future_df = pd.concat(
#                     [future_df.reset_index(drop=True), fallback], axis=1)

#             return future_df[covariates.columns]

#         state.add_warning({
#             "component": "forecasting_agent",
#             "warning": ("Future covariate values were not provided; "
#                         "the forecast assumes covariates stay constant "
#                         "at their last observed values for the entire "
#                         "forecast horizon.")})

#         last_row = covariates.iloc[[-1]]
#         future = pd.concat([last_row] * horizon, ignore_index=True)
#         return future

from functools import partial

import pandas as pd
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
        models = self._get_potential_models(state)

        train_size = int(0.8 * len(data))
        horizon = state.forecast_horizon

        # CHANGED: determine whether evaluation should use real future
        # covariate values or the same "hold constant" assumption the
        # final forecast uses -- see _determine_covariate_mode below.
        covariate_mode = self._determine_covariate_mode(
            covariates, horizon, state)

        best_result, results, failures = best_model_walk_forward(
            models, data, covariates, train_size, horizon,
            covariate_mode=covariate_mode)

        if failures:
            state.add_warning({"component": "forecasting_agent",
                               "warning": ("Some models could not be evaluated"
                                           f"and were skipped: {failures}")})

        # CHANGED: added -- make it visible in state (and therefore to
        # the final response) whenever the reported metrics reflect the
        # honest "we don't know future covariates" assumption, so a
        # data scientist reading the output isn't misled into thinking
        # the evaluation numbers benefited from information the real
        # forecast doesn't have.
        if covariates is not None and covariate_mode == "persist":
            state.add_warning({
                "component": "forecasting_agent",
                "warning": (
                    "Future covariate values are not available, so "
                    "both the reported evaluation metrics and the "
                    "final forecast assume covariates stay constant "
                    "at their last observed values -- the metrics are "
                    "not inflated by access to real future covariate "
                    "values.")})

        return best_result, results

    def _determine_covariate_mode(self, covariates, horizon, state):
        """
        Decide whether walk-forward evaluation should see real,
        historical future covariate values for each test fold
        ("known"), or the same last-observed-value assumption the
        final forecast actually uses ("persist").

        CHANGED: added. Previously walk_forward_validation always used
        the real historical covariate slice for every test fold,
        unconditionally -- which silently answers a different,
        materially easier question ("how good is this model if we
        already know the future?") than what forecast() actually does
        at serving time. Only use "known" mode when genuine future
        covariate values have actually been supplied for the full
        forecast horizon (state.forecast_covariates) -- meaning the
        real forecast will *also* get to use them, so evaluating with
        real values is a fair comparison. Otherwise default to
        "persist", matching the fallback in _get_future_covariates.
        """
        if covariates is None:
            return "persist"

        future = state.forecast_covariates
        if future is not None and len(future) >= horizon:
            return "known"

        return "persist"

    def forecast(self, result, data, covariates, state, horizon):
        """
        Fit the selected model on all available data
        and generate the requested forecast.
        """
        model = result["model_class"]()

        future_covariates = None
        if covariates is not None:
            future_covariates = self._get_future_covariates(
                covariates, horizon, state)

        model.fit(data, covariates=covariates)
        prediction = model.predict(horizon, covariates=future_covariates)

        return prediction

    def _get_potential_models(self, state):
        """
        Return the forecasting model classes/factories available to
        the agent, matched to the dataset's own sampling frequency.

        CHANGED: was `_get_potential_models(self, summary)`, returning
        bare HoltWinters/SARIMAModel classes that always used their
        hardcoded default seasonal period (7, i.e. weekly) no matter
        what the dataset's actual granularity was. For this monthly
        rainfall dataset that meant modeling an annual seasonal cycle
        as if it repeated every 7 rows instead of every 12. Both are
        now built via functools.partial with state.seasonal_period
        (set by DataAgent from the inferred sampling frequency,
        falling back to 7 if it can't be determined), with `.name`
        attached manually so the rest of the pipeline -- which reads
        `model_class.name` and calls `model_class()` with no arguments
        -- keeps working unchanged.
        """
        seasonal_period = max(2, state.seasonal_period or 7)

        holt_winters_factory = partial(HoltWinters,
                                       seasonality=seasonal_period)
        holt_winters_factory.name = HoltWinters.name

        sarima_factory = partial(SARIMAModel,
                                 seasonal_order=(1, 1, 1, seasonal_period))
        sarima_factory.name = SARIMAModel.name

        return [NaiveModel, holt_winters_factory, XGBoost, sarima_factory]

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
        Retrieves the historical covariate data for model fitting.
        """
        data = state.data
        target = state.target_column
        covariates = state.covariates
        if not covariates:
            return None

        valid_covariates = [column for column in covariates
                            if column in data.columns and column != target]

        if not valid_covariates:
            return None
        return data[valid_covariates]

    def _get_future_covariates(self, covariates, horizon, state):
        """
        Determine covariate values to use during prediction.
        """
        future = state.forecast_covariates

        if future is not None and len(future) > 0:
            future_df = (future if isinstance(future, pd.DataFrame)
                         else pd.DataFrame(future))

            if len(future_df) < horizon:
                state.add_warning({
                    "component": "forecasting_agent",
                    "warning": (
                        f"Only {len(future_df)} future covariate rows "
                        f"were provided for a {horizon}-step forecast; "
                        "the remaining steps repeat the last provided "
                        "row.")})
                last_row = future_df.iloc[[-1]]
                padding = pd.concat(
                    [last_row] * (horizon - len(future_df)),
                    ignore_index=True)
                future_df = pd.concat([future_df, padding],
                                      ignore_index=True)
            elif len(future_df) > horizon:
                future_df = future_df.iloc[:horizon].reset_index(drop=True)

            missing_columns = [c for c in covariates.columns
                               if c not in future_df.columns]
            if missing_columns:
                state.add_warning({
                    "component": "forecasting_agent",
                    "warning": (
                        f"Future values were not provided for "
                        f"{missing_columns}; the forecast assumes "
                        "these stay constant at their last observed "
                        "value for the entire forecast horizon.")})
                last_row = covariates.iloc[[-1]][missing_columns]
                fallback = pd.concat([last_row] * horizon,
                                     ignore_index=True)
                future_df = pd.concat(
                    [future_df.reset_index(drop=True), fallback], axis=1)

            return future_df[covariates.columns]

        state.add_warning({
            "component": "forecasting_agent",
            "warning": ("Future covariate values were not provided; "
                        "the forecast assumes covariates stay constant "
                        "at their last observed values for the entire "
                        "forecast horizon.")})

        last_row = covariates.iloc[[-1]]
        future = pd.concat([last_row] * horizon, ignore_index=True)
        return future
