# from agent.base_agent import BaseAgent
# import json


# class DataAgent(BaseAgent):

#     def __init__(self, llm, tools, prompt_builder, memory):
#         super().__init__(
#             name="data_agent",
#             llm=llm,
#             tools=tools,
#             prompt_builder=prompt_builder,
#             memory=memory,
#             system_prompt="You are the Data Agent in a multi-agent time"
#             "series forecasting system.\n\n"

#             "Your job is to load and understand the dataset needed "
#             "to answer the user's request, using the read_csv and "
#             "analyze_dataset tools available to you.\n\n"

#             "Once the dataset is loaded and analyzed, identify:\n"
#             "- the target column to forecast\n"
#             "- the date/time column, if one exists\n"
#             "- any other columns that could serve as useful "
#             "covariates (exogenous variables) for forecasting the "
#             "target -- for example known future values like a "
#             "promotion flag, holiday indicator, or planned price\n\n"

#             "Use the user's explicitly stated target column if they "
#             "named one. If they did not, and there is exactly one "
#             "numerical column that isn't clearly a covariate, use "
#             "that.\n\n"

#             "When you are finished, respond with ONLY the following "
#             "JSON object and nothing else -- no markdown, no "
#             "explanation before or after it:\n\n"
#             "{\n"
#             '  "target_column": "...",\n'
#             '  "date_column": "...",\n'
#             '  "covariates": ["...", "..."],\n'
#             '  "data_summary": "..."\n'
#             "}\n\n"
#             "If a field cannot be determined, use null (or an empty "
#             "list for covariates).")

#     def run(self, task, state):
#         """
#         Running the data_agent
#         """
#         # state.set_current_agent(self.name)

#         # try:
#         #     # Load the dataset
#         #     read_csv = self.tools.get("read_csv")

#         #     if read_csv is None:
#         #         raise ValueError("read_csv tool is not registered.")

#         #     # The filename is currently supplied in the user's request.
#         #     # For your current project, extract it from the task/request.
#         #     file_path = self._extract_file_path(
#         #         task,
#         #         state.user_request
#         #     )

#         #     read_csv.execute({
#         #         "file_path": file_path
#         #     })

#         #     if state.data is None:
#         #         raise ValueError("Dataset could not be loaded.")

#         #     # Target was explicitly supplied by the user.
#         #     target = self._extract_target(
#         #         task,
#         #         state.user_request,
#         #         state.data.columns
#         #     )

#         #     if target is not None:
#         #         state.target_column = target

#         #     # Basic dataset summary
#         #     analyzer = self.tools.get("analyze_dataset")

#         #     if analyzer is not None:
#         #         state.data_summary = analyzer.execute({})

#         #     state.mark_agent_complete(self.name)

#         #     return state

#         # except Exception as e:
#         #     state.add_error({
#         #         "agent": self.name,
#         #         "error": str(e)
#         #     })
#         #     state.current_agent = None
#         #     return state

#         # If user explicitly provided a target, preserve it.

#         result = super().run(task, state)

#         parsed = {}
#         try:
#             parsed = json.loads(result)
#         except (json.JSONDecodeError, TypeError):
#             state.add_error({"agent": self.name,
#                              "error": ("DataAgent returned invalid JSON:"
#                                        "unable to determine "
#                                        "target/date/covariate columns"
#                                        "from the LLM's response.")})

#         # if state.target_column is not None or :
#         #     target = self._extract_target(task, state.user_request,
#         #                                   state.data.columns)

#         #     if target:
#         #         state.target_column = target

#         # Do NOT let the LLM overwrite an explicitly supplied target.

#         if state.data is not None and hasattr(state.data, "columns"):
#             explicit_target = self._extract_target(
#                 task, state.user_request, state.data.columns)

#         try:
#             state.target_column = (explicit_target or
#                                    parsed.get("target_column")
#                                    or state.target_column)

#             # if state.target_column is None:
#             # try:
#             # parsed = json.loads(result)
#             # state.target_column = parsed.get("target_column")
#             state.date_column = parsed.get("date_column")
#             state.forecast_covariates = parsed.get("covariates", [])
#             state.data_summary = parsed.get("data_summary")
#         except (json.JSONDecodeError, TypeError):
#             state.add_error({
#                 "agent": self.name,
#                 "error": "DataAgent returned invalid JSON."
#             })

#         state.mark_agent_complete(self.name)
#         return result

#     def _extract_target(self, task, request, columns):
#         """
#         Extract target column from the user's request.
#         """

#         text = f"{request} {task}".lower()

#         for column in columns:
#             if column.lower() in text:
#                 if "target" in text:
#                     return column

#         return None

#     def _extract_file_path(self, task, request):
#         """
#         Extract CSV filename from the user's request.
#         """

#         text = f"{request} {task}"

#         start = text.find('"')
#         end = text.find('"', start + 1)

#         if start != -1 and end != -1:
#             return text[start + 1:end]

#         # fallback
#         for word in text.split():
#             if word.lower().endswith(".csv"):
#                 return word.strip("'\"., ")

#         raise ValueError("Could not determine the CSV file path.")


import json
import re

import pandas as pd

from agent.base_agent import BaseAgent


class DataAgent(BaseAgent):
    """
    Agent responsible for loading and understanding the dataset:
    identifying the target column, the date/time column, historical
    covariates, and the forecast horizon, using the read_csv and
    analyze_dataset tools.
    """

    # CHANGED: added -- used by _extract_horizon_request/
    # _determine_forecast_horizon
    # to translate a natural-language duration ("7 months", "13 days")
    # into an approximate number of days, so it can be compared against
    # the dataset's own sampling interval.
    _UNIT_TO_DAYS = {
        "day": 1, "days": 1,
        "week": 7, "weeks": 7,
        "month": 30.44, "months": 30.44,
        "year": 365.25, "years": 365.25,
    }

    _HORIZON_PATTERN = re.compile(
        r"(\d+)\s*(days?|weeks?|months?|years?)\b", re.IGNORECASE)

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(
            name="data_agent",
            llm=llm,
            tools=tools,
            prompt_builder=prompt_builder,
            memory=memory,
            system_prompt="You are the Data Agent in a multi-agent time"
            "series forecasting system.\n\n"

            "Your job is to load and understand the dataset needed "
            "to answer the user's request, using the read_csv and "
            "analyze_dataset tools available to you.\n\n"

            "Once the dataset is loaded and analyzed, identify:\n"
            "- the target column to forecast\n"
            "- the date/time column, if one exists\n"
            "- any other columns that could serve as useful historical "
            "covariates (exogenous variables) for forecasting the "
            "target -- for example known drivers like a promotion "
            "flag, holiday indicator, or price\n\n"

            "Use the user's explicitly stated target column if they "
            "named one. If they did not, and there is exactly one "
            "numerical column that isn't clearly a covariate, use "
            "that.\n\n"

            "You do not need to determine the forecast horizon -- "
            "that is handled separately.\n\n"

            "When you are finished, respond with ONLY the following "
            "JSON object and nothing else -- no markdown, no "
            "explanation before or after it:\n\n"
            "{\n"
            '  "target_column": "...",\n'
            '  "date_column": "...",\n'
            '  "covariates": ["...", "..."],\n'
            '  "data_summary": "..."\n'
            "}\n\n"
            "If a field cannot be determined, use null (or an empty "
            "list for covariates).")

    def run(self, task, state):
        """
        Running the data_agent
        """
        result = super().run(task, state)

        parsed = {}
        try:
            parsed = json.loads(self._strip_markdown_fence(result))
        except (json.JSONDecodeError, TypeError):
            state.add_error({
                "agent": self.name,
                "error": ("DataAgent returned invalid JSON: unable to "
                          "determine target/date/covariate columns "
                          "from the LLM's response.")
            })

        explicit_target = None
        if state.data is not None and hasattr(state.data, "columns"):
            explicit_target = self._extract_target(
                task, state.user_request, state.data.columns)

        state.target_column = (explicit_target or
                               parsed.get("target_column")
                               or state.target_column)
        state.date_column = parsed.get("date_column") or state.date_column

        # CHANGED: this used to write into state.forecast_covariates.
        # These are the *historical* covariate columns identified in
        # the loaded dataset (used by forecasting_agent to fit the
        # model), which is what state.covariates is meant to hold.
        # state.forecast_covariates is reserved for actual *future*
        # covariate values, which nothing here determines.
        state.covariates = parsed.get("covariates") or state.covariates

        state.data_summary = parsed.get("data_summary") or state.data_summary

        # CHANGED: added -- determine the forecast horizon (in periods,
        # matching the dataset's own sampling frequency) from phrases
        # like "forecast next 7 months" or "next 13 days" in the
        # user's request. Must run after state.date_column is set
        # above, since it's used to infer the dataset's frequency.
        self._determine_forecast_horizon(task, state)

        state.mark_agent_complete(self.name)
        return result

    def _strip_markdown_fence(self, text):
        """
        Strip ```json / ``` markdown fences the LLM may wrap its JSON
        response in, despite being told not to.
        """
        if not isinstance(text, str):
            return text

        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        return text

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

        NOTE: currently unused -- the LLM is expected to extract the
        file path itself and pass it as the `file_path` argument when
        it calls the read_csv tool.
        """
        text = f"{request} {task}"

        start = text.find('"')
        end = text.find('"', start + 1)

        if start != -1 and end != -1:
            return text[start + 1:end]

        for word in text.split():
            if word.lower().endswith(".csv"):
                return word.strip("'\"., ")

        raise ValueError("Could not determine the CSV file path.")

    def _extract_horizon_request(self, text):
        """
        Find a "<count> <unit>" duration phrase in free text, e.g.
        "forecast the next 7 months" -> (7, "months").

        NOTE: takes the first match found. If a request contains more
        than one count+unit phrase (e.g. "compare against the last 30
        days and forecast the next 7 days"), this may pick up the
        wrong one -- acceptable for now, but worth tightening (e.g.
        anchoring on "next"/"forecast"/"future") if that turns out to
        matter in practice.
        """
        match = self._HORIZON_PATTERN.search(text)
        if not match:
            return None

        count = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        return count, unit

    def _infer_data_interval_days(self, state):
        """
        Estimate the dataset's sampling interval, in days, from the
        median gap between consecutive timestamps in state.date_column.
        Returns None if it can't be determined (no date column, too
        few rows, unparseable dates).
        """
        if state.data is None or state.date_column is None:
            return None

        if state.date_column not in state.data.columns:
            return None

        try:
            dates = pd.to_datetime(state.data[state.date_column],
                                   errors="coerce").dropna().sort_values()
            if len(dates) < 2:
                return None

            median_seconds = dates.diff().dropna().dt.total_seconds().median()
            if not median_seconds or median_seconds <= 0:
                return None

            return median_seconds / 86400

        except Exception:
            return None

    def _determine_forecast_horizon(self, task, state):
        """
        Set state.forecast_horizon from a natural-language duration in
        the user's request, translated into the dataset's own sampling
        frequency (e.g. "next 7 months" over daily data becomes ~213
        daily steps; over monthly data it stays 7 steps).

        If no duration phrase is found, state.forecast_horizon is left
        as-is (its existing value, or the AgentState default of 7).
        """
        text = f"{state.user_request} {task}"
        parsed = self._extract_horizon_request(text)
        if parsed is None:
            return

        count, unit = parsed
        requested_days = count * self._UNIT_TO_DAYS[unit]

        interval_days = self._infer_data_interval_days(state)

        if interval_days is None:
            # Can't determine the dataset's own frequency -- fall back
            # to treating the requested count as the horizon directly.
            # This is correct whenever the requested unit already
            # matches the data's granularity (e.g. "13 days" on daily
            # data), which is the common case, but may be wrong
            # otherwise (e.g. "7 months" on daily data with no usable
            # date column).
            state.forecast_horizon = max(1, count)
            state.add_warning({
                "component": "data_agent",
                "warning": (
                    f"Could not infer the dataset's sampling frequency; "
                    f"interpreting '{count} {unit}' as {count} forecast "
                    "periods directly rather than converting units.")})
            return

        horizon = max(1, round(requested_days / interval_days))
        state.forecast_horizon = horizon
