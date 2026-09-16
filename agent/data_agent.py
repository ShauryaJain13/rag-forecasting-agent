import json
import re

import pandas as pd

from agent.base_agent import BaseAgent


class DataAgent(BaseAgent):
    """
    Agent responsible for loading and understanding the dataset:
    identifying the target column, reconstructing a usable date/time
    column from whatever shape the dataset provides it in, historical
    covariates, columns to ignore, and the forecast horizon.
    """

    _UNIT_TO_DAYS = {"hour": 1 / 24, "hours": 1 / 24,
                     "day": 1, "days": 1,
                     "week": 7, "weeks": 7,
                     "month": 30.44, "months": 30.44,
                     "year": 365.25, "years": 365.25,
                     }

    _HORIZON_PATTERN = re.compile(
        r"(\d+)\s*(hours?|days?|weeks?|months?|years?)\b", re.IGNORECASE)

    _DATE_TIME_ROLES = ("timestamp", "date", "time",
                        "year", "month", "day",
                        "hour", "minute", "second")

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

            "Once the dataset is loaded and analyzed, identify:\n\n"

            "1. TARGET COLUMN -- the column to forecast. Use the "
            "user's explicitly stated target if they named one. "
            "Otherwise, if there is exactly one numerical column that "
            "isn't clearly a covariate or a date/time part, use "
            "that.\n\n"

            "2. DATE/TIME ROLES -- datasets represent time in "
            "different shapes. Identify which columns, if any, "
            "correspond to each of these roles, using ONLY the exact "
            "column names present in the dataset (never invent a "
            "name). Leave a role null if no column fills it. Common "
            "shapes you will see:\n"
            "   - A single column that is already a full date or "
            "timestamp (e.g. 'Date', 'Timestamp') -> set "
            "\"timestamp\".\n"
            "   - Separate 'Year'/'Month'/'Day' columns (and "
            "optionally 'Hour'/'Minute'/'Second' for sub-daily data) "
            "-> set the matching individual role fields.\n"
            "   - A calendar date column plus a separate time-of-day "
            "column, where the same date repeats across multiple rows "
            "at different times -> set \"date\" and \"time\".\n"
            "   - No usable date/time information at all -> leave "
            "every role null; rows will be treated as sequential "
            "periods in their existing order.\n\n"

            "3. COVARIATES -- other columns that could serve as "
            "useful historical exogenous variables for forecasting "
            "the target (e.g. a promotion flag, holiday indicator, "
            "price, or another measured driver). Do not include "
            "columns already used as a date/time role.\n\n"

            "4. IGNORE COLUMNS -- any remaining columns that are "
            "neither the target, a covariate, nor a date/time role "
            "(e.g. an ID column, an index column, free-text notes).\n\n"

            "5. REQUESTED HORIZON -- if the user specified how far "
            "ahead to forecast (e.g. 'next 3 days', 'the coming 6 "
            "months', 'forecast 10 periods'), extract exactly the "
            "number and unit they used. Use unit \"periods\" if they "
            "gave a count with no time unit. Do NOT convert units or "
            "do any arithmetic yourself -- just report what they "
            "said. Use null for count and unit if no horizon was "
            "specified.\n\n"

            "When you are finished, respond with ONLY the following "
            "JSON object as plain assistant text -- never as a tool "
            "or function call, and with no markdown and no "
            "explanation before or after it:\n\n"
            "{\n"
            '  "target_column": "... or null",\n'
            '  "date_time_roles": {\n'
            '    "timestamp": "... or null",\n'
            '    "date": "... or null",\n'
            '    "time": "... or null",\n'
            '    "year": "... or null",\n'
            '    "month": "... or null",\n'
            '    "day": "... or null",\n'
            '    "hour": "... or null",\n'
            '    "minute": "... or null",\n'
            '    "second": "... or null"\n'
            "  },\n"
            '  "covariates": ["...", "..."],\n'
            '  "ignore_columns": ["...", "..."],\n'
            '  "requested_horizon": {"count": <int or null>, '
            '"unit": "hours|days|weeks|months|years|periods or null"},\n'
            '  "data_summary": "..."\n'
            "}\n\n"
            "Do not invent or call any tool that isn't in your "
            "available tools list, and do not wrap this JSON in a "
            "tool invocation of any kind.")

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

        llm_target = self._validate_column(state, parsed.get("target_column"))
        explicit_target = None
        if state.data is not None and hasattr(state.data, "columns"):
            explicit_target = self._extract_target(
                task, state.user_request, state.data.columns)

        state.target_column = (llm_target or explicit_target
                               or state.target_column)

        state.covariates = (self._validate_columns(state,
                                                   parsed.get("covariates"))
                            or state.covariates)
        llm_ignore = self._validate_columns(state,
                                            parsed.get("ignore_columns"))
        if llm_ignore:
            state.ignore_columns = list(set(state.ignore_columns)
                                        | set(llm_ignore))

        state.data_summary = parsed.get("data_summary") or state.data_summary

        self._build_datetime_column(state, parsed.get("date_time_roles"))
        self._infer_frequency(state)
        self._apply_requested_horizon(task, state,
                                      parsed.get("requested_horizon"))

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

    def _validate_column(self, state, column_name):
        """
        Resolve an LLM-provided column name against the real dataset
        columns, case-insensitively. Returns None if it doesn't
        correspond to an actual column (LLMs occasionally invent or
        slightly misname columns).
        """
        if (column_name is None or state.data is None
                or not hasattr(state.data, "columns")):
            return None

        columns_lookup = {column.lower(): column
                          for column in state.data.columns}
        return columns_lookup.get(str(column_name).lower())

    def _validate_columns(self, state, column_names):
        """
        Same as _validate_column, but for a list. Silently drops any
        entries that don't resolve to a real column.
        """
        if not column_names:
            return None

        resolved = [self._validate_column(state, name)
                    for name in column_names]
        resolved = [name for name in resolved if name is not None]
        return resolved or None

    def _extract_target(self, task, request, columns):
        """
        Fallback only: extract a target column the user named in free
        text, used when the LLM's own target_column didn't resolve to
        a real column. Requires a column name to appear NEAR an
        actual target/forecast/predict cue (within 50 characters),
        not just anywhere in the message.
        """
        text = f"{request} {task}"
        text_lower = text.lower()

        cue_pattern = re.compile(
            r"\btarget(?:\s*column|\s*variable)?\b|\bforecast\b|\bpredict\b",
            re.IGNORECASE)
        cue_ends = [match.end() for match in cue_pattern.finditer(text_lower)]
        if not cue_ends:
            return None

        candidates = []
        for column in columns:
            pattern = re.compile(r"\b" + re.escape(column.lower()) + r"\b")
            for match in pattern.finditer(text_lower):
                distances = [match.start() - cue for cue in cue_ends
                             if match.start() >= cue]
                if not distances:
                    continue

                distance = min(distances)
                if distance > 50:
                    continue

                candidates.append((distance, -len(column), column))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def _extract_file_path(self, task, request):
        """
        Extract CSV filename from the user's request.
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

    def _resolve_role_columns(self, state, roles):
        """
        Validate the LLM's date_time_roles against real dataset
        columns, dropping any role that doesn't resolve.
        """
        if not roles or state.data is None:
            return {}

        resolved = {}
        for role in self._DATE_TIME_ROLES:
            column = self._validate_column(state, roles.get(role))
            if column is not None:
                resolved[role] = column
        return resolved

    def _build_datetime_column(self, state, roles):
        """
        Construct a single usable datetime column from whatever
        date/time role columns the LLM identified. Handles:
        - a single ready-made timestamp/date column ("timestamp")
        - a calendar date column plus a separate time-of-day column
          ("date" + "time") -- e.g. multiple readings per day
        - date parts split across separate columns ("year"/"month"/
          "day"/"hour"/"minute"/"second"), any subset of them
        - no date/time information at all (does nothing; downstream
          code already handles state.date_column staying None)

        Consumed columns are recorded in state.ignore_columns so they
        can never be mistaken for numeric covariates downstream, even
        if the LLM's covariates list mistakenly included one of them.
        """
        resolved = self._resolve_role_columns(state, roles)
        if not resolved:
            return

        try:
            if "timestamp" in resolved:
                # Already a single, real column -- use it directly
                # rather than duplicating it into a synthetic one.
                parsed_dates = pd.to_datetime(
                    state.data[resolved["timestamp"]], errors="coerce")
                if parsed_dates.isna().any():
                    return
                state.date_column = resolved["timestamp"]
                return

            if "date" in resolved:
                date_part = pd.to_datetime(state.data[resolved["date"]],
                                           errors="coerce")
                if "time" in resolved:
                    combined = pd.to_datetime(
                        date_part.dt.strftime("%Y-%m-%d") + " " +
                        state.data[resolved["time"]].astype(str),
                        errors="coerce")
                else:
                    combined = date_part

            elif "year" in resolved:
                combined = pd.to_datetime(dict(
                    year=state.data[resolved["year"]],
                    month=state.data.get(resolved.get("month"), 1),
                    day=state.data.get(resolved.get("day"), 1),
                    hour=state.data.get(resolved.get("hour"), 0),
                    minute=state.data.get(resolved.get("minute"), 0),
                    second=state.data.get(resolved.get("second"), 0),
                ), errors="coerce")

            else:
                return

            if combined.isna().any():
                return

            state.data["__inferred_date__"] = combined
            state.date_column = "__inferred_date__"

            consumed = list(resolved.values())
            state.ignore_columns = list(set(state.ignore_columns)
                                        | set(consumed))
            state.add_warning({
                "component": "data_agent",
                "warning": (f"Reconstructed a date/time column from "
                            f"{consumed}.")})

        except Exception as e:
            state.add_warning({
                "component": "data_agent",
                "warning": (f"Could not build a date/time column from "
                            f"{list(resolved.values())}: {e}")})

    def _infer_frequency(self, state):
        """
        Label the dataset's sampling frequency and pick a matching
        seasonal period for HoltWinters/SARIMA, from the median gap
        between timestamps in state.date_column.
        """
        interval_days = self._infer_data_interval_days(state)
        if interval_days is None:
            return

        if interval_days <= 0.1:
            label, period = "sub-hourly", 24
        elif interval_days <= 3 / 24:
            label, period = "hourly", 24
        elif interval_days <= 3:
            label, period = "daily", 7
        elif interval_days <= 10:
            label, period = "weekly", 52
        elif interval_days <= 45:
            label, period = "monthly", 12
        elif interval_days <= 120:
            label, period = "quarterly", 4
        else:
            label, period = "yearly", 1

        state.data_frequency = label
        state.seasonal_period = period

    def _extract_horizon_request(self, text):
        """
        Fallback only: find a "<count> <unit>" duration phrase in
        free text via regex, used when the LLM's own requested_horizon
        was missing or unusable.
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

    def _apply_requested_horizon(self, task, state, requested_horizon):
        """
        Convert the (count, unit) duration into a period count
        matching the dataset's own sampling frequency. Prefers the
        LLM's requested_horizon; falls back to a regex extraction
        from the raw text only if the LLM gave nothing usable.
        """
        count = requested_horizon.get("count") if requested_horizon else None
        unit = requested_horizon.get("unit") if requested_horizon else None

        if count is None:
            text = f"{state.user_request} {task}"
            fallback = self._extract_horizon_request(text)
            if fallback is None:
                return
            count, unit = fallback

        if not unit or unit == "periods":
            state.forecast_horizon = max(1, int(count))
            return

        unit_key = str(unit).lower()
        if unit_key not in self._UNIT_TO_DAYS:
            unit_key = unit_key.rstrip("s")

        if unit_key not in self._UNIT_TO_DAYS:
            state.forecast_horizon = max(1, int(count))
            return

        requested_days = count * self._UNIT_TO_DAYS[unit_key]
        interval_days = self._infer_data_interval_days(state)

        if interval_days is None:
            state.forecast_horizon = max(1, int(count))
            state.add_warning({
                "component": "data_agent",
                "warning": (
                    f"Could not infer the dataset's sampling frequency; "
                    f"interpreting '{count} {unit}' as {count} forecast "
                    "periods directly rather than converting units.")
            })
            return

        horizon = max(1, round(requested_days / interval_days))
        state.forecast_horizon = horizon
