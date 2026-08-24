import numpy as np
import pandas as pd


class AnomalyDetectionTool:
    """
    Tool for detecting anomalous observations in a time series.

    The tool performs deterministic statistical analysis.
    The AnomalyAgent decides when and how this tool should be used.
    """

    name = "detect_anomalies"

    description = ("Detect anomalous observations in a numerical time series. "
                   "Supports IQR and Z-score based detection methods.")

    def __init__(self, state):
        self.state = state

    def schema(self):
        """
        Schema exposed to the LLM through the tool registry.

        CHANGED: wrapped in {"type": "function", "function": {...}}.
        Without this wrapper, this schema doesn't match the shape Groq's
        API expects (it matches registry.Tool's wrapped format), so mixing
        this tool into a ToolRegistry alongside properly-wrapped tools
        (Calculator, ReadFile, ReadCSV) would cause a 400 validation error
        the moment this tool got registered and sent to the LLM.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore"],
                            "description": "Method used to detect anomalies."
                        },
                        "threshold": {
                            "type": "number",
                            "description": (
                                "Sensitivity threshold. For zscore this is "
                                "the z-score threshold. For IQR this is "
                                "the IQR multiplier.")
                        }
                    },
                    "required": ["method"]
                }
            }
        }

    def execute(self, arguments):
        """
        Detect anomalies using the requested method.
        """
        if self.state.data is None:
            raise ValueError("No dataset is loaded.")

        method = arguments.get("method", "iqr")
        threshold = arguments.get("threshold")
        series = self._get_series()
        if method == "iqr":
            anomalies = self._iqr(series, threshold)

        elif method == "zscore":
            anomalies = self._zscore(series, threshold)

        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

        self.state.anomalies = anomalies
        return {"method": method,
                "number_of_anomalies": len(anomalies),
                "anomalies": anomalies}

    def _get_series(self):
        """
        Extract the numerical time series from the dataset.

        Currently assumes:
        - A pandas Series, or
        - A DataFrame containing exactly one numerical column.
        """

        data = self.state.data
        if isinstance(data, pd.Series):
            series = data
        elif isinstance(data, pd.DataFrame):
            numerical_columns = (data.select_dtypes(include=np.number).columns)

            if len(numerical_columns) == 0:
                raise ValueError("No numerical columns found in dataset.")

            if len(numerical_columns) > 1:
                target_column = getattr(self.state, "target_column", None)

                if target_column is not None:
                    series = data[target_column]
                else:
                    raise ValueError("Multiple numerical columns found. A "
                                     "target column must be specified.")
            else:
                series = data[numerical_columns[0]]
        else:
            raise TypeError("state.data must be a pandas Series or DataFrame.")

        return series.dropna()

    def _iqr(self, series, threshold=None):
        """
        Detect anomalies using the Interquartile Range method.
        """
        if threshold is None:
            threshold = 1.5
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = ((series < lower_bound) | (series > upper_bound))

        return self._format_anomalies(series, mask, lower_bound, upper_bound)

    def _zscore(self, series, threshold=None):
        """
        Detect anomalies using Z-score.
        """
        if threshold is None:
            threshold = 3.0
        mean = series.mean()
        std = series.std()

        if std == 0:
            return []

        z_scores = ((series - mean) / std)

        mask = np.abs(z_scores) > threshold

        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        return self._format_anomalies(series, mask, lower_bound, upper_bound)

    def _format_anomalies(self, series, mask, lower_bound, upper_bound):
        """
        Convert detected anomalies into a serializable format.
        """

        anomaly_indices = series.index[mask]
        anomalies = []
        for index in anomaly_indices:
            anomalies.append({"timestamp": str(index),
                              "value": float(series.loc[index]),
                              "lower_bound": float(lower_bound),
                              "upper_bound": float(upper_bound)})

        return anomalies