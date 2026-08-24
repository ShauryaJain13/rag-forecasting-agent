class DataAnalyzer:
    """
    Tool for performing basic structural analysis
    of a pandas DataFrame.
    """

    name = "analyze_dataset"

    description = ("Analyze the structure and quality of a dataset. "
                   # CHANGED: added space, was "dataset."Returns"
                   "Returns information about rows, columns, data types, "
                   "missing values, and duplicated rows.")

    def __init__(self, state):
        self.state = state

    def schema(self):
        """
        Schema of the data analysis tool

        CHANGED: wrapped in {"type": "function", "function": {...}} to
        match what Groq's API requires (same issue as anomalies.py).
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

    def execute(self, arguments):
        """
        Perform deterministic dataset analysis.
        """
        data_frame = self.state.data
        if data_frame is None:
            raise ValueError("No dataset is currently loaded.")

        summary = {"rows": len(data_frame),
                   "columns": list(data_frame.columns),
                   "data_types": (data_frame.dtypes.astype(str).to_dict()),
                   "missing_values": (data_frame.isnull().sum().to_dict()),
                   "duplicated_rows": int(data_frame.duplicated().sum())}

        self.state.data_summary = summary
        return summary
