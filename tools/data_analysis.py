class DataAnalyzer:
    """
    Tool for performing basic structural analysis
    of a pandas DataFrame.
    """

    name = "analyze_dataset"

    description = (
        "Analyze the dataset structure and quality. "
        "Provide the target column if specified by the user."
    )

    def __init__(self, state):
        self.state = state

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": (
                                "The target column specified by the user. "
                                "Use null if no target was specified."
                            )
                        }
                    },
                    "required": []
                }
            }
        }

    def execute(self, arguments):

        data_frame = self.state.data

        if data_frame is None:
            raise ValueError("No dataset is currently loaded.")

        target_column = arguments.get("target_column")

        # If the LLM/user specified a target, validate it.
        if target_column:

            if target_column not in data_frame.columns:
                raise ValueError(
                    f"Target column '{target_column}' does not exist. "
                    f"Available columns: {list(data_frame.columns)}"
                )

            self.state.target_column = target_column

        # Otherwise automatically select the target
        # if there is exactly one numerical column.
        else:

            numerical_columns = data_frame.select_dtypes(
                include="number"
            ).columns.tolist()

            if len(numerical_columns) == 1:
                self.state.target_column = numerical_columns[0]

        summary = {
            "rows": len(data_frame),
            "columns": list(data_frame.columns),
            "data_types": (
                data_frame.dtypes.astype(str).to_dict()
            ),
            "missing_values": (
                data_frame.isnull().sum().to_dict()
            ),
            "duplicated_rows": int(
                data_frame.duplicated().sum()
            )
        }

        self.state.data_summary = summary

        return {
            "summary": summary,
            "target_column": self.state.target_column
        }