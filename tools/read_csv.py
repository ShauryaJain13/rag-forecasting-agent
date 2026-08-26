import pandas as pd


class ReadCSV:
    """
    Tool for loading a CSV file into the shared multi-agent state.
    """

    name = "read_csv"
    description = ("Load a CSV file from disk into the shared dataset. "
                   "The loaded data becomes available to all other agents "
                   "as state.data.")

    def __init__(self, state):
        self.state = state

    def schema(self):
        """
        Returns the schema of the tool
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file to load."
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }

    def execute(self, arguments):
        """
        Loads the CSV into state.data and returns a lightweight summary
        (not the full DataFrame -- see note in state.py's _data_preview).
        """
        file_path = arguments.get("file_path")
        if not file_path:
            raise ValueError("A file_path is required.")

        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            return f"Error: file was not found at {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"

        self.state.data = data
        return {"rows": len(data), "columns": list(data.columns)}
