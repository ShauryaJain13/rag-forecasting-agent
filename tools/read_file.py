class ReadFile:
    """
    This class is a tool that an LLM can call. It serves as a file reader
    """

    def execute(self, file_path: str):
        """
        Executes the inputted expression that LLM deems is appropriate for
        this tool and returns the answer
        """
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: file was not found as {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"
