# class ReadFile:
#     """
#     This class is a tool that an LLM can call. It serves as a file reader
#     """

#     def execute(self, file_path: str):
#         """
#         Executes the inputted expression that LLM deems is appropriate for
#         this tool and returns the answer
#         """
#         try:
#             with open(file_path, "r", encoding='utf-8') as f:
#                 return f.read()
#         except FileNotFoundError:
#             return f"Error: file was not found as {file_path}"
#         except Exception as e:
#             return f"Error reading file: {e}"

class ReadFile:
    """
    This class is a tool that an LLM can call. It serves as a file reader.

    CHANGED: same interface fix as calculator.py -- added name/
    description/schema, and execute() now takes the `arguments` dict
    every other tool receives instead of a raw `file_path: str`.
    """

    name = "read_file"
    description = "Read and return the contents of a text file from disk."

    def schema(self):
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
                            "description": "Path to the text file to read."
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }

    def execute(self, arguments):
        """
        Executes the file read the LLM deems appropriate for this tool
        and returns the contents.
        """
        file_path = arguments.get("file_path")
        if not file_path:
            raise ValueError("A 'file_path' is required.")

        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: file was not found at {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"