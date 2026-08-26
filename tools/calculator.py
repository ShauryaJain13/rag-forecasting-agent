# class Calculator:
#     """
#     This class is a tool that an LLM can call. It serves as a calculator
#     """

#     # def __init__(self):
#     #     self.name = "calculator"
#     #     self.description = "Evaluate mathematical expressions"

#     def execute(self, expression: str):
#         """
#         Executes the inputted expression that LLM deems is appropriate for
#         this tool and returns the answer
#         """
#         try:
#             return eval(expression)
#         except Exception as e:
#             return f"Error evaluating expression: {e}"

class Calculator:
    """
    This class is a tool that an LLM can call. It serves as a calculator.

    CHANGED: this tool previously had no name/description/schema and its
    execute() took a raw `expression: str`. BaseAgent._execute_tool_call
    always calls tool.execute(arguments) with a parsed dict, and
    ToolRegistry.register(tool) requires tool.name -- so as written this
    class couldn't be registered or invoked through the same interface
    every other tool in this codebase (ReadCSV, DataAnalyzer,
    AnomalyDetectionTool) uses. Brought in line with that interface.
    """

    name = "calculator"
    description = "Evaluate a mathematical expression and return the result."

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": ("The mathematical expression "
                                            "to evaluate, e.g. "
                                            "'2 + 2 * 5'.")
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

    def execute(self, arguments):
        """
        Executes the expression the LLM deems appropriate for this tool
        and returns the answer.
        """
        expression = arguments.get("expression")
        if not expression:
            raise ValueError("An 'expression' is required.")

        try:
            # NOTE: eval() is unsafe for untrusted input in production --
            # worth swapping for a restricted parser (e.g. via `ast` or
            # a library like `numexpr`/`asteval`) before this is exposed
            # beyond a local CLI.
            return eval(expression)
        except Exception as e:
            return f"Error evaluating expression: {e}"
