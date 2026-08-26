class Calculator:
    """
    This class is a tool that an LLM can call. It serves as a calculator.
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
            return eval(expression)
        except Exception as e:
            return f"Error evaluating expression: {e}"
