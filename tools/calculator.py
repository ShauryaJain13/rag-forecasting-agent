class Calculator:
    """
    This class is a tool that an LLM can call. It serves as a calculator
    """

    # def __init__(self):
    #     self.name = "calculator"
    #     self.description = "Evaluate mathematical expressions"

    def execute(self, expression: str):
        """
        Executes the inputted expression that LLM deems is appropriate for
        this tool and returns the answer
        """
        try:
            return eval(expression)
        except Exception as e:
            return f"Error evaluating expression: {e}"
