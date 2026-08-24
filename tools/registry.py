class Tool:
    """
    This class serves as a registry for all the tools the LLM can access
    """

    def __init__(self, name, description, function, arguments):
        self.name = name
        self.description = description
        self.function = function
        self.arguments = arguments

    def schema(self):
        """
        Returns the schema of the tool called
        """
        return {"type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.arguments}}

    def execute(self, arguments):
        """
        Executes the tool call
        """
        return self.function(**arguments)


class ToolRegistry:
    """
    Stores and manages all available tools
    """
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        """
        Registering a new tool in the registry
        """
        self.tools[tool.name] = tool

    def get(self, tool_name):
        """
        Getting the tool by its name
        """
        return self.tools.get(tool_name)

    def schemas(self):
        """
        Returns the schemas for all available tools
        """
        return [tool.schema() for tool in self.tools.values()]

    def list_tools(self):
        """
        Returns a list of names of all tools available
        """
        return list(self.tools.keys())
