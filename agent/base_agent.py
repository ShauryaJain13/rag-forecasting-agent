import json


class BaseAgent:
    """
    This class is the Base Agent class. It is used as a base-level
    agent that will serve as the blueprint for the other classes
    """

    def __init__(self, name, llm, tools, system_prompt, prompt_builder,
                 memory, max_iterations=10):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder
        # self.current_state = current_state
        self.memory = memory
        self.max_iterations = max_iterations

    def run(self, task, state):
        """
        Run the code until satisfactory conditions are met
        accept the input, generate output, and keep the loop of ReAct until
        all conditions are met
        """
        self.memory.add({
            "role": "user",
            "content": task
        })

        for _ in range(self.max_iterations):
            message = self._build_context(task, state)

            try:
                response = self.llm.generate(message, self.tools.schemas())
            except Exception as e:
                state.add_error({"agent": self.name,
                                 "error": str(e)})
                return f"An error occured: {e}"

            tool_call = self._is_tool_call(response)
            if tool_call is None:
                if response is None:
                    return "Sorry, I could not generate a response"

                final_answer = response.content
                self.memory.add({"role": "assistant",
                                "content": final_answer})
                return final_answer

            try:
                result = self._handle_tool_call(tool_call)
            except Exception as e:
                state.add_error({"agent": self.name,
                                 "error": str(e)})
                result = {"result": "error",
                          "error": str(e)}
            self._add_tool_call_result(response, tool_call, result)

        return f"Maximum iterations {self.max_iterations} reached, agent"\
               "has stopped"

    def _is_tool_call(self, response):
        """
        This function determines whether the LLM response is to do a tool call
        or is the final output
        """
        if response is None:
            return None

        if not response.tool_calls:
            return None

        return response.tool_calls[0]

    def _handle_tool_call(self, call):
        """
        Processes one tool call from the LLM
        """
        tool_name = call.function.name
        arguments = self._parse_arguments(call.function.arguments)
        return self._execute_tool_call(tool_name, arguments)

    def _parse_arguments(self, arguments):
        """
        Parses the expression that will be entered in the argument
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid tool arguments: {arguments}") from e

        if isinstance(arguments, dict):
            return arguments

        raise TypeError(f"Unknown argument type: {type(arguments)}")

    def _execute_tool_call(self, tool_name, arguments):
        """
        The actual execution of the tool call itself-processing it from the llm
        """
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name} is not registered")

        return tool.execute(arguments)

        # if tool is None:
        #     state.add_error("Tool is not registered")
        #     raise ValueError(f"Tool {tool_name} is not registered")

        # try:
        #     return tool.execute(arguments)
        # except Exception as e:
        #     return {"status": "error",
        #             "error": str(e)}

    def _add_tool_call_result(self, response, call, result):
        """
        Adding the tool call and its result to the memory
        """
        tool_calls = [tool_call.model_dump()
                      for tool_call in response.tool_calls]

        self.memory.add({"role": "assistant",
                         "content": response.content,
                         "tool_calls": tool_calls})
        self.memory.add({"role": "tool",
                         "tool_call_id": call.id,
                         "content": str(result)})

    def _build_context(self, task, state):
        """
        Create context for the LLM if necessary
        """
        context = state.to_dict()
        return self.prompt_builder.build_messages(self.memory,
                                                  self.system_prompt, context)
