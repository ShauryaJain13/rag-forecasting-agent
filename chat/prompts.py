import json

from agent.memory import Memory


class Prompt_Builder:
    """
    Builds messages that are sent to the LLM.

    The prompt can contain:
    - a system prompt describing the role of the agent
    - conversation history
    - the current shared AgentState/context
    """

    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def build_messages(self, memory: Memory = None, system_prompt=None,
                       context=None) -> list:
        """
        Build the messages sent to the LLM.
        """
        messages = []
        current_system_prompt = (system_prompt if system_prompt is not None
                                 else self.system_prompt)

        if current_system_prompt is not None:
            messages.append({"role": "system",
                             "content": current_system_prompt})

        if memory is not None:
            if hasattr(memory, "get_messages"):
                messages.extend(memory.get_messages())

            elif isinstance(memory, list):
                messages.extend(memory)
            else:
                raise TypeError("history must be a Memory object or a list")

        if context is not None:
            if isinstance(context, dict):
                context_text = json.dumps(context, default=str, indent=2)
            else:
                context_text = str(context)
            messages.append({"role": "system",
                             "content": ("Current shared state of the "
                                         "multi-agent system:\n\n"
                                         f"{context_text}")})
        return messages

    def get_system_prompt(self):
        """
        Return the current system prompt.
        """
        return self.system_prompt

    def set_system_prompt(self, system_prompt):
        """
        Set the system prompt.
        """
        self.system_prompt = system_prompt
