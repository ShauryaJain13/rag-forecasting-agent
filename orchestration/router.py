from agent.memory import Memory
import json


class Router:
    """
    The Router determines which specialist agent should act next.

    It uses the LLM to examine the current task and shared state,
    then delegates the next piece of work to the appropriate agent.
    """

    def __init__(self, llm, prompt_builder):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.memory = Memory()

        self.available_agents = {
            "data_agent": (
                "Responsible for understanding, inspecting, "
                "cleaning, and preparing datasets."),

            "forecasting_agent": (
                "Responsible for evaluating forecasting models, "
                "selecting an appropriate model, and producing "
                "forecasts."),

            "anomaly_agent": (
                "Responsible for detecting and analyzing anomalies "
                "in the dataset and determining whether they may "
                "affect forecasting.")}

    def route(self, task, state):
        """
        Decide which agent should act next, or answer directly.
        """
        context = state.to_dict()
        system_prompt = self._build_system_prompt()
        messages = self.prompt_builder.build_messages(
            memory=self.memory,
            system_prompt=system_prompt,
            context={"user_request": task,
                     "current_state": context})

        try:
            response = self.llm.generate(messages, tools=None)
        except Exception as e:
            raise RuntimeError(f"Router LLM failed: {e}") from e

        if response is None:
            raise RuntimeError("Router LLM returned no response.")

        decision = self._parse_response(response.content)
        self._validate_decision(decision)
        return decision

    def _build_system_prompt(self):
        """
        Build the instructions given to the routing LLM.
        """
        agents = "\n".join(f"- {name}: {description}" for name, description
                           in self.available_agents.items())

        return f"""
You are the Router for a multi-agent data forecasting system.

Your job is to determine which specialist agent should act next,
OR to answer the user directly if no specialist is needed.

Available agents:

{agents}

Use "direct_response" instead of a specialist agent when the user's
message does not require data analysis, anomaly detection, or
forecasting -- for example: greetings, thanks, small talk, or
general questions you can answer yourself using only the shared
state already available.

You must examine:
1. The user's original request.
2. What has already been completed.
3. The information currently available in shared state.
4. Any errors or warnings.
5. What work is still required.

Do NOT perform data analysis, anomaly detection, or forecasting
yourself. Delegate that work to the appropriate specialist agent.

Return ONLY valid JSON in one of these two exact formats:

For delegating to a specialist:
{{
    "agent": "agent_name",
    "task": "specific task for the selected agent",
    "reason": "brief explanation for why this agent should act next"
}}

For answering directly (no specialist needed):
{{
    "agent": "direct_response",
    "task": "",
    "reason": "brief explanation for why no specialist is needed",
    "response": "the actual message to send back to the user"
}}

The "agent" field MUST be one of:

- data_agent
- forecasting_agent
- anomaly_agent
- direct_response

Do not return markdown.
Do not include additional fields.
"""

    def _parse_response(self, response):
        """
        Convert the LLM's response into a Python dictionary.
        """
        response = response.strip()

        if response.startswith("```"):
            response = response.replace("```json", "").replace("```", ""
                                                               ).strip()

        try:
            decision = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Router returned invalid JSON: {response}")from e

        if not isinstance(decision, dict):
            raise ValueError("Router response must be a JSON object.")

        return decision

    def _validate_decision(self, decision):
        """
        Ensure the router returned a valid routing decision.
        """
        required_fields = {"agent", "task", "reason"}
        missing = required_fields - decision.keys()
        if missing:
            raise ValueError(f"Router decision is missing fields: {missing}")

        valid_agents = set(self.available_agents.keys()) | {"direct_response"}
        # CHANGED: "direct_response" is now a valid value alongside the
        # three specialists, since the router can answer directly instead
        # of always being forced to delegate.

        if decision["agent"] not in valid_agents:
            raise ValueError(f"Unknown agent: {decision['agent']}")

        if not isinstance(decision["task"], str):
            raise TypeError("Router task must be a string.")

        if not isinstance(decision["reason"], str):
            raise TypeError("Router reason must be a string.")

        # CHANGED: when the router chooses direct_response, it must also
        # supply the actual text to send back -- otherwise the orchestrator
        # would have nothing to show the user.
        if decision["agent"] == "direct_response":
            if "response" not in decision:
                raise ValueError(
                    "direct_response decisions must include a 'response' "
                    "field.")
            if not isinstance(decision["response"], str):
                raise TypeError("Router response must be a string.")
