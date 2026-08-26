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
            "data_agent": ("Responsible for understanding, inspecting, "
                           "cleaning, and preparing datasets."),

            "rag_agent": ("Responsible for retrieving relevant information "
                          "from the knowledge base and providing contextual "
                          "information from stored documents."),

            "forecasting_agent": ("Responsible for evaluating forecasting "
                                  "models, selecting an appropriate model, "
                                  "and producing forecasts."),

            "anomaly_agent": ("Responsible for detecting and analyzing "
                              "anomalies in the dataset and determining "
                              "whether they may affect forecasting.")}

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
        Building system prompt for routing
        """
        agents = "\n".join(
            f"- {name}: {description}"
            for name, description in self.available_agents.items()
        )

        return f"""
You are the Router of a multi-agent forecasting system.

Choose the SINGLE next agent required to answer the user's request.

Available agents:

{agents}

Rules:

1. Use data_agent when the dataset must be loaded, inspected,
   cleaned, or understood.

2. Use rag_agent when information from the knowledge base
   is relevant to the request.

3. Use anomaly_agent when anomaly detection or anomaly
   analysis is required.

4. Use forecasting_agent when a forecast is required.

5. Do not repeat an agent whose required work is already
   completed.

6. Examine completed_agents and the current state before
   deciding.

7. Use direct_response when all required work is complete
   or when no specialist is required.

8. Do not perform the specialist's work yourself.

Return ONLY valid JSON.

Specialist:

{{
    "agent": "agent_name",
    "task": "specific task",
    "reason": "why this agent is needed"
}}

Direct response:

{{
    "agent": "direct_response",
    "task": "",
    "reason": "why no more agents are needed",
    "response": "answer to the user"
}}

Valid agents:
- data_agent
- rag_agent
- anomaly_agent
- forecasting_agent
- direct_response
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
