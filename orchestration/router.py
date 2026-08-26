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
        Build the instructions given to the routing LLM.
        """
        agents = "\n".join(f"- {name}: {description}"
                           for name, description in
                           self.available_agents.items())

        return f"""
You are the Router for a multi-agent data forecasting system.

Your job is to determine which specialist agent should act NEXT.

IMPORTANT:
This is an iterative multi-agent system.

You are NOT required to create the entire workflow at once.
Instead, select the single most appropriate agent to perform
the next required piece of work.

After the selected agent finishes, the shared AgentState will
be updated and you will be called again. You must then examine
the updated state and decide what should happen next.

Available agents:

{agents}

Your available agents have the following responsibilities:

DATA AGENT:
Understands, inspects, cleans, validates, and prepares datasets.
Use this when the dataset needs to be loaded, understood, or
prepared before another operation can be performed.

RAG AGENT:
Retrieves relevant information from the knowledge base,
documentation, and stored domain-specific information.
Use this when the user's request requires information that
may exist in the knowledge base.

ANOMALY AGENT:
Detects and analyzes anomalies in the dataset and determines
whether they may affect the analysis or forecasting process.

FORECASTING AGENT:
Evaluates forecasting models, selects an appropriate model,
and produces forecasts.

You must examine:

1. The user's original request.
2. The current shared AgentState.
3. Which agents have already completed their work.
4. What information is already available.
5. Any errors or warnings in the state.
6. What work is still required to satisfy the user's request.

WORKFLOW RULES:

1. Select only ONE agent at a time.

2. Do not select an agent if its required work has already
   been completed.

3. If the user requires dataset analysis or forecasting and
   the dataset has not yet been loaded or understood,
   select data_agent first.

4. data_agent should generally run before anomaly_agent when
   anomaly detection requires understanding or preparation
   of the dataset.

5. data_agent should generally run before forecasting_agent
   when forecasting requires dataset preparation.

6. Select rag_agent when the user's request requires
   information from documentation, domain knowledge, stored
   files, or the knowledge base.

7. RAG is NOT automatically required for every forecasting
   request. Only use rag_agent when retrieved knowledge would
   actually help answer the user's request.

8. Select anomaly_agent when anomaly detection or anomaly
   analysis is explicitly requested, or when anomaly analysis
   is necessary to satisfy the forecasting/analysis request.

9. Select forecasting_agent only when the user has requested
   forecasting or when producing a forecast is necessary to
   answer the request.

10. If RAG has already retrieved relevant information, do not
    run rag_agent again unless the existing information is
    insufficient for the task.

11. If all required specialist work has been completed, use
    direct_response.

12. Do not perform data analysis, anomaly detection,
    forecasting, or document retrieval yourself. Delegate
    these tasks to the appropriate specialist agent.

13. Do not invent results that are not present in the shared
    state.

14. When selecting an agent, provide a specific task that
    explains exactly what that agent should do next.

15. Prefer the smallest number of agent executions necessary
    to answer the user's request.

EXAMPLE WORKFLOW:

If the user asks:
"Forecast the next 30 days using information about holiday
effects from our documentation."

A possible sequence is:
data_agent -> rag_agent -> forecasting_agent -> direct_response

However, you must determine the actual next step from the
CURRENT AgentState rather than blindly following this example.

For example, if data_agent has already completed its work,
do not select data_agent again.

If the user asks:
"What does our documentation say about holiday effects?"

The appropriate sequence may simply be:
rag_agent -> direct_response

If the user asks:
"Find anomalies in my dataset."

The appropriate sequence may be:
data_agent
    ->
anomaly_agent
    ->
direct_response

If the user says: "Hello, how are you?"

Use:
direct_response
You must return ONLY valid JSON.
For a specialist agent, use EXACTLY:

{{
    "agent": "agent_name",
    "task": "specific task for the selected agent",
    "reason": "brief explanation for why this agent should act next"
}}

For a direct response, use EXACTLY:

{{
    "agent": "direct_response",
    "task": "",
    "reason": "brief explanation for why no specialist is needed",
    "response": "the actual message to send to the user"
}}

The "agent" field MUST be one of:

- data_agent
- rag_agent
- forecasting_agent
- anomaly_agent
- direct_response

Do not return a list of agents.

Do not return a workflow or plan.

Return ONLY the SINGLE next action.

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
