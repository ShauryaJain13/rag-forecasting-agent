from orchestration.state import AgentState


class Orchestrator:
    """
    This class is the functionality behind the MAS. It orchestrates which data
    is passed to which agent, when it is passed, and what to do next
    """
    def __init__(self, router, llm, prompt_builder, data_agent,
                 forecasting_agent, anomaly_agent, max_iterations=10):
        self.router = router
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.agents = {"data_agent": data_agent,
                       "forecasting_agent": forecasting_agent,
                       "anomaly_agent": anomaly_agent}
        self.max_iterations = max_iterations
        self.final_response = None

    def run(self, task, state=None):
        """
        The actual running of the function
        """
        if state is None:
            state = AgentState(task)

        for _ in range(self.max_iterations):
            if self.is_task_complete(state):
                break

            try:
                decision = self.router.route(task, state)
            except Exception as e:
                state.add_error({"component": "orchestrator",
                                 "error": f"routing failed: {e}"})
                break

            # CHANGED: this is the main fix. When the router decides the
            # message doesn't need a specialist agent (greetings, thanks,
            # general questions), it returns "direct_response" plus the
            # actual text to send back. Previously there was no such
            # option -- the router was forced to pick data_agent /
            # forecasting_agent / anomaly_agent even for "hello", and
            # is_task_complete() could never become true without both
            # state.data and state.forecast being set, so the loop would
            # burn through all max_iterations before falling through to
            # generate_final_response() with an essentially empty state
            # to summarize -- which is why you were seeing blank output.
            if decision["agent"] == "direct_response":
                self.final_response = decision["response"]
                return self.final_response

            next_agent_name = decision["agent"]

            if next_agent_name not in self.agents:
                state.add_error({"component": "orchestrator",
                                 "error": f"unknown agent {next_agent_name}"})
                break

            agent_task = decision["task"]
            next_agent = self.agents[next_agent_name]
            state.set_current_agent(next_agent.name)

            try:
                next_agent.run(agent_task, state)
            except Exception as e:
                state.add_error({"agent": next_agent_name,
                                 "error": str(e)})
                break
        else:
            state.add_error({
                "component": "orchestrator",
                "error": (f"Maximum iterations ({self.max_iterations}"
                          ") reached before the task was completed.")})

        self.final_response = self.generate_final_response(task, state)
        return self.final_response

    def is_task_complete(self, state):
        """
        Determine whether enough work has been completed
        to produce a final answer.

        NOTE: still requires both state.data and state.forecast, which is
        correct for actual forecasting requests, but only reachable now
        because direct_response gives conversational messages a proper
        exit path that doesn't depend on this check at all.
        """
        if state.data is None or state.forecast is None:
            return False

        return True

    def generate_final_response(self, task, state):
        """
        Generate the final response to the user using
        the information accumulated in shared state.
        """

        context = state.to_dict()

        messages = self.prompt_builder.build_messages(
            memory=[],
            system_prompt=(
                "You are the final response generator for "
                "a data forecasting system. Use the results "
                "contained in the shared state to answer "
                "the user's request clearly. Explain the "
                "selected model, forecast, and any relevant "
                "warnings or anomalies."),
            context=context)

        response = self.llm.generate(messages, tools=None)
        if response is None:
            return "Unable to generate a final response."

        return response.content
