from orchestration.orchestrator import Orchestrator
from orchestration.router import Router
from orchestration.state import AgentState

from agent.data_agent import DataAgent
from agent.anomaly_agent import AnomalyAgent
from agent.forecasting_agent import ForecastingAgent

from tools.registry import ToolRegistry
from tools.read_csv import ReadCSV
from tools.data_analysis import DataAnalyzer
from tools.anomalies import AnomalyDetectionTool

from chat.llm import LLMClient
from chat.prompts import Prompt_Builder
from agent.memory import Memory


class Controller:
    """
    Controls and initializes the Multi-Agent System.
    """

    def __init__(self):
        # CHANGED: only components that don't depend on a per-request
        # AgentState live here now. Previously ToolRegistry objects and
        # agents were all built once here, at import/startup time --
        # before any AgentState existed -- but ReadCSV/DataAnalyzer/
        # AnomalyDetectionTool all take `state` in their constructor, so
        # there was no valid state to give them, and nothing was ever
        # registered into the registries at all. Agents therefore always
        # had empty tool lists (self.tools.schemas() == []), meaning the
        # LLM could never call a single tool, ever.
        self.llm = LLMClient()
        self.prompt_builder = Prompt_Builder()
        self.memory = Memory()

        self.router = Router(llm=self.llm, prompt_builder=self.prompt_builder)

    def _build_agents(self, state):
        """
        CHANGED (new method): build tool registries and agents bound to
        this specific request's AgentState, so state-dependent tools have
        something real to read from and write to.
        """
        data_tools = ToolRegistry()
        data_tools.register(ReadCSV(state))
        data_tools.register(DataAnalyzer(state))

        anomaly_tools = ToolRegistry()
        anomaly_tools.register(AnomalyDetectionTool(state))

        # NOTE: forecasting_tools stays empty on purpose. ForecastingAgent
        # (see forecasting_agent.py) never calls super().run(), so it
        # never goes through the LLM tool-calling loop at all -- it
        # selects/fits/predicts deterministically instead. forecast.py's
        # ForecastTool has no consumer under the current design.
        forecasting_tools = ToolRegistry()

        data_agent = DataAgent(llm=self.llm, tools=data_tools,
                               prompt_builder=self.prompt_builder,
                               memory=self.memory)

        anomaly_agent = AnomalyAgent(llm=self.llm, tools=anomaly_tools,
                                     prompt_builder=self.prompt_builder,
                                     memory=self.memory)

        forecasting_agent = ForecastingAgent(llm=self.llm,
                                             tools=forecasting_tools,
                                             prompt_builder=self.
                                             prompt_builder,
                                             memory=self.memory)

        return data_agent, anomaly_agent, forecasting_agent

    def run(self, task):
        """
        Start the Multi-Agent System.

        Creates a shared AgentState and gives it to
        the Orchestrator.
        """
        state = AgentState(task)

        # CHANGED: agents (and their tool registries) are now built here,
        # per request, tied to this call's state -- instead of once at
        # startup with no state to bind to.
        data_agent, anomaly_agent, forecasting_agent = (
            self._build_agents(state))

        orchestrator = Orchestrator(router=self.router,
                                    llm=self.llm,
                                    prompt_builder=self.prompt_builder,
                                    data_agent=data_agent,
                                    forecasting_agent=forecasting_agent,
                                    anomaly_agent=anomaly_agent,
                                    max_iterations=10)

        response = orchestrator.run(task=task, state=state)
        return response
