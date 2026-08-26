from orchestration.router import Router

from agent.data_agent import DataAgent
from agent.anomaly_agent import AnomalyAgent
from agent.forecasting_agent import ForecastingAgent
from agent.rag_agent import RAGAgent
from agent.memory import Memory

from chat.llm import LLMClient
from chat.prompts import Prompt_Builder

from rag.document import DocumentLoader
from rag.chunker import TextChunker
from rag.embeddings import Embedder
from rag.vector_state import VectorStore
from rag.retriever import Retriever
from rag.pipeline import RAGPipeline, KnowledgeBase

from tools.registry import ToolRegistry
from tools.knowledge import KnowledgeBaseTool
from tools.read_csv import ReadCSV
from tools.data_analysis import DataAnalyzer
from tools.anomalies import AnomalyDetectionTool


class Controller:
    """
    Controls the execution of the multi-agent forecasting system.

    The Controller is responsible for:

    1. Constructing the system.
    2. Connecting the agents.
    3. Connecting the RAG pipeline.
    4. Binding tools to the current AgentState.
    5. Asking the Router which agent should act next.
    6. Executing the selected agent.
    7. Detecting when the requested work is complete.
    8. Generating the final response.
    """

    def __init__(self, max_steps=10):

        self.max_steps = max_steps

        self.llm = LLMClient()
        self.prompt_builder = Prompt_Builder()
        self.router = Router(llm=self.llm, prompt_builder=self.prompt_builder)

        self.state_bound_tools = []
        self.knowledge_base_tool = self._build_rag()
        self.agents = self._build_agents()

    def _build_rag(self):
        """
        Builds the complete RAG system. It has two separate paths:
        one which is for ingestion (adding document), and the other for
        retrieving the information needed
        """
        embedder = Embedder()
        vector_store = VectorStore()
        retriever = Retriever(embedder=embedder, vector_store=vector_store,
                              top_k=5, minimum_threshold=0.6)

        rag_pipeline = RAGPipeline(retriever=retriever)
        self.knowledge_base = KnowledgeBase(
            loader=DocumentLoader(),
            chunker=TextChunker(chunk_size=800, chunk_overlap=100),
            embedder=embedder, vector_storage=vector_store)

        knowledge_base_tool = KnowledgeBaseTool(rag_pipeline=rag_pipeline)

        return knowledge_base_tool

    def index_document(self, filepath):
        """
        Add a document to the knowledge base.
        """
        return self.knowledge_base.index_file(filepath)

    def _build_agents(self):
        """
        Construct all specialist agents.
        """
        agents = {}

        data_tools = ToolRegistry()
        read_csv_tool = ReadCSV(state=None)
        data_analyzer_tool = DataAnalyzer(state=None)
        data_tools.register(read_csv_tool)
        data_tools.register(data_analyzer_tool)
        self.state_bound_tools.extend([read_csv_tool, data_analyzer_tool])
        agents["data_agent"] = DataAgent(llm=self.llm, tools=data_tools,
                                         prompt_builder=self.prompt_builder,
                                         memory=Memory())

        anomaly_tools = ToolRegistry()
        anomaly_tool = AnomalyDetectionTool(state=None)
        anomaly_tools.register(anomaly_tool)
        self.state_bound_tools.append(anomaly_tool)
        agents["anomaly_agent"] = AnomalyAgent(llm=self.llm,
                                               tools=anomaly_tools,
                                               prompt_builder=self.
                                               prompt_builder,
                                               memory=Memory())

        agents["forecasting_agent"] = ForecastingAgent(llm=self.llm,
                                                       tools=ToolRegistry(),
                                                       prompt_builder=self.
                                                       prompt_builder,
                                                       memory=Memory())

        agents["rag_agent"] = RAGAgent(
            name="rag_agent", llm=self.llm, tools=ToolRegistry(),
            system_prompt=("You retrieve relevant information from the "
                           "knowledge base to support data analysis and "
                           "forecasting."),
            prompt_builder=self.prompt_builder, memory=Memory(),
            knowledge_base_tool=self.knowledge_base_tool)

        return agents

    def run(self, state):
        """
        Execute the complete multi-agent workflow.
        """
        for tool in self.state_bound_tools:
            tool.state = state

        for agent in self.agents.values():
            if agent.memory is not None:
                agent.memory.clear_history()

        for _ in range(self.max_steps):
            decision = self.router.route(state.user_request, state)
            agent_name = decision["agent"]

            if agent_name == "direct_response":
                state.final_response = decision.get("response", "")
                state.current_agent = None
                return state

            if agent_name not in self.agents:
                state.add_error({"component": "controller",
                                 "error": (f"Unknown agent selected by"
                                           "router: "f"{agent_name}")})
                return state

            agent = self.agents[agent_name]
            task = decision.get("task", "")

            try:
                agent.run(task, state)
                print("\n===== DEBUG =====")
                print("Agent executed:", agent_name)
                print("Completed agents:", state.completed_agents)
                print("Target column:", state.target_column)
                print("Forecast:", state.forecast)
                print("Final response:", state.final_response)
                print("Errors:", state.errors)
                print("=================\n")

            except Exception as e:
                state.add_error({"agent": agent_name,
                                 "error": str(e)})

                print("MAX STEPS REACHED")
                print("Completed agents:", state.completed_agents)
                print("Forecast:", state.forecast)
                print("Final response:", state.final_response)
                print("Errors:", state.errors)

                state.current_agent = None
                return state

            if self._workflow_complete(state):
                return self._generate_final_response(state)

        state.add_error({"component": "controller",
                         "error": (f"Maximum number of agent steps "
                                   f"({self.max_steps}) reached.")})
        return state

    def _workflow_complete(self, state):
        """
        Determines whether the requested work has been completed.
        This prevents the Router from having to decide when the
        entire system is finished.
        The Controller knows what outputs actually exist in state.
        """
        request = state.user_request.lower()
        forecasting_requested = any(
            word in request
            for word in ["forecast", "forecasting", "predict", "prediction",
                         "future"])

        if forecasting_requested:
            return state.forecast is not None

        anomaly_requested = any(word in request
                                for word in ["anomaly", "anomalies",
                                             "outlier", "outliers"])

        if anomaly_requested:
            return (state.anomaly_analysis is not None or
                    len(state.anomalies) > 0)

        rag_requested = any(word in request for word in ["documentation",
                                                         "document",
                                                         "knowledge base",
                                                         "according to",
                                                         "according"])
        if rag_requested:
            return (len(state.retrieved_documents) > 0)
        return False

    def _generate_final_response(self, state):
        """
        Generates the final natural-language answer from the
        completed AgentState. The final LLM does NOT perform the analysis
        itself. It simply turns the results produced by the specialist
        agents into a useful response for the user.
        """
        context = state.to_dict()
        system_prompt = """
You are the final response generator for a multi-agent
forecasting copilot.

The specialist agents have already performed the required work.

Your job is ONLY to communicate their results to the user.

Do NOT perform new analysis.

Do NOT invent values.

Do NOT modify numerical results.

Use only the information present in AgentState.

If a forecast was produced, clearly explain:

- the target column
- the forecast horizon
- the selected model
- the forecast values
- relevant model evaluation metrics, if available

If anomalies were detected, explain the relevant anomaly
results.

If RAG information was retrieved, use it when relevant and
mention the relevant sources when appropriate.

Answer the user's original question directly.

Keep the response clear and reasonably concise.
"""
        final_memory = Memory()
        messages = self.prompt_builder.build_messages(
            memory=final_memory, system_prompt=system_prompt,
            context={"user_request": state.user_request,
                     "current_state": context})

        try:
            response = self.llm.generate(messages, tools=None)
            if response is None:
                raise RuntimeError("Final response LLM returned no response.")

            state.final_response = response.content
            state.current_agent = None
            return state

        except Exception as e:
            state.add_error({"component": "final_response",
                             "error": str(e)})
            state.final_response = self._build_fallback_response(state)
            state.current_agent = None
            return state

    def _build_fallback_response(self, state):
        """
        Generates a basic response without making another LLM call.
        This is useful if the final response LLM hits a rate limit
        or otherwise fails.
        """
        if state.forecast is not None:
            response = "Forecast generated successfully."
            if state.target_column:
                response += (f"\nTarget column: {state.target_column}")

            if state.selected_model:
                response += (f"\nSelected model: {state.selected_model}")

            if state.forecast_metrics:
                response += (f"\nForecast metrics: {state.forecast_metrics}")

            response += (f"\n\nForecast:\n {state.forecast}")
            return response

        if state.anomalies:
            response = ("Anomaly detection completed.")
            response += (f"\n\nAnomalies:\n {state.anomalies}")

            if state.anomaly_analysis:
                response += (f"\n\nAnalysis:\n {state.anomaly_analysis}")
            return response

        if state.rag_context:
            response = ("I found the following relevant information "
                        "in the knowledge base:\n\n")
            response += state.rag_context
            return response

        if state.errors:
            return ("I was unable to complete the requested task.\n\n"
                    f"Errors:\n {state.errors}")

        return ("I was unable to generate a response for the requested task.")
