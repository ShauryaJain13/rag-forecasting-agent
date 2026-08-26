# from orchestration.router import Router
# # from orchestration.state import AgentState

# from agent.data_agent import DataAgent
# from agent.anomaly_agent import AnomalyAgent
# from agent.forecasting_agent import ForecastingAgent
# from agent.rag_agent import RAGAgent

# from chat.llm import LLMClient
# from chat.prompts import Prompt_Builder

# from rag.embeddings import Embedder
# from rag.vector_state import VectorStore
# from rag.retriever import Retriever
# from rag.pipeline import RAGPipeline
# from tools.knowledge import KnowledgeBaseTool


# class Controller:
#     """
#     Controls the execution of the multi-agent system.

#     The Controller is responsible for constructing the system
#     and repeatedly executing the agent selected by the Router.
#     """

#     def __init__(self, max_steps=10):

#         self.max_steps = max_steps

#         self.llm = LLMClient()
#         self.prompt_builder = Prompt_Builder()

#         self.knowledge_base_tool = self._build_rag()

#         self.agents = self._build_agents()

#         self.router = Router(
#             llm=self.llm,
#             prompt_builder=self.prompt_builder
#         )

#     def _build_rag(self):
#         """
#         Build the RAG retrieval pipeline.
#         """

#         embedder = Embedder()

#         vector_store = VectorStore()

#         retriever = Retriever(
#             embedder=embedder,
#             vector_store=vector_store,
#             top_k=5,
#             minimum_threshold=0.6
#         )

#         rag_pipeline = RAGPipeline(
#             retriever=retriever
#         )

#         return KnowledgeBaseTool(
#             rag_pipeline=rag_pipeline
#         )

#     def _build_agents(self):
#         """
#         Construct and register all specialist agents.
#         """

#         agents = {}

#         agents["data_agent"] = DataAgent(
#             name="data_agent",
#             llm=self.llm,
#             tools=[],
#             system_prompt="",
#             prompt_builder=self.prompt_builder,
#             memory=None
#         )

#         agents["rag_agent"] = RAGAgent(
#             name="rag_agent",
#             llm=self.llm,
#             tools=[],
#             system_prompt="",
#             prompt_builder=self.prompt_builder,
#             memory=None,
#             knowledge_base_tool=self.knowledge_base_tool
#         )

#         agents["anomaly_agent"] = AnomalyAgent(
#             name="anomaly_agent",
#             llm=self.llm,
#             tools=[],
#             system_prompt="",
#             prompt_builder=self.prompt_builder,
#             memory=None
#         )

#         agents["forecasting_agent"] = ForecastingAgent(
#             name="forecasting_agent",
#             llm=self.llm,
#             tools=[],
#             system_prompt="",
#             prompt_builder=self.prompt_builder,
#             memory=None
#         )

#         return agents

#     def run(self, state):
#         """
#         Execute the multi-agent workflow.
#         """

#         for _ in range(self.max_steps):

#             decision = self.router.route(
#                 state.user_request,
#                 state
#             )

#             agent_name = decision["agent"]

#             if agent_name == "direct_response":

#                 state.final_response = decision.get(
#                     "response",
#                     ""
#                 )

#                 state.current_agent = None

#                 return state

#             if agent_name not in self.agents:

#                 state.add_error(
#                     f"Unknown agent: {agent_name}"
#                 )

#                 return state

#             agent = self.agents[agent_name]

#             try:
#                 state = agent.run(state)

#             except Exception as e:

#                 state.add_error(
#                     f"{agent_name} failed: {str(e)}"
#                 )

#                 state.current_agent = None

#                 return state

#         state.add_error(
#             "Maximum number of agent steps reached."
#         )

#         return state


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
    Controls the execution of the multi-agent system.

    The Controller is responsible for constructing the system
    and repeatedly executing the agent selected by the Router.
    """

    def __init__(self, max_steps=10):

        self.max_steps = max_steps

        self.llm = LLMClient()
        self.prompt_builder = Prompt_Builder()

        # CHANGED: tools that read/write shared state (ReadCSV,
        # DataAnalyzer, AnomalyDetectionTool) are constructed once here
        # with state=None, then re-pointed at the *current* AgentState at
        # the top of every run() call. They can't be bound to a real
        # state at __init__ time because Controller is built once at app
        # startup, before any AgentState exists -- app.py creates a
        # brand new AgentState per user turn.
        self.state_bound_tools = []

        self.knowledge_base_tool = self._build_rag()

        self.agents = self._build_agents()

        self.router = Router(
            llm=self.llm,
            prompt_builder=self.prompt_builder
        )

    def _build_rag(self):
        """
        Build the RAG retrieval pipeline, plus the ingestion pipeline
        (KnowledgeBase) used to actually load documents into it.

        CHANGED: previously this only built the retrieval side
        (Embedder/VectorStore/Retriever/RAGPipeline/KnowledgeBaseTool).
        Nothing anywhere ever called DocumentLoader/TextChunker/
        VectorStore.add_documents, so the vector store was permanently
        empty and rag_agent could never retrieve anything.
        self.knowledge_base + index_document() below are the missing
        ingestion path -- see app.py's new "index <filepath>" command.
        """

        embedder = Embedder()

        vector_store = VectorStore()

        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=5,
            minimum_threshold=0.6
        )

        rag_pipeline = RAGPipeline(
            retriever=retriever
        )

        # CHANGED: TextChunker no longer takes a `document` constructor
        # argument -- see chunker.py, it was dead/unused there.
        self.knowledge_base = KnowledgeBase(
            loader=DocumentLoader(),
            chunker=TextChunker(chunk_size=800, chunk_overlap=100),
            embedder=embedder,
            vector_storage=vector_store
        )

        return KnowledgeBaseTool(
            rag_pipeline=rag_pipeline
        )

    def index_document(self, filepath):
        """
        Ingest a document (.pdf, .txt, or .csv) into the knowledge base
        so rag_agent has something to retrieve.
        """
        return self.knowledge_base.index_file(filepath)

    def _build_agents(self):
        """
        Construct and register all specialist agents.

        CHANGED (applies to all four agents below):
        1. DataAgent/AnomalyAgent/ForecastingAgent's __init__ signatures
           are (llm, tools, prompt_builder, memory) -- they hardcode
           their own name/system_prompt internally. Passing name= and
           system_prompt= (as before) raised TypeError immediately.
        2. tools=[] was a plain list. BaseAgent.run() calls
           self.tools.schemas() and self.tools.get(name), which a list
           doesn't support. Real ToolRegistry instances are built and
           registered below instead, with the actual tools each agent
           needs.
        3. memory=None. BaseAgent.run() unconditionally calls
           self.memory.add(...), which crashes on None. Every agent now
           gets its own Memory().
        """

        agents = {}

        # --- Data Agent: needs to load a CSV and analyze it ---
        data_tools = ToolRegistry()
        read_csv_tool = ReadCSV(state=None)
        data_analyzer_tool = DataAnalyzer(state=None)
        data_tools.register(read_csv_tool)
        data_tools.register(data_analyzer_tool)
        self.state_bound_tools.extend([read_csv_tool, data_analyzer_tool])

        agents["data_agent"] = DataAgent(
            llm=self.llm,
            tools=data_tools,
            prompt_builder=self.prompt_builder,
            memory=Memory()
        )

        # --- Anomaly Agent: needs to run anomaly detection ---
        anomaly_tools = ToolRegistry()
        anomaly_tool = AnomalyDetectionTool(state=None)
        anomaly_tools.register(anomaly_tool)
        self.state_bound_tools.append(anomaly_tool)

        agents["anomaly_agent"] = AnomalyAgent(
            llm=self.llm,
            tools=anomaly_tools,
            prompt_builder=self.prompt_builder,
            memory=Memory()
        )

        # --- Forecasting Agent: deterministic, doesn't use tools/LLM ---
        agents["forecasting_agent"] = ForecastingAgent(
            llm=self.llm,
            tools=ToolRegistry(),
            prompt_builder=self.prompt_builder,
            memory=Memory()
        )

        # --- RAG Agent ---
        agents["rag_agent"] = RAGAgent(
            name="rag_agent",
            llm=self.llm,
            tools=ToolRegistry(),
            system_prompt=(
                "You retrieve relevant context from the knowledge base "
                "to support dataset analysis and forecasting."
            ),
            prompt_builder=self.prompt_builder,
            memory=Memory(),
            knowledge_base_tool=self.knowledge_base_tool
        )

        return agents

    def run(self, state):
        """
        Execute the multi-agent workflow until the Router determines
        that the task is complete.
        """

        # Bind state-dependent tools to this request
        for tool in self.state_bound_tools:
            tool.state = state

        # Reset agent memories for this request
        for agent in self.agents.values():
            if agent.memory is not None:
                agent.memory.clear_history()

        for step in range(self.max_steps):

            try:
                decision = self.router.route(
                    state.user_request,
                    state
                )

            except Exception as e:
                state.add_error({
                    "component": "router",
                    "error": str(e)
                })
                return state

            agent_name = decision["agent"]
            task = decision.get("task", "")

            # -------------------------
            # DIRECT RESPONSE
            # -------------------------

            if agent_name == "direct_response":

                state.final_response = decision.get(
                    "response",
                    ""
                )

                state.current_agent = None

                return state

            # -------------------------
            # INVALID AGENT
            # -------------------------

            if agent_name not in self.agents:

                state.add_error({
                    "component": "controller",
                    "error": f"Unknown agent: {agent_name}"
                })

                return state

            agent = self.agents[agent_name]

            # -------------------------
            # RUN AGENT
            # -------------------------

            try:
                agent.run(task, state)

            except Exception as e:

                state.add_error({
                    "agent": agent_name,
                    "error": str(e)
                })

                state.current_agent = None

                return state

            # -------------------------
            # CHECK AGENT RESULT
            # -------------------------

            if agent_name not in state.completed_agents:

                state.add_error({
                    "agent": agent_name,
                    "error": (
                        f"{agent_name} returned without marking "
                        "itself as complete."
                    )
                })

                state.current_agent = None

                return state

        # -------------------------
        # MAX STEPS
        # -------------------------

        state.add_error({
            "component": "controller",
            "error": (
                f"Maximum number of agent steps "
                f"({self.max_steps}) reached."
            )
        })

        return state
