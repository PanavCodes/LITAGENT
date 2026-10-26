"""LangGraph StateGraph workflow construction with CRAG correction loop & uncertainty.

Per PRD.md §7.5 (FR-AGT-1, 7, 8) & ARCHITECTURE.md §3:
Assembles router, graph branch, vector branch, grader, rewrite loop,
generate, and uncertainty scorer into a robust LangGraph StateGraph.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from litagent.agent.generate import GenerateNode
from litagent.agent.grader import GraderNode
from litagent.agent.graph_branch import GraphBranchNode
from litagent.agent.rewrite import RewriteNode
from litagent.agent.router import RouterNode
from litagent.agent.state import AgentState
from litagent.agent.uncertainty import compute_uncertainty
from litagent.agent.vector_branch import VectorBranchNode
from litagent.config import load_config
from litagent.llm import BaseLLM, get_llm
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.retrieval.text2cypher import Text2CypherRetriever
from litagent.stores.graph_base import GraphStore
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

logger = logging.getLogger(__name__)


def abstain_node(state: AgentState) -> AgentState:
    """Handles out-of-scope queries by abstaining without hallucination."""
    state["answer"] = (
        "This question is outside the scope of the scientific literature review corpus. "
        "LitAgent only answers questions related to retrieval-augmented generation and information retrieval."
    )
    state["citations"] = []
    state["uncertainty"] = None
    return state


def both_branches_node(
    state: AgentState,
    graph_node: GraphBranchNode,
    vector_node: VectorBranchNode,
) -> AgentState:
    """Executes both graph and vector retrieval branches and merges evidence."""
    state = graph_node(state)
    state = vector_node(state)
    return state


def route_decision(state: AgentState) -> str:
    """Conditional edge selector based on router output."""
    route = state.get("route", "vector")
    if route == "graph":
        return "graph_branch"
    if route == "both":
        return "both_branches"
    if route == "none":
        return "abstain"
    return "vector_branch"


def grader_decision(state: AgentState, threshold: float = 0.6, max_iterations: int = 2) -> str:
    raw_grade = state.get("grade")
    grade = 0.0 if raw_grade is None else raw_grade
    iteration = state.get("iteration", 0)

    if grade >= threshold or iteration >= max_iterations:
        return "generate"
    return "rewrite"



def uncertainty_node(state: AgentState) -> AgentState:
    """Computes runtime uncertainty and applies confidence flags."""
    compute_uncertainty(state)
    return state


def build_litagent_graph(
    llm: BaseLLM | None = None,
    graph_store: GraphStore | None = None,
    faiss_store: FAISSVectorStore | None = None,
    bm25_store: BM25Store | None = None,
    reranker: CrossEncoderReranker | None = None,
    grader_threshold: float = 0.6,
    max_iterations: int = 2,
    enable_loop: bool = True,
) -> Any:
    """Build and compile the complete Agentic RAG workflow with CRAG loop."""
    load_config()
    active_llm = llm or get_llm(use_fake=True)
    active_graph = graph_store or NetworkXGraphStore()
    active_faiss = faiss_store or FAISSVectorStore()
    active_bm25 = bm25_store or BM25Store()
    active_reranker = reranker or CrossEncoderReranker(use_model=False)

    text2cypher = Text2CypherRetriever(graph_store=active_graph, llm=active_llm)

    router = RouterNode(llm=active_llm)
    graph_branch = GraphBranchNode(text2cypher=text2cypher)
    vector_branch = VectorBranchNode(
        faiss_store=active_faiss,
        bm25_store=active_bm25,
        reranker=active_reranker,
    )
    grader = GraderNode(llm=active_llm)
    rewriter = RewriteNode(llm=active_llm)
    generator = GenerateNode(llm=active_llm)

    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router", router)
    workflow.add_node("graph_branch", graph_branch)
    workflow.add_node("vector_branch", vector_branch)
    workflow.add_node(
        "both_branches",
        lambda s: both_branches_node(s, graph_branch, vector_branch),
    )
    workflow.add_node("grader", grader)
    workflow.add_node("rewrite", rewriter)
    workflow.add_node("generate", generator)
    workflow.add_node("uncertainty", uncertainty_node)
    workflow.add_node("abstain", abstain_node)

    # Initial Routing
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "graph_branch": "graph_branch",
            "vector_branch": "vector_branch",
            "both_branches": "both_branches",
            "abstain": "abstain",
        },
    )

    if enable_loop:
        # Branches -> Grader
        workflow.add_edge("graph_branch", "grader")
        workflow.add_edge("vector_branch", "grader")
        workflow.add_edge("both_branches", "grader")

        # Grader -> Generate (if sufficient or max iter) or Rewrite
        workflow.add_conditional_edges(
            "grader",
            lambda s: grader_decision(s, threshold=grader_threshold, max_iterations=max_iterations),
            {
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )

        # Rewrite -> Re-retrieve
        workflow.add_conditional_edges(
            "rewrite",
            route_decision,
            {
                "graph_branch": "graph_branch",
                "vector_branch": "vector_branch",
                "both_branches": "both_branches",
                "abstain": "abstain",
            },
        )
    else:
        # Non-agentic / baseline: straight from branch to generate
        workflow.add_edge("graph_branch", "generate")
        workflow.add_edge("vector_branch", "generate")
        workflow.add_edge("both_branches", "generate")

    workflow.add_edge("generate", "uncertainty")
    workflow.add_edge("uncertainty", END)
    workflow.add_edge("abstain", END)

    return workflow.compile()


class LitAgentWorkflow:
    """High-level workflow runner interface used by CLI, Gradio UI, and Eval."""

    def __init__(
        self,
        config: Any = None,
        llm_client: Any = None,
        graph_store: Any = None,
        faiss_store: Any = None,
        bm25_store: Any = None,
        dense_retriever: Any = None,
        sparse_retriever: Any = None,
        reranker: Any = None,
        text2cypher: Any = None,
    ) -> None:
        self.config = config or load_config()
        self.llm = llm_client
        self.graph_store = graph_store
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.text2cypher = text2cypher

        enable_loop = self.config.agent.max_iterations > 0 if hasattr(self.config, "agent") else True
        max_iter = self.config.agent.max_iterations if hasattr(self.config, "agent") else 2
        thresh = self.config.agent.grader_threshold if hasattr(self.config, "agent") else 0.6

        self.graph_app = build_litagent_graph(
            llm=self.llm,
            graph_store=self.graph_store,
            faiss_store=self.faiss_store,
            bm25_store=self.bm25_store,
            reranker=self.reranker,
            grader_threshold=thresh,
            max_iterations=max_iter,
            enable_loop=enable_loop,
        )

    def run(self, question: str) -> dict[str, Any]:
        """Execute LitAgent workflow for a question and return structured results."""
        forced = None
        if hasattr(self.config, "agent") and self.config.agent.forced_route is not None:
            forced = self.config.agent.forced_route

        init_state: AgentState = {
            "question": question,
            "current_query": question,
            "forced_route": forced,
            "iteration": 0,
            "llm_calls": 0,
            "rows": [],
            "chunks": [],
            "evidence": [],
            "citations": [],
            "errors": [],
            "trace": [],
        }
        result = self.graph_app.invoke(init_state)

        ans = result.get("answer") or ""
        u_rep = result.get("uncertainty")
        u_score = u_rep.score if u_rep else 0.5
        u_signals = u_rep.signals if u_rep else {}
        is_abstain = (
            (u_rep.abstain if u_rep else False)
            or result.get("route") == "none"
            or "outside the scope" in ans.lower()
            or "not have sufficient information" in ans.lower()
        )

        return {
            "answer": ans,
            "route": result.get("route", "vector"),
            "route_rationale": result.get("route_rationale", ""),
            "cypher": result.get("cypher"),
            "cypher_status": result.get("cypher_status"),
            "rows": result.get("rows", []),
            "passages": result.get("chunks", []),
            "citations": result.get("citations", []),
            "grade": result.get("grade"),
            "iteration": result.get("iteration", 0),
            "llm_calls": result.get("llm_calls", 0),
            "uncertainty_score": u_score,
            "signals": u_signals,
            "is_abstain": is_abstain,
            "state": result,
        }
