"""Modern Gradio Web UI for LitAgent literature review assistant.

Per PRD.md §7.8 (FR-UI-2, FR-UI-4) & PHASES.md Phase 7:
- Interactive chat interface with real-time inspection side panels
- Route & Rationale inspection
- Graph Cypher query & execution status viewer
- Retrieved passages & relevance score inspector
- CRAG correction loop sufficiency & missing info monitor
- Uncertainty calibration gauge & signal decomposition (S1, S2, S5)
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from litagent.agent.workflow import LitAgentWorkflow
from litagent.config import load_config
from litagent.llm import get_llm
from litagent.retrieval.dense import DenseRetriever
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.retrieval.sparse import SparseRetriever
from litagent.retrieval.text2cypher import Text2CypherRetriever
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

logger = logging.getLogger(__name__)


def create_workflow() -> LitAgentWorkflow:
    """Initialize stores and build the LitAgent workflow."""
    cfg = load_config()
    index_dir = Path("data/index")

    llm_client = get_llm(cfg.llm, use_fake=True)

    graph_store = NetworkXGraphStore(index_dir / "graph.pkl")
    faiss_store = FAISSVectorStore(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "chunk_metadata.json",
    )
    bm25_store = BM25Store(index_dir / "bm25.pkl")

    dense_retriever = DenseRetriever(faiss_store, top_k=20)
    sparse_retriever = SparseRetriever(bm25_store, top_k=20)
    reranker = CrossEncoderReranker(use_model=False)
    text2cypher = Text2CypherRetriever(graph_store=graph_store, llm=llm_client)

    return LitAgentWorkflow(
        config=cfg,
        llm_client=llm_client,
        graph_store=graph_store,
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        text2cypher=text2cypher,
    )


def build_app(workflow: LitAgentWorkflow | None = None) -> gr.Blocks:
    """Build the Gradio interface."""
    active_wf = workflow or create_workflow()

    def process_query(question: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, str, str, str, str, str]:
        if not question.strip():
            return history, "", "", "", "", "", ""

        res = active_wf.run(question)
        answer = res.get("answer", "")
        route = res.get("route", "vector").upper()
        rationale = res.get("route_rationale", "")
        cypher = res.get("cypher") or "None (Query directed to vector branch)"
        cypher_status = res.get("cypher_status") or "N/A"

        passages = res.get("passages", [])
        passage_text = ""
        for idx, p in enumerate(passages[:5]):
            cid = getattr(p, "chunk_id", f"chk_{idx}")
            txt = getattr(p, "text", str(p))[:200]
            score = getattr(p, "score", 0.0)
            passage_text += f"**[{idx+1}] {cid}** (Score: {score:.3f})\n{txt}...\n\n"
        if not passage_text:
            passage_text = "No vector passages retrieved for this route."

        grade = res.get("grade")
        grade_text = f"Sufficiency Grade: {grade:.2f}/1.00\nIterations: {res.get('iteration', 0)}" if grade is not None else "Direct Answer (Loop skipped or threshold satisfied)"

        u_score = res.get("uncertainty_score", 0.5)
        signals = res.get("signals", {})
        s1 = signals.get("S1", 0.5)
        s2 = signals.get("S2", 0.5)
        s5 = signals.get("S5", 0.5)
        conf_label = "HIGH CONFIDENCE" if u_score < 0.4 else ("MEDIUM CONFIDENCE" if u_score < 0.7 else "LOW CONFIDENCE / ABSTAIN")
        uncertainty_text = f"**Status:** {conf_label}\n**Uncertainty Score:** {u_score:.3f}\n\n- S1 (Retrieval Margin): {s1:.3f}\n- S2 (Evidence Grade): {s2:.3f}\n- S5 (Graph Query Status): {s5:.3f}"

        citations = res.get("citations", [])
        cit_text = ""
        for c in citations:
            tag = getattr(c, "tag", "Ref")
            title = getattr(c, "title", "Paper")
            url = getattr(c, "url", "#")
            cit_text += f"- **[{tag}]** [{title}]({url})\n"
        if not cit_text:
            cit_text = "No explicit citation keys cited."

        new_history = list(history)
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": answer})

        route_desc = f"Route: {route} (Status: {cypher_status})\nRationale: {rationale}"
        return new_history, route_desc, cypher, passage_text, grade_text, uncertainty_text, cit_text

    with gr.Blocks(title="LitAgent Reviewer") as demo:

        gr.Markdown(
            "# 📚 LitAgent: Agentic Hybrid RAG for Scientific Literature Review\n"
            "An agentic hybrid RAG system uniting Citation Knowledge Graphs, Dense & Sparse Vector Search, "
            "Corrective Reflection Loops, and Principled Uncertainty Calibration."
        )

        with gr.Row():
            # Left column: Chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(label="Literature Review Agent", height=450)

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a research question",
                        placeholder="e.g. Which papers in the corpus cite Dense Passage Retrieval?",
                        lines=2,
                        scale=8,
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=2)

                gr.Examples(
                    examples=[
                        ["Who wrote the paper Dense Passage Retrieval for Open-Domain Question Answering?"],
                        ["Which papers in the corpus cite Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks?"],
                        ["What core problem or methodology is addressed in ColBERT?"],
                        ["How many papers in the corpus were published in 2024?"],
                        ["What is the weather forecast in Tokyo today?"],
                    ],
                    inputs=query_input,
                )

            # Right column: Live Inspection Panel
            with gr.Column(scale=4):
                gr.Markdown("### 🔍 Execution & Reasoning Inspector")
                with gr.Tabs():
                    with gr.TabItem("🧭 Route & Cypher"):
                        route_box = gr.Textbox(label="Router Classification", lines=2, interactive=False)
                        cypher_box = gr.Code(label="Generated Cypher Query", language="sql", interactive=False)
                    with gr.TabItem("📄 Retrieved Evidence"):
                        passages_box = gr.Markdown(label="Top Passages")
                    with gr.TabItem("🔄 CRAG Loop"):
                        loop_box = gr.Textbox(label="Sufficiency & Iterations", lines=3, interactive=False)
                    with gr.TabItem("📊 Uncertainty & Signals"):
                        uncertainty_box = gr.Markdown(label="Uncertainty Decomposition")
                    with gr.TabItem("📑 Citations"):
                        citations_box = gr.Markdown(label="Referenced Works")

        submit_btn.click(
            fn=process_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot, route_box, cypher_box, passages_box, loop_box, uncertainty_box, citations_box],
        )
        query_input.submit(
            fn=process_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot, route_box, cypher_box, passages_box, loop_box, uncertainty_box, citations_box],
        )

    return demo


def main() -> None:
    custom_css = """
    .gradio-container { max-width: 1300px !important; margin: auto; }
    .inspector-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    """
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css,
    )



if __name__ == "__main__":
    main()
