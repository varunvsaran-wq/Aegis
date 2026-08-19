"""Gradio demo for Aegis with a harness on/off toggle.

The toggle is the demo moment: the same question and corpus are answered with
the reliability harness off (plain hybrid RAG) and on (structuring, injection
defense, citation contract, NLI verification), so a viewer can watch citations
appear, ungrounded answers abstain, and injected corpus text get neutralized.

Launch with ``aegis serve --demo`` or ``python -m aegis.serve.demo``. The
default model is the deterministic mock so the demo boots with no credentials;
set ``AEGIS_SERVE_MODEL`` to use a real model.
"""

from __future__ import annotations

import os

_SAMPLE_DOCS = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It was designed by the engineer Gustave Eiffel and "
    "completed in 1889 for the World's Fair.\n\n"
    "The Louvre is the world's most-visited museum, located in Paris. It holds "
    "Leonardo da Vinci's Mona Lisa."
)


def build_demo():
    """Construct and return the Gradio Blocks app (does not launch it)."""
    import gradio as gr

    from aegis.serve.engine import AegisEngine

    engine = AegisEngine(model=os.environ.get("AEGIS_SERVE_MODEL", "mock"))

    def _index(corpus: str) -> str:
        docs = [d for d in corpus.split("\n\n---\n\n") if d.strip()] or [corpus]
        n = engine.index(docs)
        return f"Indexed {n} chunks."

    def _answer(question: str, harness: bool):
        if engine._retriever is None:
            engine.index([corpus_default])
        r = engine.answer(question, harness=harness)
        status = []
        if r.abstained:
            status.append("ABSTAINED")
        if r.blocked:
            status.append("BLOCKED (injection defense)")
        if r.grounded is True:
            status.append("grounded ✓")
        elif r.grounded is False:
            status.append("ungrounded ✗")
        header = f"**Answer:** {r.answer or '(none)'}\n\n"
        header += f"**Citations:** {', '.join(r.citations) or '(none)'}\n\n"
        if status:
            header += f"**Status:** {', '.join(status)}\n\n"
        header += (
            f"_mode: {'harnessed' if r.harnessed else 'raw RAG'} · "
            f"llm calls: {r.llm_calls} · latency: {r.latency_s:.3f}s_"
        )
        ctx = "\n\n".join(
            f"[{c['chunk_id']}] {c['text']}" for c in r.retrieved
        )
        return header, ctx

    corpus_default = _SAMPLE_DOCS

    with gr.Blocks(title="Aegis — RAG reliability harness") as demo:
        gr.Markdown(
            "# Aegis\nModel-agnostic RAG reliability harness. Toggle the harness "
            "to compare plain RAG against the full pipeline (query structuring, "
            "injection defense, citation contract, NLI verification)."
        )
        with gr.Row():
            corpus = gr.Textbox(
                label="Corpus (separate documents with a line containing ---)",
                value=corpus_default,
                lines=8,
            )
        index_btn = gr.Button("Index corpus")
        index_status = gr.Markdown()
        index_btn.click(_index, inputs=corpus, outputs=index_status)

        with gr.Row():
            question = gr.Textbox(label="Question", value="Who designed the Eiffel Tower?")
            harness = gr.Checkbox(label="Harness on", value=True)
        answer_btn = gr.Button("Answer", variant="primary")
        out_answer = gr.Markdown()
        out_ctx = gr.Textbox(label="Retrieved context", lines=8)
        answer_btn.click(
            _answer, inputs=[question, harness], outputs=[out_answer, out_ctx]
        )

    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
