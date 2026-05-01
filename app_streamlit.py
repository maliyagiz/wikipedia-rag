"""Streamlit chat UI for the local Wikipedia RAG.

Run:
    streamlit run app_streamlit.py
"""
from __future__ import annotations

import time

import streamlit as st

from src.generator import clean_answer
from src.rag import RAG

st.set_page_config(page_title="Local Wikipedia RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag() -> RAG:
    return RAG()


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []   # list[dict(role, content, hits, routing, latency)]


_init_state()
rag = get_rag()

with st.sidebar:
    st.title("📚 Local Wikipedia RAG")
    st.caption("Ollama + numpy + mxbai-embed-large. Fully local.")
    st.metric("Indexed chunks", rag.store.count())
    show_ctx = st.toggle("Show retrieved context", value=True)
    if st.button("🧹 Clear chat"):
        st.session_state.history = []
        st.rerun()
    if st.button("⚠️ Reset vector store"):
        rag.store.reset()
        st.success("Vector store wiped. Re-run `python -m scripts.run_ingest`.")

    st.divider()
    st.markdown(
        "**Try:**\n"
        "- Who was Albert Einstein and what is he known for?\n"
        "- Where is the Eiffel Tower located?\n"
        "- Compare Albert Einstein and Nikola Tesla.\n"
        "- Which famous place is located in Turkey?\n"
        "- Who is the president of Mars?"
    )

st.title("Ask about famous people & places")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_ctx and msg.get("hits"):
            with st.expander(f"Sources ({msg['routing']}, {msg.get('latency', 0):.1f}s)"):
                for i, h in enumerate(msg["hits"], 1):
                    title = h.metadata.get("title", "?")
                    kind = h.metadata.get("type", "?")
                    st.markdown(f"**[{i}] {kind}: {title}** — distance `{h.distance:.3f}`")
                    st.text(h.text[:600] + ("..." if len(h.text) > 600 else ""))

prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        t0 = time.time()
        with st.spinner("Retrieving and generating…"):
            stream, hits, routing = rag.ask_stream(prompt)
            buf = ""
            for tok in stream:
                buf += tok
                placeholder.markdown(buf + "▌")
        latency = time.time() - t0
        buf = clean_answer(buf)
        placeholder.markdown(buf)

        if show_ctx and hits:
            with st.expander(f"Sources ({routing}, {latency:.1f}s)"):
                for i, h in enumerate(hits, 1):
                    title = h.metadata.get("title", "?")
                    kind = h.metadata.get("type", "?")
                    st.markdown(f"**[{i}] {kind}: {title}** — distance `{h.distance:.3f}`")
                    st.text(h.text[:600] + ("..." if len(h.text) > 600 else ""))

    st.session_state.history.append({
        "role": "assistant",
        "content": buf,
        "hits": hits,
        "routing": routing,
        "latency": latency,
    })
