"""Streamlit 前端：只负责交互与展示，所有能力都通过 HTTP 调后端。

这样拆的好处（面试会被问"为什么不做成一个 Streamlit 单文件"）：
- 前端可以随时换（Streamlit / Vue / 微信小程序），后端接口不变；
- 流式输出、引用、调试信息都是接口协议的一部分，不是 UI 里的临时拼装；
- 后端能独立部署、独立压测，前端只是它的一个客户端。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("DOCQA_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 300

st.set_page_config(page_title="DocQA · 文档问答", page_icon="📚", layout="wide")


# ------------------------------------------------------------------ HTTP 封装
def api_get(path: str) -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_delete(path: str) -> None:
    requests.delete(f"{API_BASE}{path}", timeout=30).raise_for_status()


def api_upload(filename: str, data: bytes) -> dict[str, Any]:
    files = {"file": (filename, data, "application/octet-stream")}
    resp = requests.post(f"{API_BASE}/api/v1/documents/upload", files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def stream_answer(question: str, history: list[dict], top_k: int) -> Iterator[dict[str, Any]]:
    payload = {"question": question, "history": history, "top_k": top_k}
    with requests.post(
        f"{API_BASE}/api/v1/ask/stream", json=payload, stream=True, timeout=TIMEOUT
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:].strip()
            if data == "[DONE]":
                break
            yield json.loads(data)


# ------------------------------------------------------------------ 展示组件
def render_citations(citations: list[dict[str, Any]] | None) -> None:
    if not citations:
        return
    st.markdown("**📎 引用来源**")
    cols = st.columns(min(len(citations), 2))
    for i, cite in enumerate(citations):
        with cols[i % len(cols)]:
            label = f"[{cite['index']}] {cite['doc_name']}"
            if cite.get("section"):
                label += f" · {cite['section'][:24]}"
            with st.expander(label, expanded=False):
                st.caption(
                    f"第 {cite.get('page', 0)} 页 · chunk `{cite['chunk_id']}` · "
                    f"相关度 {cite.get('score', 0):.3f}"
                )
                st.write(cite.get("snippet", ""))


def render_debug(debug: dict[str, Any] | None) -> None:
    if not debug:
        return
    with st.expander("🔍 检索过程（向量 / BM25 / 融合 / 精排）", expanded=False):
        if debug.get("rewritten_query") and debug["rewritten_query"] != debug.get("original_query"):
            st.info(f"查询改写：{debug['original_query']} → **{debug['rewritten_query']}**")
        tabs = st.tabs(["向量召回", "BM25 召回", "RRF 融合", "精排结果"])
        for tab, key in zip(tabs, ["vector", "bm25", "fused", "rerank"], strict=False):
            with tab:
                rows = debug.get(key) or []
                if rows:
                    st.dataframe(
                        [
                            {
                                "chunk": r["chunk_id"],
                                "分数": round(r["score"], 4),
                                "排名": r["rank"],
                                **{k: v for k, v in (r.get("scores") or {}).items()},
                            }
                            for r in rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("（空）")
        timings = debug.get("timings_ms") or {}
        if timings:
            st.caption(
                "耗时：" + " · ".join(f"{k} {v:.0f}ms" for k, v in timings.items())
            )


def render_agent(resp: dict[str, Any]) -> None:
    st.markdown(resp.get("answer", ""))
    steps = resp.get("steps") or []
    if steps:
        with st.expander(f"🤖 Agent 执行过程（{len(steps)} 步，终止原因：{resp.get('stopped_reason')}）"):
            for step in steps:
                st.markdown(
                    f"**第 {step['step']} 步** · `{step.get('action', '-')}` "
                    f"({step.get('latency_ms', 0):.0f}ms)"
                )
                if step.get("thought"):
                    st.caption(f"思考：{step['thought']}")
                if step.get("action_input"):
                    st.code(json.dumps(step["action_input"], ensure_ascii=False), language="json")
                if step.get("observation"):
                    st.text(step["observation"][:500])
                if step.get("error"):
                    st.error(step["error"])
                st.divider()
    render_citations(resp.get("citations"))


def render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander(f"🤖 Agent 执行过程（{len(msg['steps'])} 步）"):
                for step in msg["steps"]:
                    st.markdown(f"**第 {step['step']} 步** · `{step.get('action', '-')}`")
                    if step.get("thought"):
                        st.caption(step["thought"])
                    if step.get("observation"):
                        st.text(step["observation"][:400])
        render_citations(msg.get("citations"))
        if msg.get("debug"):
            render_debug(msg["debug"])
        if msg.get("stats"):
            stats = msg["stats"]
            st.caption(
                f"⏱ {stats.get('latency_ms', 0):.0f}ms · "
                f"引用覆盖 {stats.get('citation_coverage', 0):.0%} · "
                f"tokens {stats.get('total_tokens', 0)}"
            )


# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("📚 DocQA")
    health = api_get("/api/v1/health")
    if health:
        st.success(f"✅ 服务在线 v{health.get('version', '?')}")
        st.caption(
            f"LLM `{health.get('llm_provider')}/{health.get('llm_model')}`  \n"
            f"Embedding `{health.get('embedding_model')}`  \n"
            f"Rerank `{'on' if health.get('rerank_enabled') else 'off'}`  \n"
            f"索引 {health.get('documents', 0)} 篇 / {health.get('chunks', 0)} 块"
        )
    else:
        st.error("❌ 后端未启动")
        st.code(
            ".venv\\Scripts\\python -m uvicorn docqa.api.main:app --port 8000",
            language="powershell",
        )
        st.stop()

    st.divider()
    st.subheader("📄 文档")
    docs = api_get("/api/v1/documents") or []
    if docs:
        for doc in docs:
            cols = st.columns([4, 1])
            cols[0].caption(f"{doc['doc_name']}（{doc['chunks']} 块）")
            if cols[1].button("🗑", key=f"del-{doc['doc_id']}", help="删除该文档"):
                api_delete(f"/api/v1/documents/{doc['doc_id']}")
                st.rerun()
    else:
        st.caption("还没有文档，先上传一份")

    uploaded = st.file_uploader("上传 PDF / TXT / Markdown", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("入库", type="primary"):
        with st.spinner("解析 + 切分 + 向量化..."):
            try:
                result = api_upload(uploaded.name, uploaded.getvalue())
                st.success(
                    f"{result['doc_name']}：{result['pages']} 页 / "
                    f"{result['chunks']} 块 / {result['elapsed_ms']:.0f}ms"
                )
                for warning in result.get("warnings") or []:
                    st.warning(warning)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"入库失败：{exc}")

    st.divider()
    st.subheader("⚙️ 参数")
    mode = st.radio("模式", ["RAG 问答", "Agent 多步"], horizontal=True)
    top_k = st.slider("Top-K 上下文", 1, 10, 4)
    debug = st.checkbox("显示检索过程", value=False)
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ 主区域
st.title("文档问答助手")
st.caption("混合检索（向量 + BM25）→ RRF 融合 → Cross-Encoder 精排 → 带引用的回答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("问点什么，例如：学籍异动需要什么材料？")
if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ][-6:]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if mode == "Agent 多步":
                with st.spinner("Agent 正在规划与调用工具..."):
                    resp = api_post("/api/v1/agent", {"task": question})
                render_agent(resp)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": resp.get("answer", ""),
                        "citations": resp.get("citations"),
                        "steps": resp.get("steps"),
                    }
                )
            elif debug:
                # 需要完整 debug 字段，所以走非流式端点
                with st.spinner("检索并生成..."):
                    resp = api_post(
                        "/api/v1/ask",
                        {"question": question, "history": history, "top_k": top_k, "debug": True},
                    )
                st.markdown(resp["answer"])
                render_debug(resp.get("debug"))
                render_citations(resp.get("citations"))
                stats = {
                    "latency_ms": resp.get("latency_ms", 0),
                    "citation_coverage": (resp.get("citation_stats") or {}).get("coverage", 0),
                    "total_tokens": (resp.get("usage") or {}).get("total_tokens", 0),
                }
                st.caption(
                    f"⏱ {stats['latency_ms']:.0f}ms · "
                    f"引用覆盖 {stats['citation_coverage']:.0%} · "
                    f"tokens {stats['total_tokens']}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": resp["answer"],
                        "citations": resp.get("citations"),
                        "debug": resp.get("debug"),
                        "stats": stats,
                    }
                )
            else:
                placeholder = st.empty()
                text = ""
                citations: list[dict[str, Any]] = []
                stats: dict[str, Any] = {}
                for event in stream_answer(question, history, top_k):
                    if event["type"] == "retrieval":
                        citations = event.get("citations") or []
                    elif event["type"] == "delta":
                        text += event["content"]
                        placeholder.markdown(text + "▌")
                    elif event["type"] == "done":
                        stats = {
                            "latency_ms": event.get("latency_ms", 0),
                            "citation_coverage": (event.get("citation_stats") or {}).get("coverage", 0),
                            "total_tokens": 0,
                        }
                    elif event["type"] == "error":
                        st.error(event.get("message", "生成失败"))
                placeholder.markdown(text or "_（没有返回内容）_")
                render_citations(citations)
                st.caption(
                    f"⏱ {stats.get('latency_ms', 0):.0f}ms · "
                    f"引用覆盖 {stats.get('citation_coverage', 0):.0%}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "citations": citations,
                        "stats": stats,
                    }
                )
        except requests.HTTPError as exc:
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            st.error(f"请求失败：{detail}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"出错了：{exc}")
