"""接口层测试：真实走 HTTP，覆盖正常路径、流式、错误码。"""

from __future__ import annotations

import json


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["llm_provider"] == "mock"
    assert payload["chunks"] > 0


def test_config_exposes_no_secrets(client):
    resp = client.get("/api/v1/config")
    assert resp.status_code == 200
    body = resp.text
    assert "api_key" not in body.lower()
    assert resp.json()["embedding"]["provider"] == "hashing"


def test_documents_and_stats(client, ingested):
    docs = client.get("/api/v1/documents").json()
    assert any(d["doc_id"] == ingested.doc_id for d in docs)

    stats = client.get("/api/v1/stats").json()
    assert stats["chunks"] >= ingested.chunks


def test_upload_text_document(client):
    content = (
        "# 图书馆规则\n\n"
        "第一条 本科生每次可借阅图书十册，借期三十天。\n"
        "第二条 逾期未还的，按每册每天零点一元收取滞纳金。\n"
    ).encode()
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("图书馆.md", content, "text/markdown")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["chunks"] > 0
    assert payload["doc_name"] == "图书馆.md"


def test_upload_rejects_unsupported_type(client):
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("a.exe", b"binary", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_ask_returns_answer_and_citations(client):
    resp = client.post("/api/v1/ask", json={"question": "学籍异动怎么申请？", "top_k": 3})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["answer"]
    assert payload["citations"], "应该带回引用来源"
    assert payload["trace_id"]
    assert payload["latency_ms"] >= 0


def test_ask_validates_empty_question(client):
    resp = client.post("/api/v1/ask", json={"question": ""})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_search_endpoint_returns_debug(client):
    resp = client.post(
        "/api/v1/search", json={"question": "转专业需要什么条件？", "top_k": 3, "debug": True}
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["contexts"]
    assert payload["debug"]["vector"] is not None
    assert payload["debug"]["timings_ms"]


def test_ask_stream_emits_sse_events(client):
    with client.stream(
        "POST", "/api/v1/ask/stream", json={"question": "休学最长可以多久？", "top_k": 3}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    assert "data: " in body
    assert "[DONE]" in body
    events = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    kinds = [e["type"] for e in events]
    assert "retrieval" in kinds
    assert "delta" in kinds
    assert kinds[-1] == "done"


def test_agent_endpoint_runs_loop(client):
    resp = client.post("/api/v1/agent", json={"task": "这本书里休学最多几年？", "max_steps": 3})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["answer"]
    assert payload["steps"]
    assert payload["stopped_reason"]


def test_delete_missing_document_returns_404(client):
    resp = client.delete("/api/v1/documents/not-exist")
    assert resp.status_code == 404


def test_metrics_endpoint(client):
    payload = client.get("/api/v1/metrics").json()
    assert payload["requests"] >= 1
    assert "index" in payload
