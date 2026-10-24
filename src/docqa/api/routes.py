"""REST 接口。

接口设计上的几个决定：
- 版本前缀 `/api/v1`：后面加 breaking change 时有地方放；
- 上传用 multipart，同步返回入库结果，不搞异步任务——单机场景没必要，反而增加运维成本；
- 流式问答单独一个端点（SSE），因为"首 token 时间"和"完整响应"是两个不同的产品需求；
- 额外给一个 `/search` 纯检索端点：不花 token 就能验证召回效果，调试和评测都用它。
"""

from __future__ import annotations

import json
from functools import partial
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from docqa.errors import EmptyDocument, IndexEmpty
from docqa.logging_setup import get_logger, get_trace_id
from docqa.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    DocumentInfo,
    EvalCase,
    HealthResponse,
    IndexStats,
    RetrievalResult,
    UploadResponse,
)
from docqa.services import Services

log = get_logger("api.routes")
router = APIRouter(prefix="/api/v1", tags=["docqa"])


def _svc(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:  # 理论上不会发生，lifespan 已经装配好
        raise HTTPException(status_code=503, detail="服务尚未初始化")
    return services


# ------------------------------------------------------------------ 健康与配置
@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(request: Request) -> HealthResponse:
    return _svc(request).health()


@router.get("/config", summary="返回前端需要的非敏感配置")
async def config(request: Request) -> dict[str, Any]:
    svc = _svc(request)
    return {
        "app_name": svc.settings.app_name,
        "version": svc.settings.app_version,
        "llm": {"provider": svc.llm.name, "model": svc.llm.model},
        "embedding": {
            "provider": svc.settings.embedding.provider,
            "model": getattr(svc.embedder, "name", ""),
            "dim": getattr(svc.embedder, "dim", 0),
        },
        "rerank": {"enabled": svc.reranker.enabled, "model": svc.settings.rerank.model_name},
        "retrieval": {
            "vector_top_k": svc.settings.retrieval.vector_top_k,
            "bm25_top_k": svc.settings.retrieval.bm25_top_k,
            "final_top_k": svc.settings.retrieval.final_top_k,
            "fusion": svc.settings.retrieval.fusion,
        },
        "chunking": {
            "chunk_size": svc.settings.chunking.chunk_size,
            "chunk_overlap": svc.settings.chunking.chunk_overlap,
            "parent_size": svc.settings.chunking.parent_size,
        },
    }


@router.get("/metrics", summary="简易运行指标")
async def metrics(request: Request) -> dict[str, Any]:
    state = request.app.state.metrics
    requests = max(state["requests"], 1)
    return {
        "requests": state["requests"],
        "errors": state["errors"],
        "avg_latency_ms": round(state["total_ms"] / requests, 2),
        "index": _svc(request).repo.stats().model_dump(),
    }


# ------------------------------------------------------------------ 文档管理
@router.get("/documents", response_model=list[DocumentInfo], summary="文档列表")
async def list_documents(request: Request) -> list[DocumentInfo]:
    return _svc(request).repo.list_documents()


@router.get("/stats", response_model=IndexStats, summary="索引统计")
async def stats(request: Request) -> IndexStats:
    return _svc(request).repo.stats()


@router.post("/documents/upload", response_model=UploadResponse, summary="上传并入库")
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    svc = _svc(request)
    filename = file.filename or "uploaded"
    if not filename.lower().endswith((".pdf", ".txt", ".md", ".markdown")):
        raise HTTPException(status_code=400, detail="只支持 PDF / TXT / Markdown")

    data = await file.read()
    max_bytes = svc.settings.api.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"文件超过 {svc.settings.api.max_upload_mb}MB 限制"
        )

    try:
        response = await run_in_threadpool(svc.ingest.ingest_bytes, filename, data)
    except ValueError as exc:
        raise EmptyDocument(str(exc)) from exc

    await run_in_threadpool(svc.refresh)
    return response


@router.delete("/documents/{doc_id}", summary="删除文档")
async def delete_document(request: Request, doc_id: str) -> dict[str, Any]:
    svc = _svc(request)
    deleted = await run_in_threadpool(svc.ingest.delete, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")
    await run_in_threadpool(svc.refresh)
    return {"deleted": doc_id}


# ------------------------------------------------------------------ 检索与问答
@router.post("/search", response_model=RetrievalResult, summary="纯检索（不调用大模型）")
async def search(request: Request, payload: AskRequest) -> RetrievalResult:
    svc = _svc(request)
    if svc.retriever.chunk_count() == 0:
        raise IndexEmpty("知识库为空，请先上传文档")
    return await run_in_threadpool(
        partial(
            svc.retriever.retrieve,
            payload.question,
            history=payload.history,
            top_k=payload.top_k,
            debug=True,
        )
    )


@router.post("/ask", response_model=AskResponse, summary="RAG 问答")
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    svc = _svc(request)
    if svc.retriever.chunk_count() == 0:
        raise IndexEmpty("知识库为空，请先上传文档")
    return await run_in_threadpool(svc.answerer.answer, payload)


@router.post("/ask/stream", summary="RAG 问答（SSE 流式）")
async def ask_stream(request: Request, payload: AskRequest) -> StreamingResponse:
    svc = _svc(request)
    if svc.retriever.chunk_count() == 0:
        raise IndexEmpty("知识库为空，请先上传文档")

    def event_stream():
        try:
            for event in svc.answerer.stream(
                payload.question, payload.history, payload.top_k
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("流式问答失败")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲，否则流式会变成一次性返回
            "X-Trace-Id": get_trace_id(),
        },
    )


@router.post("/agent", response_model=AgentResponse, summary="ReAct Agent（多步工具调用）")
async def run_agent(request: Request, payload: AgentRequest) -> AgentResponse:
    svc = _svc(request)
    if svc.retriever.chunk_count() == 0:
        raise IndexEmpty("知识库为空，请先上传文档")
    return await run_in_threadpool(svc.agent.run, payload.task, payload.max_steps)


# ------------------------------------------------------------------ 评测辅助
@router.get("/eval/cases", response_model=list[EvalCase], summary="读取评测集")
async def eval_cases(request: Request) -> list[EvalCase]:
    path = _svc(request).settings.qa_path
    if not path.exists():
        return []
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(EvalCase.model_validate_json(line))
    return cases
