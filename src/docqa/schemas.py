"""跨层共享的数据结构。

统一用 Pydantic 模型有三个好处：
1. API 出入参自动校验 + 自动生成 OpenAPI；
2. 落盘（JSONL）和进程内传递用同一套结构，不用两套序列化代码；
3. 面试时能直接指着 `Citation` 说"引用是一等公民，不是拼在字符串里的"。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 文档与分块
class PageText(BaseModel):
    page: int
    text: str


class Chunk(BaseModel):
    """检索的最小单元（子块）。父块只用于喂给大模型，见 ParentChunk。"""

    id: str
    doc_id: str
    doc_name: str
    text: str
    parent_id: str
    section: str = ""
    page: int = 0
    order: int = 0
    char_start: int = 0
    char_end: int = 0


class ParentChunk(BaseModel):
    id: str
    doc_id: str
    doc_name: str
    text: str
    section: str = ""
    page: int = 0


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    pages: int = 0
    chunks: int = 0
    parents: int = 0
    chars: int = 0
    size_bytes: int = 0
    ingested_at: str = ""


class IndexStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    parents: int = 0
    embedding_model: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    updated_at: str = ""


# ---------------------------------------------------------------- 检索
class ScoredChunk(BaseModel):
    chunk_id: str
    score: float
    rank: int = 0
    source: str = ""  # vector | bm25 | fused | rerank
    scores: dict[str, float] = Field(default_factory=dict)


class ContextItem(BaseModel):
    """最终喂给大模型的一段上下文，带编号方便做引用。"""

    index: int
    chunk_id: str
    parent_id: str
    doc_name: str
    section: str = ""
    page: int = 0
    text: str
    score: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)


class RetrievalDebug(BaseModel):
    original_query: str = ""
    rewritten_query: str = ""
    vector: list[ScoredChunk] = Field(default_factory=list)
    bm25: list[ScoredChunk] = Field(default_factory=list)
    fused: list[ScoredChunk] = Field(default_factory=list)
    rerank: list[ScoredChunk] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    contexts: list[ContextItem] = Field(default_factory=list)
    # 重排后、父块扩展前的有序 chunk id。
    # 召回指标必须在"候选"上算，而不是在最终上下文上算——
    # 后者会被 top_k 截断和父块去重影响，算出来的不是检索能力。
    candidate_ids: list[str] = Field(default_factory=list)
    debug: RetrievalDebug | None = None


# ---------------------------------------------------------------- 问答
class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_name: str
    section: str = ""
    page: int = 0
    snippet: str = ""
    score: float = 0.0


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CitationStats(BaseModel):
    total: int = 0
    invalid: list[int] = Field(default_factory=list)
    coverage: float = 0.0  # 带引用的句子占比
    valid: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
    debug: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    citation_stats: CitationStats | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    trace_id: str = ""
    debug: RetrievalDebug | None = None


# ---------------------------------------------------------------- Agent
class AgentStep(BaseModel):
    step: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    error: str = ""
    latency_ms: float = 0.0


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    max_steps: int | None = Field(default=None, ge=1, le=10)


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    trace_id: str = ""
    stopped_reason: str = "final_answer"


# ---------------------------------------------------------------- 接口通用
class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    pages: int = 0
    chunks: int = 0
    parents: int = 0
    chars: int = 0
    elapsed_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    rerank_enabled: bool = False
    documents: int = 0
    chunks: int = 0


class ErrorResponse(BaseModel):
    code: str = "internal_error"
    message: str = ""
    trace_id: str = ""
    detail: Any = None


class EvalCase(BaseModel):
    id: str
    question: str
    answerable: bool = True
    gold_answer: str = ""
    gold_keywords: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_sections: list[str] = Field(default_factory=list)
    type: Literal["single_hop", "multi_hop", "table", "unanswerable"] = "single_hop"
    # 标注备注：记录"这条为什么这么标""哪里可能标偏了"，让评测集自带上下文
    note: str = ""


class EvalRow(BaseModel):
    id: str
    question: str
    retrieved_ids: list[str] = Field(default_factory=list)
    recall: float = 0.0
    recall_by_k: dict[int, float] = Field(default_factory=dict)
    reciprocal_rank: float = 0.0
    answer: str = ""
    keyword_hit: float = 0.0
    citation_precision: float = 0.0
    latency_ms: float = 0.0
    error: str = ""


class EvalReport(BaseModel):
    name: str = ""
    cases: int = 0
    cases_answerable: int = 0
    recall_at_k: float = 0.0
    recall_by_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    keyword_hit: float = 0.0
    citation_precision: float = 0.0
    refusal_accuracy: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    rows: list[EvalRow] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
