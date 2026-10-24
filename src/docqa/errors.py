"""统一的业务异常。

接口层只需要认这几种异常，就能给出稳定、可读的错误响应，
不用在每个路由里写 try/except 拼错误信息。
"""

from __future__ import annotations


class DocQAError(Exception):
    code = "docqa_error"
    http_status = 500

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigError(DocQAError):
    code = "config_error"
    http_status = 500


class IngestError(DocQAError):
    code = "ingest_error"
    http_status = 400


class UnsupportedFileType(IngestError):
    code = "unsupported_file_type"


class EmptyDocument(IngestError):
    code = "empty_document"


class IndexEmpty(DocQAError):
    code = "index_empty"
    http_status = 409


class LLMError(DocQAError):
    code = "llm_error"
    http_status = 502


class RetrievalError(DocQAError):
    code = "retrieval_error"
    http_status = 500


class RateLimited(DocQAError):
    code = "rate_limited"
    http_status = 429
