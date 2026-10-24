"""FastAPI 应用入口：中间件、异常处理、路由挂载。

中间件只做三件事，但都是面试会问的"工程化"：
1. 给每个请求分配 trace_id，并回写响应头——出问题能按 id 把日志串起来；
2. 记录耗时、状态码、token 用量；
3. 进程内令牌桶限流，防止前端误刷新把 API 额度打穿。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from docqa.api.routes import router
from docqa.config import get_settings
from docqa.errors import DocQAError
from docqa.logging_setup import get_logger, get_trace_id, set_trace_id, setup_logging
from docqa.services import build_services

log = get_logger("api")

settings = get_settings()
setup_logging(settings.api.log_level, settings.api.log_format)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services()
        log.info("服务就绪：http://%s:%s/docs", settings.api.host, settings.api.port)
        yield
        log.info("服务关闭")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="可评测、可观测的 RAG 文档问答系统：混合检索 + 精排 + Agent",
        docs_url=None,   # 用自建 Swagger 页面（走国内 CDN，避免白屏）
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id", "X-Process-Time"],
    )

    app.state.hits: defaultdict[str, deque[float]] = defaultdict(deque)
    app.state.metrics = {"requests": 0, "errors": 0, "total_ms": 0.0}

    @app.middleware("http")
    async def observability(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or set_trace_id()
        started = time.perf_counter()

        if request.url.path.startswith("/api/"):
            limited = _check_rate_limit(app, request)
            if limited:
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": "rate_limited",
                        "message": f"请求过于频繁，限制 {settings.api.rate_limit_per_min}/分钟",
                        "trace_id": trace_id,
                    },
                    headers={"X-Trace-Id": trace_id},
                )

        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            app.state.metrics["errors"] += 1
            raise

        elapsed = (time.perf_counter() - started) * 1000
        app.state.metrics["requests"] += 1
        app.state.metrics["total_ms"] += elapsed
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Process-Time"] = f"{elapsed:.1f}ms"
        log.info(
            "%s %s -> %s (%.0fms)",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response

    # ---- 统一异常 → 统一错误结构 ----
    @app.exception_handler(DocQAError)
    async def handle_docqa_error(request: Request, exc: DocQAError):  # noqa: ARG001
        log.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code,
                "message": exc.message,
                "trace_id": get_trace_id(),
                "detail": exc.detail,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):  # noqa: ARG001
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "请求参数不合法",
                "trace_id": get_trace_id(),
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException):  # noqa: ARG001
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": "http_error",
                "message": str(exc.detail),
                "trace_id": get_trace_id(),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):  # noqa: ARG001
        log.exception("未处理异常：%s", exc)
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": str(exc), "trace_id": get_trace_id()},
        )

    # ---- 自建 Swagger（不走默认 CDN）----
    @app.get("/docs", include_in_schema=False)
    async def custom_docs() -> HTMLResponse:
        return HTMLResponse(
            """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>DocQA API Docs</title>
  <link rel="stylesheet" href="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui.css" />
  <style>body{margin:0;background:#fafafa}</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      persistAuthorization: true
    });
  </script>
</body>
</html>"""
        )

    app.include_router(router)
    return app


def _check_rate_limit(app: FastAPI, request: Request) -> bool:
    limit = settings.api.rate_limit_per_min
    if limit <= 0:
        return False
    client = request.headers.get("X-Forwarded-For") or (
        request.client.host if request.client else "unknown"
    )
    client = client.split(",")[0].strip()
    window = app.state.hits[client]
    now = time.time()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        return True
    window.append(now)
    return False


app = create_app()
