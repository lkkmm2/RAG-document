"""配置层：YAML 声明默认值，环境变量覆盖，`.env` 放密钥。

优先级（高 → 低）：

    显式传参  >  DOCQA_* 环境变量  >  .env  >  configs/config.yaml  >  代码默认值

这样做的原因：
- 分块大小、Top-K 这类"实验参数"写在 YAML 里，方便做对比实验时改一行；
- 密钥、端口这类"环境相关"的东西走环境变量，不进仓库；
- 三层都有默认值，所以删掉 YAML 或者没有 .env 也能跑起来。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# src/docqa/config.py -> 仓库根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("DOCQA_CONFIG", str(ROOT_DIR / "configs" / "config.yaml")))


class LLMSettings(BaseModel):
    """大模型配置。任何 OpenAI 兼容服务都能用，把 base_url 换掉即可。"""

    provider: Literal["deepseek", "mock"] = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: float = 60.0
    # 多轮改写、Agent 这类辅助调用可以单独设一个更便宜/更快的模型
    aux_model: str = ""


class EmbeddingSettings(BaseModel):
    provider: Literal["local", "hashing"] = "local"
    model_name: str = "BAAI/bge-small-zh-v1.5"
    model_dir: str = "data/models"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    # bge 系列官方建议：query 加指令前缀，passage 不加（非对称检索）
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    local_files_only: bool = False


class RerankSettings(BaseModel):
    enabled: bool = True
    provider: Literal["cross_encoder", "none"] = "cross_encoder"
    model_name: str = "BAAI/bge-reranker-base"
    model_dir: str = "data/models"
    batch_size: int = 8
    max_length: int = 512


class ChunkingSettings(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 80
    parent_size: int = 1200
    min_chars: int = 40
    # 递归切分的分隔符优先级：先按段落，再按句号，最后才硬切
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )


class IngestSettings(BaseModel):
    max_pages: int = 400
    extract_tables: bool = False
    dedupe_header_footer: bool = True
    header_footer_ratio: float = 0.6


class RetrievalSettings(BaseModel):
    # 消融实验开关：只留一路召回，用来量化"混合检索到底带来多少提升"
    enable_vector: bool = True
    enable_bm25: bool = True
    vector_top_k: int = 20
    bm25_top_k: int = 20
    candidate_top_k: int = 20
    final_top_k: int = 4
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    # 1.0 = 关闭 MMR 去冗余；<1 时按词面 Jaccard 做多样性惩罚
    mmr_lambda: float = 1.0
    max_context_chars: int = 4000
    rewrite_query: bool = True
    rewrite_history_turns: int = 4


class StoreSettings(BaseModel):
    index_dir: str = "data/index"
    collection: str = "docqa_chunks"
    space: str = "cosine"


class AgentSettings(BaseModel):
    max_steps: int = 5
    max_tool_calls: int = 8
    no_progress_threshold: int = 2
    temperature: float = 0.0


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_min: int = 120
    max_upload_mb: int = 40
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"


class EvalSettings(BaseModel):
    qa_path: str = "data/eval/qa_set.jsonl"
    report_dir: str = "data/eval/reports"


class YamlSettingsSource(PydanticBaseSettingsSource):
    """把 configs/config.yaml 接进 pydantic-settings 的取值链。"""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ARG002
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 配置写坏了不应该炸掉服务
            return {}
        return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCQA_",
        env_nested_delimiter="__",
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "DocQA · RAG 文档问答"
    app_version: str = "2.0.0"
    docs_dir: str = "data/docs"
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 越靠前优先级越高
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls, CONFIG_PATH),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _fallback_secret(self) -> Settings:
        """兼容 DEEPSEEK_API_KEY 这种更常见的写法，省得每次查文档。"""
        if not self.llm.api_key:
            self.llm.api_key = os.getenv("DEEPSEEK_API_KEY", "") or self.llm.api_key
        if self.llm.provider == "deepseek" and not self.llm.api_key:
            # 没有 Key 不应该让整个服务起不来，降级成 mock 并打日志
            self.llm.provider = "mock"
        return self

    # ---- 路径工具：YAML 里写相对路径，运行时统一按仓库根解析 ----
    @property
    def index_dir(self) -> Path:
        return (ROOT_DIR / self.store.index_dir).resolve()

    @property
    def docs_path(self) -> Path:
        return (ROOT_DIR / self.docs_dir).resolve()

    @property
    def model_dir(self) -> Path:
        return (ROOT_DIR / self.embedding.model_dir).resolve()

    @property
    def qa_path(self) -> Path:
        return (ROOT_DIR / self.eval.qa_path).resolve()

    @property
    def report_dir(self) -> Path:
        return (ROOT_DIR / self.eval.report_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试里可以 get_settings.cache_clear() 后重读。"""
    return Settings()


# 便捷导入：from docqa.config import settings
settings = get_settings()
