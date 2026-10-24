"""测试夹具。

关键点：**测试不能依赖网络、模型和 API Key**。
所以这里把 embedding 切成哈希实现、关掉精排、LLM 用 Mock，
所有外部依赖都被替换成确定性组件，`pytest` 应该在任何机器上 10 秒内跑完。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="docqa-test-"))

# 必须在 import docqa 之前设置，Settings 是在模块导入时构造的
os.environ["DOCQA_EMBEDDING__PROVIDER"] = "hashing"
os.environ["DOCQA_RERANK__ENABLED"] = "false"
os.environ["DOCQA_LLM__PROVIDER"] = "mock"
os.environ["DOCQA_LLM__API_KEY"] = ""
os.environ["DOCQA_API__RATE_LIMIT_PER_MIN"] = "0"
os.environ["DOCQA_STORE__INDEX_DIR"] = str(_TMP / "index")
os.environ["DOCQA_DOCS_DIR"] = str(_TMP / "docs")
os.environ["DOCQA_EVAL__QA_PATH"] = str(_TMP / "qa_set.jsonl")
os.environ["DOCQA_EVAL__REPORT_DIR"] = str(_TMP / "reports")
os.environ["DOCQA_API__LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402

SAMPLE_TEXT = """第一章 总则

第一条 为规范学校学生管理，维护正常的教育教学秩序，制定本手册。
本手册适用于全日制本科生、研究生。

第二章 学籍管理

第一节 入学与注册

第五条 新生入学后，学校在三个月内按照国家招生规定进行复查。
复查合格者予以注册，取得学籍。复查不合格者，由学校区别情况予以处理。

第六条 每学期开学时，学生应当按学校规定办理注册手续。
未按学校规定缴纳学费或者其他不符合注册条件的，不予注册。

第二节 学籍异动

第十条 学生可以申请转专业。转专业应当在第一学年结束前提出，
经所在学院同意、拟转入学院考核、教务处审批后办理。

第十一条 休学期限一般以一年为期，累计不得超过两年。
休学期满应当及时申请复学，逾期不办理复学手续的，按自动退学处理。

第三章 课程与考核

第十二条 课程考核分为考试和考查两种。考试成绩按百分制记分，六十分及格。
学生一学期内缺课累计超过该课程教学时数三分之一的，取消考核资格。
"""


@pytest.fixture(scope="session")
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture(scope="session")
def services():
    from docqa.services import build_services

    return build_services()


@pytest.fixture(scope="session")
def ingested(services):
    """整个测试会话只入库一次，后面所有检索/问答测试共用这份索引。"""
    response = services.ingest.ingest_bytes("学生手册.txt", SAMPLE_TEXT.encode("utf-8"))
    services.refresh()
    return response


@pytest.fixture(scope="session")
def client(services, ingested):  # noqa: ARG001
    from fastapi.testclient import TestClient

    from docqa.api.main import app

    # 直接复用已装配好的 services，避免 lifespan 再建一套（模型/索引会被重复加载）
    app.state.services = services
    return TestClient(app)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    shutil.rmtree(_TMP, ignore_errors=True)
