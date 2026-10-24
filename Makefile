# DocQA 常用命令
# Windows 没装 make 的话，直接看 README 里的 PowerShell 版本

PY := .venv/Scripts/python.exe
PIP := $(PY) -m pip

.PHONY: help install ingest ask api web test lint fmt eval eval-fast demo clean reset

help:
	@echo "make install    安装依赖（含 dev）"
	@echo "make ingest     入库 data/docs 下的文档"
	@echo "make reset      清空索引后重新入库"
	@echo "make api        启动 FastAPI（:8000）"
	@echo "make web        启动 Streamlit（:8501）"
	@echo "make test       跑单元测试"
	@echo "make lint       ruff 检查"
	@echo "make eval       跑评测（含 LLM 回答）"
	@echo "make eval-fast  只评测检索（不调用 LLM，省钱）"
	@echo "make demo       一键演示：入库→检索→问答→Agent"
	@echo "make clean      清理缓存与索引"

install:
	$(PIP) install -e ".[dev,cn]" -i https://pypi.tuna.tsinghua.edu.cn/simple

ingest:
	$(PY) -m docqa.cli

reset:
	$(PY) -m docqa.cli --reset

api:
	$(PY) -m uvicorn docqa.api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	$(PY) -m streamlit run apps/web/app.py --server.port 8501

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts apps

fmt:
	$(PY) -m ruff format src tests scripts apps

eval:
	$(PY) scripts/eval.py --with-answer

eval-fast:
	$(PY) scripts/eval.py

demo:
	$(PY) scripts/demo.py

clean:
	rm -rf data/index data/eval/reports .pytest_cache .ruff_cache
