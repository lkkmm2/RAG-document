# legacy/ —— v1 旧代码（仅作对照，不参与运行）

这四个文件是项目第一版，**已不参与任何运行流程**，保留原因只有一个：
面试时可以对照讲"我发现了什么问题、改成了什么"。

| 文件 | v1 里是什么 | v2 改成了什么 |
|---|---|---|
| `app_v1.py` | Streamlit 单文件，自己 import OpenAI 直接调模型 | 前端只调 HTTP，后端 `src/docqa/api/` |
| `api_v1.py` | FastAPI 另写一份上传/检索逻辑，和 app.py 平行 | `src/docqa/` 分层，`services.py` 单一装配 |
| `testapi.py` | 手工 requests 脚本 | `tests/` 45 个 pytest + `scripts/smoke_test.py` |
| `pdfreadtest.py` | 打印 PDF 前 500 字，手工排查 | `loader.py` + `cleaner.py` + 入库警告 |

## v1 的具体问题（面试可讲）

1. **两套平行实现**：`app.py` 和 `api.py` 各写一份切分/检索，README 还写着"前后端分离"，
   但 `app.py` 从未请求过 `api.py`。改一处漏一处。
2. **切分**：`app.py` 按 `\n` 累加 300 字无重叠，`api.py` 定长 300 切 50。
   两者不一致，且都会切断句子、跨越章节。
3. **向量化**：Chroma 默认 embedding（`all-MiniLM-L6-v2`，英文语料训练），中文召回很差。
4. **引用**：只返回整段文本，没有页码、没有 chunk id，无法核对。
5. **密钥**：`api_key=""` 硬编码在源码里，且进过 git 历史。
6. **工程**：无测试、无日志、无 Docker、`requirements.txt` 只有 4 行且不锁版本。
7. **模板残留**：0 字节的 `images/*.png`、空的 `notebooks/01_eda.ipynb`、
   与 RAG 无关的 `configs/config.yaml`（`batch_size/epochs/lr`）。

> 这些文件保留为 `.py.txt` 之外的普通 `.py`，但**不要** import 它们，
> 也不要让它们进 ruff/pytest 的扫描范围（`pyproject.toml` 已限定 `src tests scripts apps`）。
