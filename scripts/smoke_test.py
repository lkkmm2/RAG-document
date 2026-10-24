"""接口冒烟测试：把 10 个端点挨个打一遍。

    python scripts/smoke_test.py
    python scripts/smoke_test.py --base http://127.0.0.1:8000

为什么需要它：
单元测试用 TestClient 走的是进程内调用，不走真实 HTTP 栈。
这个脚本用 requests 打真实端口，能覆盖 Uvicorn 的编码、SSE 缓冲、
CORS、中间件顺序这些"只有真起服务才会暴露"的问题。
（Windows 上用 PowerShell 发中文 JSON 很容易踩到 Latin-1 编码坑，所以用 Python 写。）
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

OK = "✅"
FAIL = "❌"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocQA 接口冒烟测试")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="学生申请转专业需要满足什么条件？")
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")
    passed = failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"{OK} {name} {detail}")
        else:
            failed += 1
            print(f"{FAIL} {name} {detail}")

    # 1) 健康检查
    resp = requests.get(f"{base}/api/v1/health", timeout=15)
    health = resp.json()
    check("GET /health", resp.status_code == 200 and health["status"] == "ok",
          f"llm={health.get('llm_provider')} chunks={health.get('chunks')}")

    # 2) 配置（不能泄露密钥）
    resp = requests.get(f"{base}/api/v1/config", timeout=15)
    check("GET /config 不含密钥", resp.status_code == 200 and "api_key" not in resp.text.lower())
    cfg = resp.json()
    print(f"   embedding={cfg['embedding']['model']} dim={cfg['embedding']['dim']} "
          f"rerank={cfg['rerank']['enabled']} fusion={cfg['retrieval']['fusion']}")

    # 3) 文档与索引统计
    docs = requests.get(f"{base}/api/v1/documents", timeout=15).json()
    stats = requests.get(f"{base}/api/v1/stats", timeout=15).json()
    check("GET /documents", len(docs) >= 1, f"{len(docs)} 篇")
    check("GET /stats", stats["chunks"] > 0, f"{stats['chunks']} 块（{stats['embedding_model']}）")

    # 4) 纯检索
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base}/api/v1/search",
        json={"question": args.question, "top_k": 3, "debug": True},
        timeout=120,
    )
    latency = (time.perf_counter() - t0) * 1000
    search = resp.json()
    check("POST /search", resp.status_code == 200 and len(search["contexts"]) > 0,
          f"{len(search['contexts'])} 条上下文 / {len(search['candidate_ids'])} 候选 / {latency:.0f}ms")
    for ctx in search["contexts"][:3]:
        scores = ctx["scores"]
        print(f"   [{ctx['index']}] p{ctx['page']} {ctx['section'][:28]} "
              f"vec={scores.get('vector')} bm25={scores.get('bm25')} rerank={scores.get('rerank')}")

    # 5) 问答
    resp = requests.post(f"{base}/api/v1/ask", json={"question": args.question, "top_k": 3}, timeout=180)
    answer = resp.json()
    check("POST /ask", resp.status_code == 200 and bool(answer["answer"]),
          f"引用 {len(answer['citations'])} 条 / {answer['latency_ms']:.0f}ms / trace={answer['trace_id']}")
    check("引用编号合法", not answer["citation_stats"]["invalid"], str(answer["citation_stats"]["invalid"]))
    print(f"   答案：{answer['answer'][:100].replace(chr(10), ' ')}")

    # 6) 流式
    events: list[dict] = []
    with requests.post(
        f"{base}/api/v1/ask/stream", json={"question": args.question, "top_k": 3},
        stream=True, timeout=180,
    ) as resp:
        check("POST /ask/stream 状态码", resp.status_code == 200,
              f"content-type={resp.headers.get('content-type')}")
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:].strip()
            if data == "[DONE]":
                break
            events.append(json.loads(data))
    kinds = [e["type"] for e in events]
    check("SSE 事件序列", "retrieval" in kinds and "delta" in kinds and kinds[-1] == "done",
          f"{len(events)} 个事件：{sorted(set(kinds))}")

    # 7) Agent
    resp = requests.post(f"{base}/api/v1/agent", json={"task": args.question, "max_steps": 3}, timeout=180)
    agent = resp.json()
    check("POST /agent", resp.status_code == 200 and bool(agent["answer"]),
          f"{len(agent['steps'])} 步 / 终止={agent['stopped_reason']}")

    # 8) 错误码
    check("DELETE 不存在文档 → 404",
          requests.delete(f"{base}/api/v1/documents/nope", timeout=15).status_code == 404)
    check("空问题 → 422",
          requests.post(f"{base}/api/v1/ask", json={"question": ""}, timeout=15).status_code == 422)

    # 9) 指标
    metrics = requests.get(f"{base}/api/v1/metrics", timeout=15).json()
    check("GET /metrics", metrics["requests"] > 0,
          f"{metrics['requests']} 次请求 / 平均 {metrics['avg_latency_ms']}ms")

    print(f"\n通过 {passed} 项，失败 {failed} 项")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
