"""
复现脚本 v2：通过 max-queued-requests + 并发压力触发 _abort_on_queued_limit
这是更精确触发 "快路径 AbortReq vs 慢路径 BatchStrOutput" 竞态的方式。

服务启动命令（关键参数）：
  python -m sglang.launch_server \
      --model /share/global/models/Qwen3-30B-A3B \
      --port 30000 \
      --mem-fraction-static 0.6 \
      --max-running-requests 4 \
      --max-queued-requests 6 \
      --tp 4

  说明：
    --max-running-requests 4  : GPU 同时只跑 4 个请求，KV cache 快速打满
    --max-queued-requests 6   : 等待队列最多 6 个，超出触发 _abort_on_queued_limit
    --mem-fraction-static 0.6 : 减少 KV cache，更快触发 retract

运行本脚本：
  python reproduce_state_race_v2.py
"""

import asyncio
import time
import sys
import json
import aiohttp

BASE_URL = "http://localhost:30000"
MODEL = "/share/global/models/Qwen3-30B-A3B"

# 足够长的 prompt，保证请求需要很多 decode 步骤，让 detokenizer 积压输出
LONG_PROMPT = (
    "你是一位资深工程师，请详细讲解以下30个技术主题，每个主题至少写300字，"
    "包括原理、使用场景和代码示例：\n" +
    "\n".join([f"{i+1}. 主题{i+1}：{'技术概念' * 10}" for i in range(30)])
)


async def wait_for_server(timeout=300):
    print("等待服务就绪", end="", flush=True)
    async with aiohttp.ClientSession() as session:
        for _ in range(timeout // 3):
            try:
                async with session.get(
                    f"{BASE_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as r:
                    if r.status == 200:
                        print(" OK")
                        return True
            except Exception:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(3)
    print(" 超时")
    return False


async def send_streaming_request(session, req_id, results):
    """发送流式请求并收集结果"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": 2000,
        "stream": True,
        "temperature": 0.1,
    }
    t0 = time.time()
    tokens = 0
    http_status = None
    finish_reason = None
    err = None

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            http_status = resp.status
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        tokens += 1
                    fr = chunk["choices"][0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                except Exception:
                    pass
    except Exception as e:
        err = type(e).__name__

    elapsed = time.time() - t0
    results[req_id] = dict(status=http_status, tokens=tokens,
                           finish=finish_reason, elapsed=elapsed, err=err)

    tag = f"HTTP {http_status}" if http_status else f"ERR:{err}"
    print(f"  [{req_id:02d}] {tag:12s} tokens={tokens:4d}  t={elapsed:.1f}s  finish={finish_reason}")


async def main():
    if not await wait_for_server():
        sys.exit(1)

    print()
    print("=" * 65)
    print("复现目标：TokenizerManager state race condition")
    print()
    print("触发路径：")
    print("  并发请求 → KV cache 打满 → retract_decode")
    print("         → _abort_on_queued_limit (队列满) 或 reqs_to_abort (极端OOM)")
    print("         → AbortReq 走快路径 → 状态被删")
    print("         → in-flight BatchStrOutput 到达 → 找不到状态 → 报错")
    print()
    print("成功标志（查看服务端日志）：")
    print("  修复前: ERROR  ... state was deleted in TokenizerManager")
    print("  修复后: WARNING ... state was already deleted ... expected when force-aborted")
    print("=" * 65)

    connector = aiohttp.TCPConnector(limit=200, force_close=False)

    for wave in range(1, 4):
        print(f"\n--- 第 {wave} 波：并发 40 请求 ---")
        results = {}
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(send_streaming_request(session, i, results))
                for i in range(40)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        total = len(results)
        aborted_503 = sum(1 for v in results.values() if v["status"] == 503)
        aborted_500 = sum(1 for v in results.values() if v["status"] == 500)
        ok = sum(1 for v in results.values() if v["status"] == 200)
        print(f"  统计: 总={total}, 成功={ok}, 503(队列满)={aborted_503}, 500(OOM)={aborted_500}")

        if aborted_503 > 0 or aborted_500 > 0:
            print("  ✓ 有请求被 abort，race condition 应该已触发，检查服务端日志")

        await asyncio.sleep(1)

    await connector.close()
    print("\n脚本执行完毕，请查看服务端日志。")


if __name__ == "__main__":
    asyncio.run(main())
