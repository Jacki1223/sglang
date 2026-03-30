"""
复现脚本：TokenizerManager state race condition
错误信息: "Received output for rid=... but the state was deleted in TokenizerManager."

原理：
  KV cache 打满 → retract_decode 触发 → AbortReq 走快路径直达 TokenizerManager
  而 in-flight 的 BatchStrOutput 走慢路径（经过 Detokenizer）
  快路径先到 → 状态删除 → 慢路径的消息到达时找不到状态 → 报错

用法：
  步骤1: 在另一个终端启动服务
    python -m sglang.launch_server \
        --model /share/global/models/Qwen3-30B-A3B \
        --port 30000 \
        --mem-fraction-static 0.6 \
        --max-running-requests 8 \
        --tp 4

  步骤2: 运行本脚本
    python reproduce_state_race.py
"""

import asyncio
import time
import sys
import aiohttp

BASE_URL = "http://localhost:30000"
MODEL = "/share/global/models/Qwen3-30B-A3B"

# 长 prompt，目的是让每个请求占用大量 KV cache
LONG_PROMPT = "请详细解释以下每个概念，每个概念写500字以上：" + "、".join([
    f"概念{i}" for i in range(30)
])


async def wait_for_server(timeout=300):
    """等待服务启动就绪"""
    print("等待服务就绪...", end="", flush=True)
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        print(" 就绪!")
                        return True
            except Exception:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(3)
    print(" 超时!")
    return False


async def send_request(session, req_id, results):
    """发送一个流式长请求"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": LONG_PROMPT}
        ],
        "max_tokens": 1500,
        "stream": True,
        "temperature": 0.7,
    }
    start = time.time()
    token_count = 0
    error = None
    status = None

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            status = resp.status
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    token_count += 1
    except asyncio.CancelledError:
        error = "cancelled"
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    results[req_id] = {
        "status": status,
        "tokens": token_count,
        "elapsed": elapsed,
        "error": error,
    }
    status_str = f"status={status}" if status else f"error={error}"
    print(f"  请求 {req_id:02d}: {status_str}, tokens={token_count}, elapsed={elapsed:.1f}s")


async def run_phase(label, n_requests, connector):
    """并发发送 n_requests 个请求"""
    print(f"\n[{label}] 并发发送 {n_requests} 个长请求...")
    results = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, i, results) for i in range(n_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
    aborted = sum(1 for v in results.values() if v["status"] in (503, 500))
    print(f"[{label}] 完成: {len(results)} 请求, 其中 {aborted} 个被 abort (503/500)")
    return results


async def main():
    # 检查服务是否就绪
    connector = aiohttp.TCPConnector(limit=100)
    tmp_session = aiohttp.ClientSession(connector=connector)
    ready = await wait_for_server()
    await tmp_session.close()

    if not ready:
        print("服务未就绪，退出")
        sys.exit(1)

    print("\n" + "="*60)
    print("开始复现 TokenizerManager state race condition")
    print("="*60)
    print("目标: 触发 KV cache 满 → retract_decode → AbortReq 快路径")
    print("观察: 服务日志中出现以下内容即为复现成功:")
    print("  修复前: ERROR ... state was deleted in TokenizerManager")
    print("  修复后: WARNING ... state was already deleted ... expected when force-aborted")
    print("="*60)

    connector = aiohttp.TCPConnector(limit=200)

    # 第一轮：发大量请求，迅速打满 KV cache
    # 目的：触发 retract_decode，并让 AbortReq 与 in-flight BatchStrOutput 产生竞争
    await run_phase("轮次1-打满KV", n_requests=30, connector=connector)

    await asyncio.sleep(2)

    # 第二轮：再来一波，确保在 retract 期间有持续输出 in-flight
    await run_phase("轮次2-持续压力", n_requests=30, connector=connector)

    await connector.close()

    print("\n" + "="*60)
    print("脚本结束。请检查服务端日志中是否出现上述 WARNING/ERROR。")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
