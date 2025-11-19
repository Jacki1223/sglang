#!/usr/bin/env python3
"""
SGLang调度优化 - 测试脚本

这个脚本会发送一系列请求到SGLang服务器，测试优化效果。
"""

import argparse
import time
from typing import List

import requests


def send_request(
    base_url: str, prompt: str, max_tokens: int = 100
) -> dict:
    """发送单个生成请求"""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    response.raise_for_status()
    return response.json()


def test_basic_functionality(base_url: str):
    """测试基本功能"""
    print("\n🧪 测试1: 基本功能")
    print("=" * 50)

    prompts = [
        "Once upon a time",
        "In a galaxy far far away",
        "The quick brown fox",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n  请求 {i}/{len(prompts)}: {prompt[:30]}...")
        start = time.time()
        result = send_request(base_url, prompt, max_tokens=50)
        elapsed = time.time() - start

        generated = result["choices"][0]["text"]
        print(f"  ✅ 生成: {generated[:50]}...")
        print(f"  ⏱️  耗时: {elapsed:.2f}s")

    print("\n✅ 基本功能测试通过！")


def test_concurrent_requests(base_url: str, num_requests: int = 10):
    """测试并发请求"""
    print(f"\n🧪 测试2: 并发请求 (n={num_requests})")
    print("=" * 50)

    import concurrent.futures

    prompts = [
        f"This is test request number {i}. Please generate some text."
        for i in range(num_requests)
    ]

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(send_request, base_url, prompt, 50)
            for prompt in prompts
        ]

        results = []
        for i, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            try:
                result = future.result()
                results.append(result)
                print(f"  ✅ 请求 {i}/{num_requests} 完成")
            except Exception as e:
                print(f"  ❌ 请求 {i}/{num_requests} 失败: {e}")

    total_time = time.time() - start_time

    print(f"\n📊 并发测试结果:")
    print(f"  - 成功请求: {len(results)}/{num_requests}")
    print(f"  - 总耗时: {total_time:.2f}s")
    print(f"  - 平均延迟: {total_time/num_requests:.2f}s")
    print(f"  - 吞吐量: {num_requests/total_time:.2f} req/s")

    print("\n✅ 并发测试完成！")


def test_varying_lengths(base_url: str):
    """测试不同长度的请求"""
    print("\n🧪 测试3: 不同输入/输出长度")
    print("=" * 50)

    test_cases = [
        {"prompt": "Short", "max_tokens": 10, "name": "短输入短输出"},
        {
            "prompt": "This is a medium length prompt with some context. " * 5,
            "max_tokens": 50,
            "name": "中等输入中等输出",
        },
        {
            "prompt": "This is a very long prompt with lots of context. " * 20,
            "max_tokens": 100,
            "name": "长输入长输出",
        },
    ]

    for test in test_cases:
        print(f"\n  测试: {test['name']}")
        print(f"  输入长度: ~{len(test['prompt'].split())} words")
        print(f"  最大输出: {test['max_tokens']} tokens")

        start = time.time()
        result = send_request(base_url, test["prompt"], test["max_tokens"])
        elapsed = time.time() - start

        print(f"  ✅ 完成，耗时: {elapsed:.2f}s")

    print("\n✅ 长度测试完成！")


def get_server_stats(base_url: str):
    """获取服务器统计信息（如果可用）"""
    try:
        response = requests.get(f"{base_url}/get_server_info")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  ℹ️  无法获取服务器统计: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="测试SGLang调度优化效果"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="SGLang服务器地址",
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="跳过基本功能测试",
    )
    parser.add_argument(
        "--skip-concurrent",
        action="store_true",
        help="跳过并发测试",
    )
    parser.add_argument(
        "--skip-lengths",
        action="store_true",
        help="跳过长度测试",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=10,
        help="并发请求数量",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   SGLang调度优化 - 测试脚本")
    print("=" * 60)
    print(f"\n📡 服务器地址: {args.base_url}")

    # 检查服务器是否可用
    print("\n🔍 检查服务器状态...")
    try:
        response = requests.get(f"{args.base_url}/health")
        if response.status_code == 200:
            print("✅ 服务器在线！")
        else:
            print(f"⚠️  服务器响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        print("\n请确保SGLang服务器正在运行:")
        print(f"  python -m sglang.launch_server \\")
        print(f"    --model-path YOUR_MODEL \\")
        print(f"    --enable-scheduling-optimizations")
        return

    # 运行测试
    try:
        if not args.skip_basic:
            test_basic_functionality(args.base_url)

        if not args.skip_concurrent:
            test_concurrent_requests(args.base_url, args.num_concurrent)

        if not args.skip_lengths:
            test_varying_lengths(args.base_url)

        # 尝试获取统计信息
        print("\n" + "=" * 60)
        print("📊 服务器统计信息")
        print("=" * 60)
        stats = get_server_stats(args.base_url)
        if stats:
            print(f"\n{stats}")
        else:
            print("\n  ℹ️  统计信息不可用")

        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n💡 提示:")
        print("  - 检查服务器日志查看优化统计信息")
        print("  - 使用 --enable-metrics 启用详细指标")
        print("  - 对比baseline版本查看性能提升")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
