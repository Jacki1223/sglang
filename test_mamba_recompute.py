#!/usr/bin/env python3
"""
Test script for Mamba State Recomputation功能

This script tests the mamba state recomputation feature by:
1. Sending requests with shared prefixes
2. Monitoring cache hit rates
3. Verifying recomputation statistics
"""

import argparse
import requests
import json
import time
from typing import List, Dict

def send_request(url: str, prompt: str, max_tokens: int = 50) -> Dict:
    """Send a generation request to SGLang server"""
    response = requests.post(
        f"{url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": 0.8,
            }
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def test_prefix_sharing(url: str):
    """Test cache hit with shared prefixes"""
    print("\n" + "="*60)
    print("Test 1: Shared Prefix Caching")
    print("="*60)

    shared_prefix = "请详细解释量子计算的"

    test_prompts = [
        shared_prefix + "基本原理和核心概念。",
        shared_prefix + "实际应用场景和案例。",
        shared_prefix + "发展历史和重要里程碑。",
        shared_prefix + "技术挑战和未来趋势。",
    ]

    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nRequest {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt[:50]}...")

        start_time = time.time()
        result = send_request(url, prompt)
        elapsed = time.time() - start_time

        results.append({
            'prompt_len': len(prompt),
            'elapsed': elapsed,
            'result': result,
        })

        print(f"Response time: {elapsed:.2f}s")
        if 'meta_info' in result and 'cached_tokens' in result['meta_info']:
            cached = result['meta_info']['cached_tokens']
            print(f"Cached tokens: {cached}")

    return results


def test_repeated_requests(url: str):
    """Test cache hit with repeated requests"""
    print("\n" + "="*60)
    print("Test 2: Repeated Request Caching")
    print("="*60)

    prompt = "解释一下人工智能和机器学习的区别。"

    print(f"\nSending the same prompt 3 times...")
    print(f"Prompt: {prompt}")

    results = []
    for i in range(3):
        print(f"\nIteration {i+1}/3")

        start_time = time.time()
        result = send_request(url, prompt, max_tokens=30)
        elapsed = time.time() - start_time

        results.append({
            'iteration': i + 1,
            'elapsed': elapsed,
            'result': result,
        })

        print(f"Response time: {elapsed:.2f}s")
        if 'meta_info' in result and 'cached_tokens' in result['meta_info']:
            cached = result['meta_info']['cached_tokens']
            print(f"Cached tokens: {cached}")
            if i > 0:
                print(f"  → Should see cache hits after first request!")

    return results


def test_incremental_prefix(url: str):
    """Test incremental prefix matching"""
    print("\n" + "="*60)
    print("Test 3: Incremental Prefix Matching")
    print("="*60)

    base = "量子"
    prompts = [
        base + "计算",
        base + "计算的",
        base + "计算的原理",
        base + "计算的原理是什么",
    ]

    print(f"\nSending prompts with incrementally longer prefixes...")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nRequest {i}/{len(prompts)}: '{prompt}'")

        start_time = time.time()
        result = send_request(url, prompt, max_tokens=20)
        elapsed = time.time() - start_time

        results.append({
            'prompt': prompt,
            'elapsed': elapsed,
            'result': result,
        })

        print(f"Response time: {elapsed:.2f}s")
        if 'meta_info' in result and 'cached_tokens' in result['meta_info']:
            cached = result['meta_info']['cached_tokens']
            print(f"Cached tokens: {cached}")

    return results


def get_server_stats(url: str) -> Dict:
    """Get server statistics"""
    try:
        response = requests.get(f"{url}/get_server_info", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Warning: Could not fetch server stats: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Test Mamba State Recomputation")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:30000",
        help="SGLang server URL"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["prefix", "repeated", "incremental", "all"],
        default="all",
        help="Which test to run"
    )
    args = parser.parse_args()

    print("="*60)
    print("Mamba State Recomputation Test Suite")
    print("="*60)
    print(f"Server URL: {args.url}")
    print(f"Test mode: {args.test}")

    # Get initial stats
    print("\nFetching initial server stats...")
    initial_stats = get_server_stats(args.url)

    # Run tests
    all_results = {}

    if args.test in ["prefix", "all"]:
        all_results['prefix'] = test_prefix_sharing(args.url)

    if args.test in ["repeated", "all"]:
        all_results['repeated'] = test_repeated_requests(args.url)

    if args.test in ["incremental", "all"]:
        all_results['incremental'] = test_incremental_prefix(args.url)

    # Get final stats
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    final_stats = get_server_stats(args.url)

    if final_stats:
        print(json.dumps(final_stats, indent=2))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, results in all_results.items():
        print(f"\n{test_name.upper()} Test:")
        print(f"  Total requests: {len(results)}")

        response_times = [r['elapsed'] for r in results]
        print(f"  Avg response time: {sum(response_times)/len(response_times):.2f}s")
        print(f"  Min response time: {min(response_times):.2f}s")
        print(f"  Max response time: {max(response_times):.2f}s")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    print("\nTo check recomputation statistics, look for log messages:")
    print("  - 'Mamba state recomputed successfully'")
    print("  - 'recompute_hit_count'")
    print("  - 'recompute_miss_count'")
    print("\nYou can also check the server logs for detailed cache behavior.")


if __name__ == "__main__":
    main()
