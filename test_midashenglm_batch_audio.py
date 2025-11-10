#!/usr/bin/env python3
"""
批量测试 MiDashengLM 音频模型的脚本

使用方法：
1. 启动 SGLang 服务器：
   python -m sglang.launch_server \
       --model mispeech/midashenglm-7b \
       --trust-remote-code \
       --enable-multimodal \
       --port 30000

2. 运行测试：
   python test_midashenglm_batch_audio.py
"""

import os
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed


# 测试音频文件 URLs
TEST_AUDIO_URLS = {
    "trump_speech": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3",
    "bird_song": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3",
}

# 测试提示词
TEST_PROMPTS = {
    "transcribe": "Please listen to this audio carefully and transcribe the content in English.",
    "describe": "Please listen to this audio and describe what you hear.",
    "summarize": "Please listen to this audio and provide a brief summary.",
}


def download_audio(url: str, cache_dir: str = "~/.cache/audio_test") -> str:
    """下载音频文件到本地缓存"""
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    file_name = url.split("/")[-1]
    file_path = os.path.join(cache_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"📥 Downloading: {file_name}")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {file_path}")
    else:
        print(f"📦 Using cached: {file_path}")
    
    return file_path


def create_audio_messages(audio_path: str, prompt: str) -> List[Dict[str, Any]]:
    """创建音频消息格式"""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_path},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def test_single_audio(
    client: openai.Client,
    audio_path: str,
    prompt: str,
    audio_name: str,
    prompt_name: str,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """测试单个音频"""
    print(f"\n{'='*80}")
    print(f"🎵 Testing: {audio_name} with prompt: {prompt_name}")
    print(f"{'='*80}")
    
    messages = create_audio_messages(audio_path, prompt)
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
        )
        
        elapsed_time = time.time() - start_time
        content = response.choices[0].message.content
        
        result = {
            "audio_name": audio_name,
            "prompt_name": prompt_name,
            "success": True,
            "response": content,
            "elapsed_time": elapsed_time,
            "tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
        }
        
        print(f"✅ Success ({elapsed_time:.2f}s)")
        print(f"Response: {content[:200]}...")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        result = {
            "audio_name": audio_name,
            "prompt_name": prompt_name,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
        }
        print(f"❌ Failed: {e}")
    
    return result


def test_batch_sequential(
    base_url: str,
    api_key: str,
    audio_urls: Dict[str, str],
    prompts: Dict[str, str],
) -> List[Dict[str, Any]]:
    """顺序批量测试"""
    print("\n" + "="*80)
    print("📊 Sequential Batch Testing")
    print("="*80)
    
    client = openai.Client(api_key=api_key, base_url=base_url)
    results = []
    
    # 下载所有音频
    audio_paths = {}
    for name, url in audio_urls.items():
        audio_paths[name] = download_audio(url)
    
    # 顺序测试
    total_start = time.time()
    for audio_name, audio_path in audio_paths.items():
        for prompt_name, prompt in prompts.items():
            result = test_single_audio(
                client, audio_path, prompt, audio_name, prompt_name
            )
            results.append(result)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"📈 Sequential Testing Complete")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tests: {len(results)}")
    print(f"Success: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"{'='*80}\n")
    
    return results


def test_batch_parallel(
    base_url: str,
    api_key: str,
    audio_urls: Dict[str, str],
    prompts: Dict[str, str],
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """并行批量测试"""
    print("\n" + "="*80)
    print(f"🚀 Parallel Batch Testing (workers={max_workers})")
    print("="*80)
    
    client = openai.Client(api_key=api_key, base_url=base_url)
    
    # 下载所有音频
    audio_paths = {}
    for name, url in audio_urls.items():
        audio_paths[name] = download_audio(url)
    
    # 准备所有测试任务
    tasks = []
    for audio_name, audio_path in audio_paths.items():
        for prompt_name, prompt in prompts.items():
            tasks.append((audio_name, audio_path, prompt_name, prompt))
    
    # 并行执行
    results = []
    total_start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                test_single_audio,
                client,
                audio_path,
                prompt,
                audio_name,
                prompt_name,
            ): (audio_name, prompt_name)
            for audio_name, audio_path, prompt_name, prompt in tasks
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"📈 Parallel Testing Complete")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tests: {len(results)}")
    print(f"Success: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Average time per test: {total_time/len(results):.2f}s")
    print(f"{'='*80}\n")
    
    return results


def print_results_summary(results: List[Dict[str, Any]]):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("📋 Results Summary")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"\n{i}. {status} {result['audio_name']} + {result['prompt_name']}")
        print(f"   Time: {result['elapsed_time']:.2f}s")
        
        if result["success"]:
            response = result["response"]
            print(f"   Response: {response[:150]}...")
        else:
            print(f"   Error: {result['error']}")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    # 配置
    BASE_URL = "http://localhost:30000/v1"
    API_KEY = "sk-123456"
    
    print("\n" + "="*80)
    print("🎤 MiDashengLM Batch Audio Testing")
    print("="*80)
    print(f"Server: {BASE_URL}")
    print(f"Audio files: {len(TEST_AUDIO_URLS)}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Total tests: {len(TEST_AUDIO_URLS) * len(TEST_PROMPTS)}")
    print("="*80)
    
    # 测试 1: 顺序测试
    results_sequential = test_batch_sequential(
        BASE_URL, API_KEY, TEST_AUDIO_URLS, TEST_PROMPTS
    )
    print_results_summary(results_sequential)
    
    # 测试 2: 并行测试（可选，取决于服务器是否支持并发）
    print("\n" + "="*80)
    print("⚠️  Note: Parallel testing requires server to support concurrent requests")
    print("="*80)
    
    # 取消注释以启用并行测试
    # results_parallel = test_batch_parallel(
    #     BASE_URL, API_KEY, TEST_AUDIO_URLS, TEST_PROMPTS, max_workers=2
    # )
    # print_results_summary(results_parallel)


if __name__ == "__main__":
    main()
