#!/usr/bin/env python3
"""
从数据集进行 MiDashengLM 离线推理

数据集格式: 音频url,中文描述
每行一个样本，逗号分隔
"""

import os
import csv
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from tqdm import tqdm


def load_dataset(dataset_path: str, limit: int = None) -> List[Tuple[str, str]]:
    """
    加载数据集
    
    Args:
        dataset_path: 数据集文件路径 (CSV 或 TXT)
        limit: 限制加载的样本数量，None表示全部加载
    
    Returns:
        List of (audio_url, description) tuples
    """
    data = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # 尝试作为CSV读取
        if dataset_path.endswith('.csv'):
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    audio_url = row[0].strip()
                    description = row[1].strip()
                    data.append((audio_url, description))
                    if limit and len(data) >= limit:
                        break
        else:
            # 作为文本文件，每行用逗号分隔
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释
                    continue
                parts = line.split(',', 1)  # 只分割第一个逗号
                if len(parts) >= 2:
                    audio_url = parts[0].strip()
                    description = parts[1].strip()
                    data.append((audio_url, description))
                    if limit and len(data) >= limit:
                        break
    
    print(f"✅ Loaded {len(data)} samples from {dataset_path}")
    return data


def download_audio(url: str, cache_dir: str = "~/.cache/audio_inference") -> str:
    """下载音频文件到本地缓存"""
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 从URL提取文件名，如果URL很长，使用hash
    if url.startswith('http://') or url.startswith('https://'):
        file_name = url.split("/")[-1].split("?")[0]  # 去除query参数
        if len(file_name) > 100 or not file_name:
            import hashlib
            file_name = hashlib.md5(url.encode()).hexdigest() + ".mp3"
    else:
        # 本地文件，直接返回路径
        return url
    
    file_path = os.path.join(cache_dir, file_name)
    
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return None
    
    return file_path


def create_messages_for_inference(audio_path: str, prompt: str = "请转录这段音频的内容。") -> List[Dict[str, Any]]:
    """
    创建推理用的消息格式
    
    Args:
        audio_path: 音频文件路径
        prompt: 提示词
    
    Returns:
        OpenAI格式的messages
    """
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


def run_offline_inference_openai_api(
    dataset: List[Tuple[str, str]],
    base_url: str = "http://localhost:30000/v1",
    api_key: str = "sk-123456",
    prompt: str = "请转录这段音频的内容。",
    max_tokens: int = 512,
    temperature: float = 0.0,
    output_file: str = "inference_results.jsonl",
):
    """
    使用 OpenAI API 格式进行离线推理（需要启动服务器）
    
    这个方法适合已经启动了 SGLang 服务器的情况
    """
    import openai
    
    client = openai.Client(api_key=api_key, base_url=base_url)
    
    results = []
    failed_count = 0
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting OpenAI API Inference")
    print(f"{'='*80}")
    print(f"Server: {base_url}")
    print(f"Samples: {len(dataset)}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (audio_url, description) in enumerate(tqdm(dataset, desc="Processing"), 1):
            try:
                # 下载音频
                audio_path = download_audio(audio_url)
                if not audio_path:
                    failed_count += 1
                    continue
                
                # 创建消息
                messages = create_messages_for_inference(audio_path, prompt)
                
                # 调用API
                start_time = time.time()
                response = client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.time() - start_time
                
                # 提取结果
                transcription = response.choices[0].message.content
                
                result = {
                    "index": idx,
                    "audio_url": audio_url,
                    "ground_truth": description,
                    "transcription": transcription,
                    "elapsed_time": elapsed,
                    "success": True,
                }
                
                # 写入文件
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
                results.append(result)
                
            except Exception as e:
                failed_count += 1
                result = {
                    "index": idx,
                    "audio_url": audio_url,
                    "ground_truth": description,
                    "error": str(e),
                    "success": False,
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                print(f"\n❌ Failed on sample {idx}: {e}")
    
    # 打印统计
    print(f"\n{'='*80}")
    print(f"📊 Inference Complete")
    print(f"{'='*80}")
    print(f"Total: {len(dataset)}")
    print(f"Success: {len(results)}")
    print(f"Failed: {failed_count}")
    if results:
        avg_time = sum(r['elapsed_time'] for r in results) / len(results)
        print(f"Average time: {avg_time:.2f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return results


def run_offline_inference_sglang_engine(
    dataset: List[Tuple[str, str]],
    model_path: str = "mispeech/midashenglm-7b",
    prompt: str = "请转录这段音频的内容。",
    max_tokens: int = 512,
    temperature: float = 0.0,
    output_file: str = "inference_results.jsonl",
    tp_size: int = 1,
):
    """
    使用 SGLang Engine 进行真正的离线推理（不需要启动服务器）
    
    这是真正的离线推理方式
    """
    from sglang import Engine
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting SGLang Engine Offline Inference")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Prompt: {prompt}")
    print(f"TP Size: {tp_size}")
    print(f"{'='*80}\n")
    
    # 初始化引擎
    print("Loading model...")
    engine = Engine(
        model_path=model_path,
        trust_remote_code=True,
        tp_size=tp_size,
    )
    
    results = []
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (audio_url, description) in enumerate(tqdm(dataset, desc="Processing"), 1):
            try:
                # 下载音频
                audio_path = download_audio(audio_url)
                if not audio_path:
                    failed_count += 1
                    continue
                
                # 创建消息
                messages = create_messages_for_inference(audio_path, prompt)
                
                # 推理
                start_time = time.time()
                outputs = engine.generate(
                    messages=messages,
                    sampling_params={
                        "temperature": temperature,
                        "max_new_tokens": max_tokens,
                    }
                )
                elapsed = time.time() - start_time
                
                # 提取结果
                transcription = outputs[0]["text"]
                
                result = {
                    "index": idx,
                    "audio_url": audio_url,
                    "ground_truth": description,
                    "transcription": transcription,
                    "elapsed_time": elapsed,
                    "success": True,
                }
                
                # 写入文件
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
                results.append(result)
                
            except Exception as e:
                failed_count += 1
                result = {
                    "index": idx,
                    "audio_url": audio_url,
                    "ground_truth": description,
                    "error": str(e),
                    "success": False,
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                print(f"\n❌ Failed on sample {idx}: {e}")
    
    # 打印统计
    print(f"\n{'='*80}")
    print(f"📊 Inference Complete")
    print(f"{'='*80}")
    print(f"Total: {len(dataset)}")
    print(f"Success: {len(results)}")
    print(f"Failed: {failed_count}")
    if results:
        avg_time = sum(r['elapsed_time'] for r in results) / len(results)
        print(f"Average time: {avg_time:.2f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MiDashengLM 离线推理")
    parser.add_argument("--dataset", type=str, required=True, help="数据集文件路径")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的样本数量")
    parser.add_argument("--prompt", type=str, default="请转录这段音频的内容。", help="推理提示词")
    parser.add_argument("--output", type=str, default="inference_results.jsonl", help="输出文件路径")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    
    # 选择推理模式
    parser.add_argument("--mode", type=str, choices=["api", "engine"], default="api",
                       help="推理模式: api=使用服务器API, engine=真正的离线推理")
    
    # API模式参数
    parser.add_argument("--base-url", type=str, default="http://localhost:30000/v1",
                       help="服务器URL (仅API模式)")
    parser.add_argument("--api-key", type=str, default="sk-123456",
                       help="API密钥 (仅API模式)")
    
    # Engine模式参数
    parser.add_argument("--model", type=str, default="mispeech/midashenglm-7b",
                       help="模型路径 (仅Engine模式)")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="张量并行大小 (仅Engine模式)")
    
    args = parser.parse_args()
    
    # 加载数据集
    dataset = load_dataset(args.dataset, limit=args.limit)
    
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查文件格式")
        return
    
    # 显示前3个样本
    print(f"\n📋 Sample data (first 3):")
    for i, (url, desc) in enumerate(dataset[:3], 1):
        print(f"  {i}. URL: {url[:60]}...")
        print(f"     描述: {desc[:60]}...")
    print()
    
    # 运行推理
    if args.mode == "api":
        print("🔵 Using OpenAI API mode (需要启动服务器)")
        results = run_offline_inference_openai_api(
            dataset=dataset,
            base_url=args.base_url,
            api_key=args.api_key,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_file=args.output,
        )
    else:  # engine
        print("🟢 Using SGLang Engine mode (真正的离线推理)")
        results = run_offline_inference_sglang_engine(
            dataset=dataset,
            model_path=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_file=args.output,
            tp_size=args.tp_size,
        )
    
    # 显示部分结果
    if results:
        print(f"\n📊 Sample results (first 3):")
        for result in results[:3]:
            print(f"\n  样本 {result['index']}:")
            print(f"    Ground truth: {result['ground_truth'][:80]}...")
            print(f"    Transcription: {result['transcription'][:80]}...")
            print(f"    Time: {result['elapsed_time']:.2f}s")


if __name__ == "__main__":
    main()
