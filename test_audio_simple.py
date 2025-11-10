#!/usr/bin/env python3
"""
简单的 MiDashengLM 音频测试脚本

最简单的使用示例
"""

import openai

def test_audio_simple():
    """最简单的音频测试示例"""
    
    # 1. 配置客户端
    client = openai.Client(
        api_key="sk-123456",
        base_url="http://localhost:30000/v1"
    )
    
    # 2. 准备音频文件路径（本地文件或URL）
    audio_path = "/path/to/your/audio.mp3"  # 修改为你的音频文件路径
    
    # 3. 创建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_path},
                },
                {
                    "type": "text",
                    "text": "Please listen to this audio and transcribe the content.",
                },
            ],
        }
    ]
    
    # 4. 调用API
    response = client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    
    # 5. 打印结果
    print("Response:")
    print(response.choices[0].message.content)


def test_multiple_audios():
    """测试多个音频文件"""
    
    client = openai.Client(
        api_key="sk-123456",
        base_url="http://localhost:30000/v1"
    )
    
    audio_files = [
        "/path/to/audio1.mp3",
        "/path/to/audio2.mp3",
        "/path/to/audio3.mp3",
    ]
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n{'='*50}")
        print(f"Testing audio {i}/{len(audio_files)}")
        print(f"{'='*50}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_path},
                    },
                    {
                        "type": "text",
                        "text": "What is this audio about?",
                    },
                ],
            }
        ]
        
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        
        print(f"Response: {response.choices[0].message.content}\n")


if __name__ == "__main__":
    print("🎤 MiDashengLM Simple Audio Test\n")
    
    # 运行简单测试
    # test_audio_simple()
    
    # 或运行批量测试
    # test_multiple_audios()
    
    print("\n⚠️  请先修改音频文件路径，然后取消注释相应的测试函数")
