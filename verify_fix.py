#!/usr/bin/env python3
"""验证修复代码是否会在新Python进程中生效"""

import sys
import subprocess

# 在新的Python进程中测试导入
test_code = """
import sys
sys.path.insert(0, '/home/user/sglang/python')

from sglang.srt.models.midashenglm import MiDashengLMModel
import inspect

# 获取load_weights源代码
source = inspect.getsource(MiDashengLMModel.load_weights)

# 检查修复代码
if 'front_end.0.' in source and 'name.replace("front_end.0.", "front_end.")' in source:
    print("✅ 修复代码存在且可被导入")
    print("✅ 重启服务后将使用修复版本")

    # 检查修复逻辑
    if '.mel_scale.fb' in source and '.melscale_fbanks' in source:
        print("✅ Buffer名称映射逻辑完整")

    exit(0)
else:
    print("❌ 修复代码不存在")
    exit(1)
"""

result = subprocess.run(
    [sys.executable, "-c", test_code],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("错误:", result.stderr)

exit(result.returncode)
