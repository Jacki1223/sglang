#!/bin/bash
# 验证MiDashengLM的safetensors文件

echo "=================================="
echo "检查MiDashengLM checkpoint文件"
echo "=================================="

# 查找模型缓存目录
CACHE_DIR=$(find ~/.cache/huggingface -type d -path "*/models--mispeech--midashenglm*/snapshots/*" 2>/dev/null | head -1)

if [ -z "$CACHE_DIR" ]; then
    echo "⚠️  未找到缓存目录"
    echo "尝试其他位置..."
    CACHE_DIR=$(find /root/.cache/huggingface -type d -path "*/models--mispeech--midashenglm*/snapshots/*" 2>/dev/null | head -1)
fi

if [ -z "$CACHE_DIR" ]; then
    echo "❌ 找不到模型缓存目录"
    echo "请确认模型已下载"
    exit 1
fi

echo "📁 缓存目录: $CACHE_DIR"
echo ""

# 列出safetensors文件
echo "📦 Safetensors文件列表:"
echo "-----------------------------------"

SAFETENSORS_FILES=$(ls "$CACHE_DIR"/*.safetensors 2>/dev/null)
FILE_COUNT=$(echo "$SAFETENSORS_FILES" | grep -c "safetensors")

if [ $FILE_COUNT -eq 0 ]; then
    echo "❌ 未找到safetensors文件"
    exit 1
fi

echo "找到 $FILE_COUNT 个safetensors文件:"
echo ""

INDEX=1
TOTAL_SIZE=0

for file in $SAFETENSORS_FILES; do
    BASENAME=$(basename "$file")
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    SIZE_MB=$((SIZE / 1024 / 1024))
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE_MB))

    echo "  $INDEX. $BASENAME"
    echo "     大小: ${SIZE_MB} MB"

    INDEX=$((INDEX + 1))
done

echo ""
echo "总大小: $((TOTAL_SIZE / 1024)) GB"

# 检查index.json
echo ""
echo "📋 检查index文件:"
echo "-----------------------------------"

INDEX_FILE="$CACHE_DIR/model.safetensors.index.json"

if [ -f "$INDEX_FILE" ]; then
    echo "✅ 找到 model.safetensors.index.json"

    # 统计引用的文件数
    UNIQUE_FILES=$(grep -o '"model-[0-9]*-of-[0-9]*.safetensors"' "$INDEX_FILE" | sort -u | wc -l)
    echo "   索引中引用的文件数: $UNIQUE_FILES"

    # 显示文件列表
    echo "   引用的文件:"
    grep -o '"model-[0-9]*-of-[0-9]*.safetensors"' "$INDEX_FILE" | sort -u | sed 's/"//g' | nl
else
    echo "⚠️  未找到index.json"
fi

# 结论
echo ""
echo "=================================="
echo "🎯 结论"
echo "=================================="

if [ $FILE_COUNT -eq 7 ]; then
    echo "✅ 所有7个checkpoint文件都存在"
    echo "✅ 模型文件完整"
elif [ $FILE_COUNT -gt 7 ]; then
    echo "✅ 找到 $FILE_COUNT 个文件（可能包含其他格式）"
elif [ $FILE_COUNT -lt 7 ]; then
    echo "⚠️  只找到 $FILE_COUNT 个文件，预期是7个"
    echo "   模型可能下载不完整"
fi

echo ""
echo "当SGLang加载时，这些文件都会被读取"
echo "进度条显示 7/7 表示所有文件都被处理了"
