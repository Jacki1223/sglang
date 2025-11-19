#!/bin/bash
# SGLang KV Cache预分配池 - 集成验证脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SGLang KV Cache预分配池 - 集成验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 验证计数
PASS=0
FAIL=0

# 检查函数
check_file() {
    local file=$1
    local desc=$2
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $desc"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} $desc (文件不存在: $file)"
        ((FAIL++))
        return 1
    fi
}

check_import() {
    local file=$1
    local import_line=$2
    local desc=$3
    if grep -q "$import_line" "$file"; then
        echo -e "${GREEN}✓${NC} $desc"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} $desc"
        ((FAIL++))
        return 1
    fi
}

check_code() {
    local file=$1
    local pattern=$2
    local desc=$3
    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓${NC} $desc"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} $desc"
        ((FAIL++))
        return 1
    fi
}

echo "1. 核心文件检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_file "python/sglang/srt/mem_cache/prealloc_pool_allocator.py" "PreallocPoolAllocator核心实现"
check_file "test/srt/test_prealloc_pool_allocator.py" "单元测试文件"
check_file "test_quick.py" "快速验证脚本"
echo ""

echo "2. 集成检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_import "python/sglang/srt/model_executor/model_runner.py" \
    "from sglang.srt.mem_cache.prealloc_pool_allocator import PreallocPoolAllocator" \
    "model_runner.py导入PreallocPoolAllocator"

check_code "python/sglang/srt/model_executor/model_runner.py" \
    "enable_prealloc = getattr(self.server_args, 'enable_kv_pool_prealloc'" \
    "model_runner.py中包含预分配池启用检查"

check_code "python/sglang/srt/model_executor/model_runner.py" \
    "PreallocPoolAllocator(" \
    "model_runner.py中实例化PreallocPoolAllocator"

check_code "python/sglang/srt/model_executor/model_runner.py" \
    "Using PreallocPoolAllocator for KV cache management" \
    "model_runner.py中包含启用日志"
echo ""

echo "3. 语法检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m py_compile python/sglang/srt/mem_cache/prealloc_pool_allocator.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} prealloc_pool_allocator.py语法正确"
    ((PASS++))
else
    echo -e "${RED}✗${NC} prealloc_pool_allocator.py语法错误"
    ((FAIL++))
fi

if python3 -m py_compile python/sglang/srt/model_executor/model_runner.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} model_runner.py语法正确"
    ((PASS++))
else
    echo -e "${RED}✗${NC} model_runner.py语法错误"
    ((FAIL++))
fi

if python3 -m py_compile test/srt/test_prealloc_pool_allocator.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} test_prealloc_pool_allocator.py语法正确"
    ((PASS++))
else
    echo -e "${RED}✗${NC} test_prealloc_pool_allocator.py语法错误"
    ((FAIL++))
fi
echo ""

echo "4. 文档检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_file "README_如何修复NoneType错误.md" "快速修复指南"
check_file "INTEGRATION_GUIDE_PreallocPool.md" "集成指南"
check_file "KV_Cache预分配池_README.md" "快速入门文档"
check_file "DEPLOYMENT_READY.md" "部署就绪文档"
check_file "SUMMARY_完整方案.md" "完整方案总结"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "通过: ${GREEN}$PASS${NC}"
echo -e "失败: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！代码已正确集成。${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  快速启动"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  # 启用KV Cache预分配池"
    echo "  export SGLANG_ENABLE_KV_POOL_PREALLOC=1"
    echo "  export SGLANG_KV_POOL_PREALLOC_RATIO=30"
    echo ""
    echo "  # 启动SGLang服务"
    echo "  python -m sglang.launch_server \\"
    echo "      --model-path meta-llama/Llama-3.1-8B-Instruct \\"
    echo "      --port 30000"
    echo ""
    echo "  # 应该看到日志："
    echo "  ${YELLOW}INFO Using PreallocPoolAllocator for KV cache management${NC}"
    echo "  ${YELLOW}INFO PreallocPool initialized: 5 pools, total_prealloc=614 pages${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ 有 $FAIL 项检查失败。请检查上述错误。${NC}"
    exit 1
fi
