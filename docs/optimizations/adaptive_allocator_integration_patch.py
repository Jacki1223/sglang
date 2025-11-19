"""
集成自适应页大小分配器的补丁文件

在 model_runner.py 的第 1903-1912 行之间添加自适应分配器支持
"""

# ============================================================================
# 文件: python/sglang/srt/model_executor/model_runner.py
# 位置: 第 1903 行附近
# ============================================================================

# 原有代码 (第 1903-1912 行):
"""
                else:
                    assert not self.is_hybrid
                    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
"""

# 修改后的代码:
"""
                else:
                    assert not self.is_hybrid

                    # 检查是否启用自适应页大小分配器
                    from sglang.srt.utils import get_bool_env_var
                    enable_adaptive_page = get_bool_env_var("SGLANG_ENABLE_ADAPTIVE_PAGE")

                    if enable_adaptive_page:
                        # 使用自适应分配器
                        from sglang.srt.mem_cache.allocator_adaptive import (
                            AdaptivePagedTokenToKVPoolAllocator
                        )
                        import os

                        # 读取页大小配置
                        page_sizes_str = os.environ.get(
                            "SGLANG_ADAPTIVE_PAGE_SIZES",
                            "16,64,256"
                        )
                        page_sizes = [int(x.strip()) for x in page_sizes_str.split(",")]

                        # 读取页大小比例配置（可选）
                        page_ratios = None
                        page_ratios_str = os.environ.get("SGLANG_ADAPTIVE_PAGE_RATIOS", None)
                        if page_ratios_str:
                            # 格式: "16:0.25,64:0.5,256:0.25"
                            try:
                                page_ratios = {}
                                for pair in page_ratios_str.split(","):
                                    size, ratio = pair.split(":")
                                    page_ratios[int(size)] = float(ratio)
                            except:
                                logger.warning(
                                    f"Invalid SGLANG_ADAPTIVE_PAGE_RATIOS format, "
                                    f"using default ratios"
                                )
                                page_ratios = None

                        self.token_to_kv_pool_allocator = AdaptivePagedTokenToKVPoolAllocator(
                            size=self.max_total_num_tokens,
                            page_sizes=page_sizes,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                            page_size_ratios=page_ratios,
                        )

                        logger.info(
                            f"Using AdaptivePagedTokenToKVPoolAllocator with "
                            f"page_sizes={page_sizes}, ratios={page_ratios}"
                        )
                    else:
                        # 使用原有的固定页大小分配器
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
"""

# ============================================================================
# 完整的修改示例
# ============================================================================

def example_modified_code():
    """这是修改后的完整代码段（仅供参考）"""

    # ... 前面的代码 ...

    if self.page_size == 1:
        if self.is_hybrid:
            self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                self.full_max_total_num_tokens,
                self.swa_max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=need_sort,
            )
        else:
            self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=need_sort,
            )
    else:
        assert not self.is_hybrid

        # ==================== 新增代码开始 ====================
        from sglang.srt.utils import get_bool_env_var
        enable_adaptive_page = get_bool_env_var("SGLANG_ENABLE_ADAPTIVE_PAGE")

        if enable_adaptive_page:
            from sglang.srt.mem_cache.allocator_adaptive import (
                AdaptivePagedTokenToKVPoolAllocator
            )
            import os

            page_sizes_str = os.environ.get("SGLANG_ADAPTIVE_PAGE_SIZES", "16,64,256")
            page_sizes = [int(x.strip()) for x in page_sizes_str.split(",")]

            page_ratios = None
            page_ratios_str = os.environ.get("SGLANG_ADAPTIVE_PAGE_RATIOS", None)
            if page_ratios_str:
                try:
                    page_ratios = {}
                    for pair in page_ratios_str.split(","):
                        size, ratio = pair.split(":")
                        page_ratios[int(size)] = float(ratio)
                except:
                    logger.warning("Invalid SGLANG_ADAPTIVE_PAGE_RATIOS, using defaults")
                    page_ratios = None

            self.token_to_kv_pool_allocator = AdaptivePagedTokenToKVPoolAllocator(
                size=self.max_total_num_tokens,
                page_sizes=page_sizes,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=need_sort,
                page_size_ratios=page_ratios,
            )

            logger.info(
                f"✓ Using Adaptive Page Allocator: sizes={page_sizes}, ratios={page_ratios}"
            )
        else:
            # ==================== 新增代码结束 ====================

            # 原有代码
            self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=need_sort,
            )
