"""
优化组件集成示例

展示如何将优化组件集成到现有的Scheduler中。
"""

import logging
from typing import List, Optional

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
from sglang.srt.managers.optimizations.adaptive_batch_sizer import AdaptiveBatchSizer
from sglang.srt.managers.optimizations.adaptive_token_ratio import (
    AdaptiveTokenRatioPredictor,
)
from sglang.srt.managers.optimizations.tiered_lpm import TieredLPMPolicy
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class OptimizedSchedulerMixin:
    """优化调度器Mixin类

    该Mixin提供了集成各种优化组件的方法。
    可以被Scheduler类继承使用。
    """

    def init_optimizations(
        self,
        server_args: ServerArgs,
        tree_cache: BasePrefixCache,
    ):
        """初始化优化组件

        Args:
            server_args: 服务器参数
            tree_cache: 前缀树缓存
        """
        # 1. Token比例预测器
        self.token_ratio_predictor = AdaptiveTokenRatioPredictor(
            window_size=1000,
            user_window_size=100,
            bucket_window_size=200,
            default_ratio=0.5,
            percentile=75,
        )
        logger.info("Initialized AdaptiveTokenRatioPredictor")

        # 2. 分层LPM策略（如果使用LPM）
        if server_args.schedule_policy == "lpm":
            self.tiered_lpm_policy = TieredLPMPolicy(
                tier_size=128, max_tiers=4, tree_cache=tree_cache
            )
            logger.info("Initialized TieredLPMPolicy")
        else:
            self.tiered_lpm_policy = None

        # 3. 自适应批大小调整器
        self.adaptive_batch_sizer = AdaptiveBatchSizer(
            max_batch_size=server_args.max_running_requests or 256,
            min_batch_size=1,
            memory_threshold=0.85,
            history_window=100,
        )
        logger.info("Initialized AdaptiveBatchSizer")

        # 启用优化标志
        self.optimizations_enabled = True

    def get_new_batch_prefill_optimized(
        self,
        waiting_queue: List[Req],
        running_batch: ScheduleBatch,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> Optional[ScheduleBatch]:
        """优化版的get_new_batch_prefill

        集成了所有优化组件的批创建逻辑。

        Args:
            waiting_queue: 等待队列
            running_batch: 运行中的批
            tree_cache: 前缀树缓存
            server_args: 服务器参数

        Returns:
            新创建的预填批，如果没有则返回None
        """
        if not waiting_queue:
            return None

        # 1. 应用调度策略
        if self.tiered_lpm_policy is not None:
            # 使用分层LPM
            self.tiered_lpm_policy.calc_priority(waiting_queue)
        else:
            # 使用原有策略
            self.policy.calc_priority(waiting_queue)

        # 2. 确定最优批大小
        current_memory_usage = self._get_current_memory_usage()
        optimal_batch_size = self.adaptive_batch_sizer.get_optimal_batch_size(
            waiting_queue, current_memory_usage
        )

        logger.debug(
            f"Optimal batch size: {optimal_batch_size} "
            f"(queue: {len(waiting_queue)}, memory: {current_memory_usage:.2f})"
        )

        # 3. 初始化PrefillAdder（使用预测的token ratio）
        can_run_list = []
        new_chunked_req = None

        for i, req in enumerate(waiting_queue):
            if len(can_run_list) >= optimal_batch_size:
                break

            # 使用预测的token ratio
            predicted_ratio = self.token_ratio_predictor.predict_ratio(req)

            # 保存预测值用于后续评估
            req.predicted_token_ratio = predicted_ratio

            adder = PrefillAdder(
                page_size=server_args.page_size,
                tree_cache=tree_cache,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                running_batch=running_batch,
                new_token_ratio=predicted_ratio,  # 使用预测值
                rem_input_tokens=server_args.max_prefill_tokens,
                rem_chunk_tokens=server_args.chunked_prefill_size,
                priority_scheduling_preemption_threshold=server_args.priority_scheduling_preemption_threshold,
            )

            # 尝试添加请求
            result = adder.add_one_req(
                req,
                has_chunked_req=(new_chunked_req is not None),
                truncation_align_size=None,
            )

            if result == AddReqResult.CONTINUE:
                can_run_list.extend(adder.can_run_list)
                new_chunked_req = adder.new_chunked_req
            elif result == AddReqResult.NO_TOKEN:
                # 没有足够的token，停止添加
                break
            # AddReqResult.OTHER: 继续尝试下一个请求

        # 4. 创建批
        if can_run_list:
            batch = ScheduleBatch.init_new(
                reqs=can_run_list,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                tree_cache=tree_cache,
            )
            return batch

        return None

    def on_request_finish_optimized(self, req: Req):
        """请求完成时的优化处理

        Args:
            req: 完成的请求
        """
        # 更新token ratio预测器
        actual_output_len = len(req.output_ids)
        predicted_ratio = getattr(req, "predicted_token_ratio", None)

        self.token_ratio_predictor.update_on_finish(
            req, actual_output_len, predicted_ratio
        )

        # 记录统计信息
        if hasattr(self, "optimization_stats"):
            self.optimization_stats["requests_finished"] += 1

    def on_retract_optimized(self):
        """发生retract时的优化处理"""
        # 更新token ratio预测器
        self.token_ratio_predictor.update_on_retract()

        # 记录统计信息
        if hasattr(self, "optimization_stats"):
            self.optimization_stats["retract_count"] += 1

    def update_batch_metrics_optimized(
        self, batch_size: int, latency: float, throughput: float
    ):
        """更新批处理性能指标

        Args:
            batch_size: 批大小
            latency: 延迟（秒）
            throughput: 吞吐量（tokens/秒）
        """
        self.adaptive_batch_sizer.update_metrics(batch_size, latency, throughput)

    def get_optimization_statistics(self):
        """获取所有优化组件的统计信息

        Returns:
            包含所有统计信息的字典
        """
        stats = {}

        # Token ratio预测器统计
        stats["token_ratio_predictor"] = self.token_ratio_predictor.get_statistics()

        # 分层LPM统计
        if self.tiered_lpm_policy is not None:
            stats["tiered_lpm"] = self.tiered_lpm_policy.get_statistics()

        # 批大小调整器统计
        stats["adaptive_batch_sizer"] = self.adaptive_batch_sizer.get_statistics()

        return stats

    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用率

        Returns:
            内存使用率（0-1之间）
        """
        # 这需要根据实际的内存管理实现
        # 这里提供一个示例实现
        try:
            if hasattr(self, "token_to_kv_pool_allocator"):
                total_size = self.token_to_kv_pool_allocator.total_size
                available_size = self.token_to_kv_pool_allocator.available_size()
                used_size = total_size - available_size
                return used_size / total_size if total_size > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")

        return 0.5  # 默认返回50%


# 集成示例
def integrate_optimizations_into_scheduler(scheduler, server_args, tree_cache):
    """将优化集成到现有Scheduler实例中

    Args:
        scheduler: Scheduler实例
        server_args: ServerArgs实例
        tree_cache: 前缀树缓存实例
    """
    # 添加OptimizedSchedulerMixin的方法到scheduler
    for method_name in dir(OptimizedSchedulerMixin):
        if not method_name.startswith("_") and callable(
            getattr(OptimizedSchedulerMixin, method_name)
        ):
            method = getattr(OptimizedSchedulerMixin, method_name)
            setattr(scheduler, method_name, method.__get__(scheduler))

    # 初始化优化组件
    scheduler.init_optimizations(server_args, tree_cache)

    logger.info("Successfully integrated optimizations into scheduler")


# 使用示例
"""
# 在Scheduler.__init__中：
def __init__(self, server_args, ...):
    # ... 原有初始化代码 ...

    # 集成优化
    if server_args.enable_scheduling_optimizations:
        from sglang.srt.managers.optimizations.integration_example import (
            integrate_optimizations_into_scheduler
        )
        integrate_optimizations_into_scheduler(self, server_args, self.tree_cache)

# 在事件循环中使用优化版本的方法：
def event_loop_normal(self):
    while True:
        # ... 接收请求 ...

        # 使用优化版本
        if self.optimizations_enabled:
            batch = self.get_new_batch_prefill_optimized(
                self.waiting_queue,
                self.running_batch,
                self.tree_cache,
                self.server_args
            )
        else:
            batch = self.get_new_batch_prefill()

        # ... 运行批 ...

        # 更新指标
        if self.optimizations_enabled:
            self.update_batch_metrics_optimized(
                len(batch.reqs),
                batch_latency,
                batch_throughput
            )
"""
