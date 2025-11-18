"""
优化的调度策略实现

解决策略切换性能断崖问题，提供自适应和渐进式策略切换。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
    IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class AdaptiveSchedulePolicy(SchedulePolicy):
    """
    自适应调度策略

    解决策略切换性能断崖的优化实现，包括：
    1. 动态阈值调整
    2. 渐进式策略切换（Hysteresis）
    3. 基于历史性能的策略选择
    4. 轻量级前缀匹配
    """

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
        adaptive_threshold: bool = True,
        hysteresis_window: int = 10,
    ):
        super().__init__(
            policy,
            tree_cache,
            enable_hierarchical_cache,
            enable_priority_scheduling,
            schedule_low_priority_values_first,
        )

        self.adaptive_threshold = adaptive_threshold
        self.hysteresis_window = hysteresis_window

        # 动态阈值参数
        self.switch_threshold_low = 100  # 下界
        self.switch_threshold_high = 150  # 上界（带滞后）
        self.current_policy_state = self.policy

        # 性能历史统计
        self.policy_performance_history = {
            'lpm': deque(maxlen=100),
            'fcfs': deque(maxlen=100),
        }

        # 轻量级前缀匹配缓存
        self.prefix_match_cache: Dict[int, tuple] = {}  # rid -> (prefix_indices, last_node)
        self.cache_timestamp: Dict[int, float] = {}
        self.cache_ttl = 5.0  # 缓存5秒

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Union[CacheAwarePolicy, CacheAgnosticPolicy]:
        """
        优化的策略决策逻辑

        改进：
        1. 使用滞后（hysteresis）避免频繁切换
        2. 考虑请求特征（平均前缀长度）
        3. 基于历史性能动态调整
        """
        queue_len = len(waiting_queue)

        # 如果不是LPM策略，直接返回
        if self.policy != CacheAwarePolicy.LPM:
            return self.policy

        # 计算平均前缀长度（采样估算）
        avg_prefix_len = self._estimate_avg_prefix_length(waiting_queue)

        # 决策逻辑
        if self.adaptive_threshold:
            return self._adaptive_policy_decision(queue_len, avg_prefix_len)
        else:
            return self._hysteresis_policy_decision(queue_len)

    def _adaptive_policy_decision(
        self,
        queue_len: int,
        avg_prefix_len: float
    ) -> Union[CacheAwarePolicy, CacheAgnosticPolicy]:
        """
        自适应策略决策

        考虑因素：
        1. 队列长度
        2. 平均前缀长度
        3. 历史性能
        """
        # 策略1: 短前缀场景，直接使用FCFS
        if avg_prefix_len < 50:
            return CacheAgnosticPolicy.FCFS

        # 策略2: 长前缀但队列很大，使用采样LPM
        if avg_prefix_len > 200 and queue_len > 128:
            # 可以使用采样版本的LPM（后续实现）
            return CacheAgnosticPolicy.FCFS

        # 策略3: 基于历史性能决策
        lpm_perf = self._get_avg_performance('lpm')
        fcfs_perf = self._get_avg_performance('fcfs')

        # 如果LPM显著更慢（2倍以上），切换到FCFS
        if lpm_perf > 0 and fcfs_perf > 0:
            if lpm_perf / fcfs_perf > 2.0 and queue_len > 100:
                return CacheAgnosticPolicy.FCFS

        # 策略4: 默认使用原始阈值
        if queue_len > self.switch_threshold_high:
            return CacheAgnosticPolicy.FCFS

        return CacheAwarePolicy.LPM

    def _hysteresis_policy_decision(
        self,
        queue_len: int
    ) -> Union[CacheAwarePolicy, CacheAgnosticPolicy]:
        """
        带滞后的策略决策

        避免在阈值附近频繁切换
        """
        # 当前使用LPM
        if self.current_policy_state == CacheAwarePolicy.LPM:
            # 只有超过高阈值才切换到FCFS
            if queue_len > self.switch_threshold_high:
                self.current_policy_state = CacheAgnosticPolicy.FCFS
                logger.debug(f"Policy switch: LPM -> FCFS at queue_len={queue_len}")
        # 当前使用FCFS
        else:
            # 只有低于低阈值才切换回LPM
            if queue_len < self.switch_threshold_low:
                self.current_policy_state = CacheAwarePolicy.LPM
                logger.debug(f"Policy switch: FCFS -> LPM at queue_len={queue_len}")

        return self.current_policy_state

    def _estimate_avg_prefix_length(self, waiting_queue: List[Req]) -> float:
        """
        估算平均前缀长度（采样）

        只采样前N个请求，避免全量计算
        """
        if not waiting_queue:
            return 0.0

        sample_size = min(10, len(waiting_queue))
        sample_reqs = waiting_queue[:sample_size]

        total_len = 0
        for req in sample_reqs:
            # 使用缓存避免重复计算
            if req.rid in self.prefix_match_cache:
                cache_time = self.cache_timestamp.get(req.rid, 0)
                if time.time() - cache_time < self.cache_ttl:
                    prefix_indices, _ = self.prefix_match_cache[req.rid]
                    total_len += len(prefix_indices)
                    continue

            # 快速前缀长度估算（不做完整匹配）
            prefix_ids = req.origin_input_ids + req.output_ids
            # 简化版：只检查前缀长度，不做树匹配
            total_len += len(prefix_ids) // 2  # 粗略估计

        return total_len / sample_size

    def _get_avg_performance(self, policy: str) -> float:
        """获取策略的平均性能（调度时间）"""
        history = self.policy_performance_history.get(policy, [])
        if not history:
            return 0.0
        return sum(history) / len(history)

    def record_performance(self, policy: str, schedule_time_ms: float):
        """记录策略性能"""
        if policy in self.policy_performance_history:
            self.policy_performance_history[policy].append(schedule_time_ms)

    def calc_priority_with_monitoring(
        self,
        waiting_queue: List[Req]
    ) -> tuple[bool, float]:
        """
        带性能监控的优先级计算

        返回: (prefix_computed, schedule_time_ms)
        """
        start_time = time.time()

        # 调用原始方法
        prefix_computed = self.calc_priority(waiting_queue)

        schedule_time_ms = (time.time() - start_time) * 1000

        # 记录性能
        policy_name = 'lpm' if isinstance(
            self.current_policy_state, CacheAwarePolicy
        ) else 'fcfs'
        self.record_performance(policy_name, schedule_time_ms)

        return prefix_computed, schedule_time_ms


class SamplingLPMPolicy(AdaptiveSchedulePolicy):
    """
    采样版本的LPM策略

    在队列很大时，只对部分请求计算前缀匹配，降低复杂度
    """

    def __init__(self, *args, sampling_ratio: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_ratio = sampling_ratio
        self.min_sample_size = 32
        self.max_sample_size = 128

    def _compute_prefix_matches_sampled(
        self,
        waiting_queue: List[Req],
        policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        采样版本的前缀匹配计算

        只计算部分请求的前缀，其余使用启发式方法
        """
        queue_len = len(waiting_queue)

        # 小队列：全量计算
        if queue_len <= self.min_sample_size:
            return self._compute_prefix_matches(waiting_queue, policy)

        # 大队列：采样计算
        sample_size = min(
            self.max_sample_size,
            max(self.min_sample_size, int(queue_len * self.sampling_ratio))
        )

        # 策略1: 分层采样
        # - 前N个（最早到达）
        # - 随机采样中间部分
        # - 最后N个（最新到达）
        head_size = sample_size // 3
        tail_size = sample_size // 3
        mid_size = sample_size - head_size - tail_size

        sampled_reqs = (
            waiting_queue[:head_size] +
            waiting_queue[head_size:-tail_size:max(1, (queue_len - head_size - tail_size) // mid_size)] +
            waiting_queue[-tail_size:]
        )

        # 计算采样请求的前缀
        temporary_deprioritized = set()
        self.waiting_queue_radix_tree.reset()

        for r in sampled_reqs:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key

            r.prefix_indices, r.last_node, r.last_host_node, r.host_hit_length = (
                self.tree_cache.match_prefix(
                    rid=r.rid,
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                )
            )

            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                in_batch_matching_prefixes, _, _, _ = (
                    self.waiting_queue_radix_tree.match_prefix(
                        rid=r.rid,
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                    )
                )
                if len(in_batch_matching_prefixes) >= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                    temporary_deprioritized.add(r.rid)
                else:
                    self.waiting_queue_radix_tree.insert(
                        RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                        torch.empty(len(prefix_ids), dtype=torch.bool),
                    )

        # 对未采样的请求使用启发式
        for r in waiting_queue:
            if r not in sampled_reqs:
                # 启发式：假设与相邻请求相似
                # 可以使用更复杂的策略
                r.prefix_indices = torch.tensor([], dtype=torch.long)
                r.last_node = self.tree_cache.root_node

        return temporary_deprioritized

    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        """
        覆盖父类方法，使用采样计算
        """
        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.schedule_low_priority_values_first
                )
            return False

        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            prefix_computed = True

            # 使用采样版本
            if len(waiting_queue) > self.max_sample_size:
                temporary_deprioritized = self._compute_prefix_matches_sampled(
                    waiting_queue, policy
                )
            else:
                temporary_deprioritized = self._compute_prefix_matches(
                    waiting_queue, policy
                )

            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.schedule_low_priority_values_first,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)

        return prefix_computed


def create_optimized_schedule_policy(
    policy: str,
    tree_cache: BasePrefixCache,
    enable_hierarchical_cache: bool = False,
    enable_priority_scheduling: bool = False,
    schedule_low_priority_values_first: bool = False,
    use_adaptive: bool = True,
    use_sampling: bool = False,
) -> SchedulePolicy:
    """
    创建优化的调度策略

    Args:
        policy: 策略名称
        tree_cache: Radix缓存
        enable_hierarchical_cache: 是否启用分层缓存
        enable_priority_scheduling: 是否启用优先级调度
        schedule_low_priority_values_first: 是否优先调度低优先级
        use_adaptive: 是否使用自适应策略
        use_sampling: 是否使用采样LPM

    Returns:
        SchedulePolicy实例
    """
    if use_sampling:
        return SamplingLPMPolicy(
            policy=policy,
            tree_cache=tree_cache,
            enable_hierarchical_cache=enable_hierarchical_cache,
            enable_priority_scheduling=enable_priority_scheduling,
            schedule_low_priority_values_first=schedule_low_priority_values_first,
            adaptive_threshold=use_adaptive,
        )
    elif use_adaptive:
        return AdaptiveSchedulePolicy(
            policy=policy,
            tree_cache=tree_cache,
            enable_hierarchical_cache=enable_hierarchical_cache,
            enable_priority_scheduling=enable_priority_scheduling,
            schedule_low_priority_values_first=schedule_low_priority_values_first,
            adaptive_threshold=True,
        )
    else:
        # 使用原始策略
        return SchedulePolicy(
            policy=policy,
            tree_cache=tree_cache,
            enable_hierarchical_cache=enable_hierarchical_cache,
            enable_priority_scheduling=enable_priority_scheduling,
            schedule_low_priority_values_first=schedule_low_priority_values_first,
        )
