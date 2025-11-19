"""
分层LPM策略

解决LPM在大队列时降级为FCFS的问题，通过分层处理保持缓存感知能力。
"""

import logging
from typing import List, Set

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

logger = logging.getLogger(__name__)


class TieredLPMPolicy:
    """分层LPM调度策略

    该策略将大队列分为多个层(tier)，每层内部使用LPM排序，
    避免了标准LPM在大队列时的性能问题。

    核心思想：
    1. 将等待队列按到达时间分为多层
    2. 每层内部使用LPM排序（计算复杂度可控）
    3. 跨层按FCFS原则（保证公平性）
    4. 动态调整层数和大小

    Args:
        tier_size: 每层的最大请求数
        max_tiers: 最大层数
        tree_cache: 前缀树缓存
    """

    def __init__(
        self,
        tier_size: int = 128,
        max_tiers: int = 4,
        tree_cache: BasePrefixCache = None,
    ):
        self.tier_size = tier_size
        self.max_tiers = max_tiers
        self.tree_cache = tree_cache

        # 统计信息
        self.total_sorts = 0
        self.tiered_sorts = 0

    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        """计算优先级并排序队列

        Args:
            waiting_queue: 等待队列（会被原地排序）

        Returns:
            是否计算了前缀匹配
        """
        queue_size = len(waiting_queue)
        self.total_sorts += 1

        if queue_size == 0:
            return False

        if queue_size <= self.tier_size:
            # 小队列：使用标准LPM
            return self._standard_lpm(waiting_queue)
        else:
            # 大队列：使用分层LPM
            self.tiered_sorts += 1
            return self._tiered_lpm(waiting_queue)

    def _standard_lpm(self, waiting_queue: List[Req]) -> bool:
        """标准LPM策略（适用于小队列）

        Args:
            waiting_queue: 等待队列

        Returns:
            True表示计算了前缀匹配
        """
        # 计算前缀匹配
        self._compute_prefix_matches(waiting_queue)

        # 按前缀长度降序排序
        waiting_queue.sort(key=lambda r: -len(r.prefix_indices))

        return True

    def _tiered_lpm(self, waiting_queue: List[Req]) -> bool:
        """分层LPM策略（适用于大队列）

        Args:
            waiting_queue: 等待队列

        Returns:
            True表示计算了前缀匹配
        """
        queue_size = len(waiting_queue)

        # 1. 计算层数
        num_tiers = min(
            (queue_size + self.tier_size - 1) // self.tier_size, self.max_tiers
        )

        # 2. 计算每层实际大小
        tier_size_actual = queue_size // num_tiers
        remainder = queue_size % num_tiers

        # 3. 分层并排序
        tiers = []
        start_idx = 0

        for i in range(num_tiers):
            # 前几层可能会多一个元素（处理余数）
            current_tier_size = tier_size_actual + (1 if i < remainder else 0)
            end_idx = start_idx + current_tier_size

            tier_reqs = waiting_queue[start_idx:end_idx]

            # 每层内部使用LPM排序
            self._compute_prefix_matches(tier_reqs)
            tier_reqs.sort(key=lambda r: -len(r.prefix_indices))

            tiers.append(tier_reqs)
            start_idx = end_idx

        # 4. 重新组合队列（按层顺序）
        waiting_queue.clear()
        for tier in tiers:
            waiting_queue.extend(tier)

        logger.debug(
            f"Tiered LPM: {queue_size} requests divided into {num_tiers} tiers "
            f"of size ~{tier_size_actual}"
        )

        return True

    def _compute_prefix_matches(self, requests: List[Req]):
        """计算请求的前缀匹配

        Args:
            requests: 请求列表
        """
        if self.tree_cache is None:
            # 如果没有树缓存，设置空前缀
            for req in requests:
                req.prefix_indices = []
                req.last_node = None
                req.last_host_node = None
                req.host_hit_length = 0
            return

        for req in requests:
            prefix_ids = req.origin_input_ids + req.output_ids
            extra_key = getattr(req, "extra_key", None)

            # 匹配前缀
            try:
                (
                    req.prefix_indices,
                    req.last_node,
                    req.last_host_node,
                    req.host_hit_length,
                ) = self.tree_cache.match_prefix(
                    rid=req.rid,
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                )
            except Exception as e:
                logger.warning(f"Prefix match failed for request {req.rid}: {e}")
                req.prefix_indices = []
                req.last_node = None
                req.last_host_node = None
                req.host_hit_length = 0

    def get_statistics(self):
        """获取统计信息

        Returns:
            统计信息字典
        """
        tiered_ratio = (
            self.tiered_sorts / self.total_sorts if self.total_sorts > 0 else 0.0
        )

        return {
            "total_sorts": self.total_sorts,
            "tiered_sorts": self.tiered_sorts,
            "tiered_ratio": tiered_ratio,
            "tier_size": self.tier_size,
            "max_tiers": self.max_tiers,
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.total_sorts = 0
        self.tiered_sorts = 0
