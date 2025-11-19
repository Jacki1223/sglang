"""
自适应批大小调整器

根据请求特征、内存状况和历史性能动态调整批大小。
"""

import logging
from collections import deque
from typing import List

from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class AdaptiveBatchSizer:
    """自适应批大小调整器

    该类根据多个因素动态调整批大小：
    1. 当前内存使用率
    2. 请求的平均输入/输出长度
    3. 历史性能指标（延迟、吞吐量）

    Args:
        max_batch_size: 最大批大小
        min_batch_size: 最小批大小
        memory_threshold: 内存使用率阈值
        history_window: 性能历史窗口大小
    """

    def __init__(
        self,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        memory_threshold: float = 0.85,
        history_window: int = 100,
    ):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_threshold = memory_threshold
        self.history_window = history_window

        # 性能历史
        self.recent_latencies = deque(maxlen=history_window)
        self.recent_throughputs = deque(maxlen=history_window)
        self.recent_batch_sizes = deque(maxlen=history_window)

        # 当前目标批大小
        self.current_target = max_batch_size // 2

        # 统计信息
        self.total_adjustments = 0
        self.increase_count = 0
        self.decrease_count = 0

    def get_optimal_batch_size(
        self, waiting_queue: List[Req], current_memory_usage: float
    ) -> int:
        """计算最优批大小

        Args:
            waiting_queue: 等待队列
            current_memory_usage: 当前内存使用率（0-1）

        Returns:
            推荐的批大小
        """
        if len(waiting_queue) == 0:
            return 0

        # 1. 基于内存约束
        max_by_memory = self._get_max_by_memory(current_memory_usage)

        # 2. 基于请求特征
        max_by_complexity = self._get_max_by_complexity(waiting_queue)

        # 3. 基于历史性能
        max_by_performance = self._get_max_by_performance()

        # 4. 综合决策
        optimal_size = min(
            max_by_memory, max_by_complexity, max_by_performance, len(waiting_queue)
        )

        # 确保在合法范围内
        optimal_size = max(self.min_batch_size, min(optimal_size, self.max_batch_size))

        # 记录调整
        if optimal_size != self.current_target:
            self.total_adjustments += 1
            if optimal_size > self.current_target:
                self.increase_count += 1
            else:
                self.decrease_count += 1

            logger.debug(
                f"Batch size adjusted: {self.current_target} -> {optimal_size} "
                f"(memory={current_memory_usage:.2f}, queue={len(waiting_queue)})"
            )

        self.current_target = optimal_size
        return optimal_size

    def _get_max_by_memory(self, current_memory_usage: float) -> int:
        """基于内存使用率计算最大批大小

        Args:
            current_memory_usage: 当前内存使用率

        Returns:
            基于内存的最大批大小
        """
        if current_memory_usage > self.memory_threshold:
            # 内存紧张，激进减小批大小
            reduction_factor = (1.0 - current_memory_usage) / (
                1.0 - self.memory_threshold
            )
            reduction_factor = max(reduction_factor, 0.3)
            return int(self.current_target * reduction_factor)
        elif current_memory_usage < self.memory_threshold * 0.7:
            # 内存充足，可以适当增加
            return self.max_batch_size
        else:
            # 正常范围
            return self.current_target

    def _get_max_by_complexity(self, waiting_queue: List[Req]) -> int:
        """基于请求复杂度计算最大批大小

        Args:
            waiting_queue: 等待队列

        Returns:
            基于复杂度的最大批大小
        """
        if len(waiting_queue) == 0:
            return self.max_batch_size

        # 计算平均输入和输出长度
        total_input_len = sum(len(req.origin_input_ids) for req in waiting_queue)
        total_output_len = sum(
            req.sampling_params.max_new_tokens for req in waiting_queue
        )

        avg_input_len = total_input_len / len(waiting_queue)
        avg_output_len = total_output_len / len(waiting_queue)

        # 复杂度因子：长输入/输出 -> 小批大小
        # 假设基准是输入512，输出512
        complexity_factor = (avg_input_len + avg_output_len) / 1024

        if complexity_factor <= 0.5:
            # 简单请求，可以大批
            return self.max_batch_size
        elif complexity_factor <= 1.0:
            # 中等复杂度
            return int(self.max_batch_size * 0.8)
        elif complexity_factor <= 2.0:
            # 较复杂
            return int(self.max_batch_size * 0.5)
        else:
            # 非常复杂
            return int(self.max_batch_size * 0.3)

    def _get_max_by_performance(self) -> int:
        """基于历史性能计算最大批大小

        Returns:
            基于性能的最大批大小
        """
        if (
            len(self.recent_latencies) < 10
            or len(self.recent_throughputs) < 10
            or len(self.recent_batch_sizes) < 10
        ):
            # 历史数据不足
            return self.current_target

        # 计算性能趋势
        latency_trend = self._calculate_trend(self.recent_latencies)
        throughput_trend = self._calculate_trend(self.recent_throughputs)

        # 决策逻辑：
        # - 延迟上升 + 吞吐量没有显著提升 -> 减小批
        # - 延迟下降 + 吞吐量提升 -> 增大批
        # - 其他情况 -> 保持

        if latency_trend > 0.1 and throughput_trend < 0.05:
            # 延迟增加但吞吐量没有提升 -> 减小批
            new_target = int(self.current_target * 0.9)
            logger.debug(
                f"Performance degradation detected, reducing batch size: "
                f"{self.current_target} -> {new_target}"
            )
            return new_target
        elif latency_trend < -0.05 and throughput_trend > 0.1:
            # 延迟降低且吞吐量提升 -> 增大批
            new_target = int(self.current_target * 1.1)
            logger.debug(
                f"Performance improvement detected, increasing batch size: "
                f"{self.current_target} -> {new_target}"
            )
            return new_target
        else:
            # 保持当前批大小
            return self.current_target

    def update_metrics(self, batch_size: int, latency: float, throughput: float):
        """更新性能指标

        Args:
            batch_size: 批大小
            latency: 批处理延迟（秒）
            throughput: 吞吐量（tokens/秒）
        """
        self.recent_batch_sizes.append(batch_size)
        self.recent_latencies.append(latency)
        self.recent_throughputs.append(throughput)

    @staticmethod
    def _calculate_trend(data: deque) -> float:
        """计算数据趋势（简单线性回归斜率）

        Args:
            data: 数据序列

        Returns:
            趋势值（正数表示上升，负数表示下降）
        """
        if len(data) < 2:
            return 0.0

        data_list = list(data)
        n = len(data_list)

        # 计算均值
        x_mean = (n - 1) / 2
        y_mean = sum(data_list) / n

        # 计算斜率
        numerator = sum((i - x_mean) * (data_list[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # 归一化斜率（相对于均值）
        normalized_slope = slope / max(abs(y_mean), 1e-6)

        return normalized_slope

    def get_statistics(self):
        """获取统计信息

        Returns:
            统计信息字典
        """
        avg_latency = (
            sum(self.recent_latencies) / len(self.recent_latencies)
            if self.recent_latencies
            else 0.0
        )
        avg_throughput = (
            sum(self.recent_throughputs) / len(self.recent_throughputs)
            if self.recent_throughputs
            else 0.0
        )
        avg_batch_size = (
            sum(self.recent_batch_sizes) / len(self.recent_batch_sizes)
            if self.recent_batch_sizes
            else 0.0
        )

        return {
            "current_target": self.current_target,
            "total_adjustments": self.total_adjustments,
            "increase_count": self.increase_count,
            "decrease_count": self.decrease_count,
            "avg_latency": avg_latency,
            "avg_throughput": avg_throughput,
            "avg_batch_size": avg_batch_size,
        }

    def reset(self):
        """重置统计信息"""
        self.recent_latencies.clear()
        self.recent_throughputs.clear()
        self.recent_batch_sizes.clear()
        self.current_target = self.max_batch_size // 2
        self.total_adjustments = 0
        self.increase_count = 0
        self.decrease_count = 0
