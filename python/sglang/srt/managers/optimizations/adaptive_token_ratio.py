"""
自适应Token比例预测器

基于历史数据预测请求的实际token使用率，减少因预估不准导致的retract。
"""

from collections import defaultdict, deque
from typing import Dict, Optional

from sglang.srt.managers.schedule_batch import Req


class AdaptiveTokenRatioPredictor:
    """自适应token比例预测器

    该类通过跟踪历史请求的实际输出长度，预测新请求的token使用率。
    支持三级预测策略：
    1. 用户级预测（如果有用户标识）
    2. 输入长度bucket预测
    3. 全局历史预测

    Args:
        window_size: 全局历史窗口大小
        user_window_size: 每用户历史窗口大小
        bucket_window_size: 每bucket历史窗口大小
        default_ratio: 默认token使用率
        percentile: 使用的百分位数（75表示75%分位数，偏保守）
    """

    def __init__(
        self,
        window_size: int = 1000,
        user_window_size: int = 100,
        bucket_window_size: int = 200,
        default_ratio: float = 0.5,
        percentile: int = 75,
    ):
        self.window_size = window_size
        self.user_window_size = user_window_size
        self.bucket_window_size = bucket_window_size
        self.default_ratio = default_ratio
        self.percentile = percentile

        # 历史数据
        self.global_history = deque(maxlen=window_size)
        self.per_user_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=user_window_size)
        )
        self.per_length_bucket_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=bucket_window_size)
        )

        # 统计信息
        self.global_ratio = default_ratio
        self.retract_count = 0
        self.total_reqs = 0
        self.prediction_hits = 0  # 预测准确的次数（实际ratio在预测±10%范围内）

    def predict_ratio(self, req: Req) -> float:
        """预测请求的token使用比例

        Args:
            req: 请求对象

        Returns:
            预测的token使用率（0-1之间的浮点数）
        """
        # 1. 尝试用户级预测
        user_id = getattr(req, "user_id", None)
        if user_id and len(self.per_user_history[user_id]) >= 10:
            user_ratio = self._calculate_percentile(
                self.per_user_history[user_id], self.percentile
            )
            return min(max(user_ratio, 0.1), 1.0)

        # 2. 尝试输入长度bucket预测
        input_len_bucket = self._get_length_bucket(len(req.origin_input_ids))
        if len(self.per_length_bucket_history[input_len_bucket]) >= 20:
            bucket_ratio = self._calculate_percentile(
                self.per_length_bucket_history[input_len_bucket], self.percentile
            )
            return min(max(bucket_ratio, 0.1), 1.0)

        # 3. 使用全局历史预测
        if len(self.global_history) >= 50:
            global_ratio = self._calculate_percentile(
                self.global_history, self.percentile
            )
            return min(max(global_ratio, 0.1), 1.0)

        # 4. 默认值
        return self.default_ratio

    def update_on_finish(
        self, req: Req, actual_output_len: int, predicted_ratio: Optional[float] = None
    ):
        """请求完成时更新统计

        Args:
            req: 完成的请求
            actual_output_len: 实际输出长度
            predicted_ratio: 之前预测的ratio（用于评估预测准确度）
        """
        max_new_tokens = req.sampling_params.max_new_tokens
        if max_new_tokens == 0:
            return

        actual_ratio = min(actual_output_len / max_new_tokens, 1.0)

        # 评估预测准确度
        if predicted_ratio is not None:
            error = abs(actual_ratio - predicted_ratio) / max(predicted_ratio, 0.1)
            if error <= 0.1:  # 误差在10%以内认为是准确预测
                self.prediction_hits += 1

        # 更新全局历史
        self.global_history.append(actual_ratio)

        # 更新用户历史
        user_id = getattr(req, "user_id", None)
        if user_id:
            self.per_user_history[user_id].append(actual_ratio)

        # 更新长度bucket历史
        input_len_bucket = self._get_length_bucket(len(req.origin_input_ids))
        self.per_length_bucket_history[input_len_bucket].append(actual_ratio)

        # 更新全局比例（指数移动平均）
        alpha = 0.05
        self.global_ratio = alpha * actual_ratio + (1 - alpha) * self.global_ratio

        self.total_reqs += 1

    def update_on_retract(self):
        """发生retract时调整策略"""
        self.retract_count += 1

        # 如果retract率过高（>10%），降低预测比例使其更保守
        if self.total_reqs > 0 and self.retract_count / self.total_reqs > 0.1:
            self.global_ratio *= 0.9
            self.default_ratio *= 0.9

    def get_statistics(self) -> Dict:
        """获取统计信息

        Returns:
            包含各种统计指标的字典
        """
        prediction_accuracy = (
            self.prediction_hits / self.total_reqs if self.total_reqs > 0 else 0.0
        )
        retract_rate = (
            self.retract_count / self.total_reqs if self.total_reqs > 0 else 0.0
        )

        return {
            "total_requests": self.total_reqs,
            "retract_count": self.retract_count,
            "retract_rate": retract_rate,
            "prediction_accuracy": prediction_accuracy,
            "global_ratio": self.global_ratio,
            "num_users_tracked": len(self.per_user_history),
            "global_history_size": len(self.global_history),
        }

    @staticmethod
    def _get_length_bucket(length: int) -> str:
        """将输入长度映射到bucket

        Args:
            length: 输入序列长度

        Returns:
            bucket名称
        """
        if length < 100:
            return "short"
        elif length < 500:
            return "medium"
        elif length < 2000:
            return "long"
        else:
            return "very_long"

    @staticmethod
    def _calculate_percentile(data: deque, percentile: int) -> float:
        """计算百分位数

        Args:
            data: 数据序列
            percentile: 百分位数（0-100）

        Returns:
            对应百分位的值
        """
        if not data:
            return 0.5

        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    def reset(self):
        """重置所有统计数据"""
        self.global_history.clear()
        self.per_user_history.clear()
        self.per_length_bucket_history.clear()
        self.global_ratio = self.default_ratio
        self.retract_count = 0
        self.total_reqs = 0
        self.prediction_hits = 0
