"""
AdaptiveTokenRatioPredictor的单元测试
"""

import unittest
from unittest.mock import Mock

from sglang.srt.managers.optimizations.adaptive_token_ratio import (
    AdaptiveTokenRatioPredictor,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.io_struct import SamplingParams


class TestAdaptiveTokenRatioPredictor(unittest.TestCase):
    """测试自适应Token比例预测器"""

    def setUp(self):
        self.predictor = AdaptiveTokenRatioPredictor(
            window_size=100, default_ratio=0.5, percentile=75
        )

    def test_initial_prediction(self):
        """测试初始预测（无历史数据）"""
        req = self._create_mock_req(input_len=100, max_new_tokens=50)
        ratio = self.predictor.predict_ratio(req)
        self.assertEqual(ratio, 0.5)  # 应该返回默认值

    def test_prediction_after_updates(self):
        """测试有历史数据后的预测"""
        # 添加一些历史数据
        for i in range(60):
            req = self._create_mock_req(input_len=100, max_new_tokens=100)
            self.predictor.update_on_finish(req, actual_output_len=70)

        # 现在预测应该基于历史数据
        req = self._create_mock_req(input_len=100, max_new_tokens=100)
        ratio = self.predictor.predict_ratio(req)

        # 70/100 = 0.7，75分位数应该接近或等于0.7
        self.assertGreaterEqual(ratio, 0.65)
        self.assertLessEqual(ratio, 0.75)

    def test_length_bucket_prediction(self):
        """测试基于长度bucket的预测"""
        # 为短输入添加历史数据（实际使用率高）
        for _ in range(25):
            req = self._create_mock_req(input_len=50, max_new_tokens=100)
            self.predictor.update_on_finish(req, actual_output_len=90)

        # 为长输入添加历史数据（实际使用率低）
        for _ in range(25):
            req = self._create_mock_req(input_len=1500, max_new_tokens=100)
            self.predictor.update_on_finish(req, actual_output_len=30)

        # 短输入应该预测高ratio
        short_req = self._create_mock_req(input_len=50, max_new_tokens=100)
        short_ratio = self.predictor.predict_ratio(short_req)
        self.assertGreater(short_ratio, 0.8)

        # 长输入应该预测低ratio
        long_req = self._create_mock_req(input_len=1500, max_new_tokens=100)
        long_ratio = self.predictor.predict_ratio(long_req)
        self.assertLess(long_ratio, 0.4)

    def test_retract_adjustment(self):
        """测试retract时的调整"""
        # 添加一些历史数据
        for _ in range(50):
            req = self._create_mock_req(input_len=100, max_new_tokens=100)
            self.predictor.update_on_finish(req, actual_output_len=80)

        initial_ratio = self.predictor.global_ratio

        # 模拟多次retract
        for _ in range(15):
            self.predictor.update_on_retract()

        # global_ratio应该降低
        self.assertLess(self.predictor.global_ratio, initial_ratio)

    def test_statistics(self):
        """测试统计信息"""
        # 添加数据
        for i in range(100):
            req = self._create_mock_req(input_len=100, max_new_tokens=100)
            self.predictor.update_on_finish(req, actual_output_len=75, predicted_ratio=0.7)

        # 模拟retract
        for _ in range(5):
            self.predictor.update_on_retract()

        stats = self.predictor.get_statistics()

        self.assertEqual(stats["total_requests"], 100)
        self.assertEqual(stats["retract_count"], 5)
        self.assertEqual(stats["retract_rate"], 0.05)
        self.assertGreater(stats["prediction_accuracy"], 0.8)  # 误差在10%内

    def test_user_level_prediction(self):
        """测试用户级预测"""
        # 为用户A添加历史数据（高使用率）
        for _ in range(30):
            req = self._create_mock_req(input_len=100, max_new_tokens=100, user_id="user_a")
            self.predictor.update_on_finish(req, actual_output_len=90)

        # 为用户B添加历史数据（低使用率）
        for _ in range(30):
            req = self._create_mock_req(input_len=100, max_new_tokens=100, user_id="user_b")
            self.predictor.update_on_finish(req, actual_output_len=30)

        # 用户A应该得到高预测
        req_a = self._create_mock_req(input_len=100, max_new_tokens=100, user_id="user_a")
        ratio_a = self.predictor.predict_ratio(req_a)
        self.assertGreater(ratio_a, 0.8)

        # 用户B应该得到低预测
        req_b = self._create_mock_req(input_len=100, max_new_tokens=100, user_id="user_b")
        ratio_b = self.predictor.predict_ratio(req_b)
        self.assertLess(ratio_b, 0.4)

    def test_window_size_limit(self):
        """测试窗口大小限制"""
        # 添加超过窗口大小的数据
        for i in range(150):
            req = self._create_mock_req(input_len=100, max_new_tokens=100)
            self.predictor.update_on_finish(req, actual_output_len=50)

        # 历史记录应该被限制在window_size
        self.assertLessEqual(len(self.predictor.global_history), self.predictor.window_size)

    def test_percentile_adjustment(self):
        """测试不同百分位数的影响"""
        # 创建不同percentile的预测器
        predictor_50 = AdaptiveTokenRatioPredictor(window_size=100, percentile=50)
        predictor_90 = AdaptiveTokenRatioPredictor(window_size=100, percentile=90)

        # 添加相同的历史数据
        for _ in range(60):
            req = self._create_mock_req(input_len=100, max_new_tokens=100)
            # 模拟变化的输出长度：30-90
            output_len = 30 + (_ % 60)
            predictor_50.update_on_finish(req, actual_output_len=output_len)
            predictor_90.update_on_finish(req, actual_output_len=output_len)

        req = self._create_mock_req(input_len=100, max_new_tokens=100)

        # 90分位数应该比50分位数更保守（更高）
        ratio_50 = predictor_50.predict_ratio(req)
        ratio_90 = predictor_90.predict_ratio(req)
        self.assertLess(ratio_50, ratio_90)

    def _create_mock_req(self, input_len, max_new_tokens, user_id=None):
        """创建模拟请求"""
        req = Mock(spec=Req)
        req.origin_input_ids = [1] * input_len
        req.output_ids = []
        req.sampling_params = Mock(spec=SamplingParams)
        req.sampling_params.max_new_tokens = max_new_tokens
        req.rid = f"req_{id(req)}"

        # 添加user_id（如果有）
        if user_id:
            # 某些请求可能有user_id字段
            req.user_id = user_id

        return req


if __name__ == "__main__":
    unittest.main()
