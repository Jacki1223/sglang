"""
优化组件的单元测试
"""

import unittest
from unittest.mock import Mock, MagicMock

from sglang.srt.managers.optimizations.adaptive_token_ratio import (
    AdaptiveTokenRatioPredictor,
)
from sglang.srt.managers.optimizations.tiered_lpm import TieredLPMPolicy
from sglang.srt.managers.optimizations.adaptive_batch_sizer import AdaptiveBatchSizer
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

    def _create_mock_req(self, input_len, max_new_tokens):
        """创建模拟请求"""
        req = Mock(spec=Req)
        req.origin_input_ids = [1] * input_len
        req.output_ids = []
        req.sampling_params = Mock(spec=SamplingParams)
        req.sampling_params.max_new_tokens = max_new_tokens
        req.rid = f"req_{id(req)}"
        return req


class TestTieredLPMPolicy(unittest.TestCase):
    """测试分层LPM策略"""

    def setUp(self):
        self.tree_cache = self._create_mock_tree_cache()
        self.policy = TieredLPMPolicy(tier_size=32, max_tiers=4, tree_cache=self.tree_cache)

    def test_small_queue_uses_standard_lpm(self):
        """测试小队列使用标准LPM"""
        queue = [self._create_mock_req(i) for i in range(20)]
        prefix_computed = self.policy.calc_priority(queue)

        self.assertTrue(prefix_computed)
        # 应该按前缀长度降序排序
        prefix_lens = [len(req.prefix_indices) for req in queue]
        self.assertEqual(prefix_lens, sorted(prefix_lens, reverse=True))

        stats = self.policy.get_statistics()
        self.assertEqual(stats["tiered_sorts"], 0)

    def test_large_queue_uses_tiered_lpm(self):
        """测试大队列使用分层LPM"""
        queue = [self._create_mock_req(i) for i in range(150)]
        prefix_computed = self.policy.calc_priority(queue)

        self.assertTrue(prefix_computed)

        stats = self.policy.get_statistics()
        self.assertEqual(stats["tiered_sorts"], 1)

    def test_tiering_preserves_fcfs_across_tiers(self):
        """测试分层保持了层间FCFS"""
        queue = [self._create_mock_req(i) for i in range(100)]

        # 记录原始顺序（按层）
        tier_size = 25  # 100/4=25
        original_tier_0 = queue[:tier_size]

        self.policy.calc_priority(queue)

        # 检查第一层的请求仍然在前25个位置
        new_tier_0_rids = {req.rid for req in queue[:tier_size]}
        original_tier_0_rids = {req.rid for req in original_tier_0}
        self.assertEqual(new_tier_0_rids, original_tier_0_rids)

    def _create_mock_tree_cache(self):
        """创建模拟树缓存"""
        cache = MagicMock()
        cache.match_prefix = MagicMock(
            side_effect=lambda rid, key: (
                [1] * (hash(rid) % 50),  # 前缀长度（模拟）
                None,  # last_node
                None,  # last_host_node
                0,  # host_hit_length
            )
        )
        return cache

    def _create_mock_req(self, idx):
        """创建模拟请求"""
        req = Mock(spec=Req)
        req.rid = f"req_{idx}"
        req.origin_input_ids = [1] * 100
        req.output_ids = []
        req.extra_key = None
        req.prefix_indices = []
        req.last_node = None
        req.last_host_node = None
        req.host_hit_length = 0
        return req


class TestAdaptiveBatchSizer(unittest.TestCase):
    """测试自适应批大小调整器"""

    def setUp(self):
        self.sizer = AdaptiveBatchSizer(
            max_batch_size=128, min_batch_size=1, memory_threshold=0.85
        )

    def test_empty_queue(self):
        """测试空队列"""
        size = self.sizer.get_optimal_batch_size([], current_memory_usage=0.5)
        self.assertEqual(size, 0)

    def test_memory_constraint(self):
        """测试内存约束"""
        queue = [self._create_mock_req(100, 100) for _ in range(200)]

        # 低内存使用率
        size_low_mem = self.sizer.get_optimal_batch_size(queue, current_memory_usage=0.5)
        self.assertGreater(size_low_mem, 0)

        # 高内存使用率
        size_high_mem = self.sizer.get_optimal_batch_size(queue, current_memory_usage=0.95)

        # 高内存时批大小应该更小
        self.assertLess(size_high_mem, size_low_mem)

    def test_complexity_constraint(self):
        """测试复杂度约束"""
        # 简单请求（短输入输出）
        simple_queue = [self._create_mock_req(50, 50) for _ in range(100)]
        size_simple = self.sizer.get_optimal_batch_size(
            simple_queue, current_memory_usage=0.5
        )

        # 复杂请求（长输入输出）
        complex_queue = [self._create_mock_req(2000, 2000) for _ in range(100)]
        size_complex = self.sizer.get_optimal_batch_size(
            complex_queue, current_memory_usage=0.5
        )

        # 复杂请求批大小应该更小
        self.assertLess(size_complex, size_simple)

    def test_performance_based_adjustment(self):
        """测试基于性能的调整"""
        queue = [self._create_mock_req(100, 100) for _ in range(200)]

        # 模拟性能恶化（延迟增加，吞吐量不变）
        for i in range(20):
            self.sizer.update_metrics(
                batch_size=64, latency=0.05 + i * 0.01, throughput=20000
            )

        size_after_degradation = self.sizer.get_optimal_batch_size(
            queue, current_memory_usage=0.5
        )

        # 批大小应该减小
        self.assertLess(size_after_degradation, 64)

    def test_min_max_bounds(self):
        """测试最小最大边界"""
        queue = [self._create_mock_req(100, 100) for _ in range(200)]

        # 即使内存很高，也不应该超过max
        size = self.sizer.get_optimal_batch_size(queue, current_memory_usage=0.99)
        self.assertLessEqual(size, self.sizer.max_batch_size)
        self.assertGreaterEqual(size, self.sizer.min_batch_size)

    def test_statistics(self):
        """测试统计信息"""
        queue = [self._create_mock_req(100, 100) for _ in range(100)]

        # 多次调整
        for _ in range(50):
            self.sizer.get_optimal_batch_size(queue, current_memory_usage=0.5)

        # 更新指标
        for i in range(30):
            self.sizer.update_metrics(batch_size=32, latency=0.05, throughput=25000)

        stats = self.sizer.get_statistics()

        self.assertGreater(stats["total_adjustments"], 0)
        self.assertAlmostEqual(stats["avg_latency"], 0.05, places=4)
        self.assertAlmostEqual(stats["avg_throughput"], 25000, places=0)
        self.assertAlmostEqual(stats["avg_batch_size"], 32, places=0)

    def _create_mock_req(self, input_len, max_new_tokens):
        """创建模拟请求"""
        req = Mock(spec=Req)
        req.origin_input_ids = [1] * input_len
        req.sampling_params = Mock(spec=SamplingParams)
        req.sampling_params.max_new_tokens = max_new_tokens
        return req


if __name__ == "__main__":
    unittest.main()
