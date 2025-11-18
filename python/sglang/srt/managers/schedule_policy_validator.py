"""
调度策略切换性能验证工具

此模块用于验证和分析调度策略切换时的性能断崖问题。
"""

import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicySwitchMetrics:
    """策略切换性能指标"""

    timestamp: float
    queue_length: int
    active_policy: str
    prefix_compute_time_ms: float
    sort_time_ms: float
    total_schedule_time_ms: float
    cache_hit_rate: float
    avg_prefix_length: float
    policy_switched: bool = False

    def __repr__(self):
        return (
            f"PolicySwitchMetrics("
            f"queue_len={self.queue_length}, "
            f"policy={self.active_policy}, "
            f"total_time={self.total_schedule_time_ms:.2f}ms, "
            f"switched={self.policy_switched})"
        )


class PolicyPerformanceMonitor:
    """
    策略性能监控器

    用于实时监控调度策略性能，检测性能断崖。
    """

    def __init__(
        self,
        window_size: int = 1000,
        performance_cliff_threshold: float = 2.0,
    ):
        """
        Args:
            window_size: 滑动窗口大小
            performance_cliff_threshold: 性能断崖阈值（倍数）
        """
        self.window_size = window_size
        self.performance_cliff_threshold = performance_cliff_threshold

        # 存储历史指标
        self.metrics_history: deque = deque(maxlen=window_size)

        # 按队列长度分桶统计
        self.bucket_stats: Dict[int, List[PolicySwitchMetrics]] = defaultdict(list)

        # 策略切换事件
        self.switch_events: List[Tuple[float, int, str, str]] = []

        # 性能统计
        self.total_schedules = 0
        self.total_switches = 0
        self.cliff_detected = 0

    def record_schedule(self, metrics: PolicySwitchMetrics):
        """记录一次调度的指标"""
        self.metrics_history.append(metrics)
        self.total_schedules += 1

        # 按队列长度分桶（每10个一桶）
        bucket = (metrics.queue_length // 10) * 10
        self.bucket_stats[bucket].append(metrics)

        # 检测策略切换
        if len(self.metrics_history) >= 2:
            prev_metrics = self.metrics_history[-2]
            if prev_metrics.active_policy != metrics.active_policy:
                self.total_switches += 1
                metrics.policy_switched = True
                self.switch_events.append((
                    metrics.timestamp,
                    metrics.queue_length,
                    prev_metrics.active_policy,
                    metrics.active_policy
                ))

                # 检测性能断崖
                if self._is_performance_cliff(prev_metrics, metrics):
                    self.cliff_detected += 1
                    logger.warning(
                        f"Performance cliff detected! "
                        f"Queue: {prev_metrics.queue_length}->{metrics.queue_length}, "
                        f"Time: {prev_metrics.total_schedule_time_ms:.2f}ms->"
                        f"{metrics.total_schedule_time_ms:.2f}ms, "
                        f"Policy: {prev_metrics.active_policy}->{metrics.active_policy}"
                    )

    def _is_performance_cliff(
        self,
        prev: PolicySwitchMetrics,
        curr: PolicySwitchMetrics
    ) -> bool:
        """检测是否出现性能断崖"""
        if prev.total_schedule_time_ms < 1.0:
            return False  # 时间太短，不可靠

        ratio = curr.total_schedule_time_ms / prev.total_schedule_time_ms

        # 如果性能突然下降超过阈值倍数
        if ratio > self.performance_cliff_threshold:
            return True

        # 或者性能突然上升超过阈值倍数（切换到更快的策略）
        if ratio < 1.0 / self.performance_cliff_threshold:
            return True

        return False

    def analyze_by_queue_length(self) -> Dict[int, Dict[str, float]]:
        """按队列长度分析性能"""
        analysis = {}

        for bucket, metrics_list in sorted(self.bucket_stats.items()):
            if not metrics_list:
                continue

            # 计算统计数据
            times = [m.total_schedule_time_ms for m in metrics_list]
            prefix_times = [m.prefix_compute_time_ms for m in metrics_list]
            policies = [m.active_policy for m in metrics_list]

            analysis[bucket] = {
                'count': len(metrics_list),
                'avg_time_ms': np.mean(times),
                'p50_time_ms': np.percentile(times, 50),
                'p95_time_ms': np.percentile(times, 95),
                'p99_time_ms': np.percentile(times, 99),
                'std_time_ms': np.std(times),
                'avg_prefix_time_ms': np.mean(prefix_times),
                'dominant_policy': max(set(policies), key=policies.count),
                'policy_distribution': {
                    policy: policies.count(policy) / len(policies)
                    for policy in set(policies)
                }
            }

        return analysis

    def detect_threshold_issues(self) -> List[Dict]:
        """检测阈值设置问题"""
        issues = []
        analysis = self.analyze_by_queue_length()

        prev_bucket = None
        for bucket in sorted(analysis.keys()):
            stats = analysis[bucket]

            if prev_bucket is not None:
                prev_stats = analysis[prev_bucket]

                # 检查性能突变
                time_ratio = stats['avg_time_ms'] / max(prev_stats['avg_time_ms'], 0.1)

                if time_ratio > self.performance_cliff_threshold:
                    issues.append({
                        'type': 'performance_cliff',
                        'severity': 'high',
                        'bucket_range': f"{prev_bucket}-{bucket}",
                        'time_increase': f"{time_ratio:.2f}x",
                        'prev_policy': prev_stats['dominant_policy'],
                        'curr_policy': stats['dominant_policy'],
                        'recommendation': (
                            f"Consider adjusting threshold or using gradual transition "
                            f"between {prev_bucket} and {bucket}"
                        )
                    })

                # 检查策略频繁切换
                if (stats['policy_distribution'] and
                    len(stats['policy_distribution']) > 1):
                    entropy = -sum(
                        p * np.log2(p)
                        for p in stats['policy_distribution'].values()
                    )
                    if entropy > 0.8:  # 高熵表示策略不稳定
                        issues.append({
                            'type': 'unstable_policy',
                            'severity': 'medium',
                            'bucket': bucket,
                            'policy_entropy': f"{entropy:.2f}",
                            'distribution': stats['policy_distribution'],
                            'recommendation': (
                                f"Policy switches frequently at queue length ~{bucket}. "
                                f"Consider adding hysteresis."
                            )
                        })

            prev_bucket = bucket

        return issues

    def generate_report(self) -> str:
        """生成性能分析报告"""
        analysis = self.analyze_by_queue_length()
        issues = self.detect_threshold_issues()

        report = ["=" * 80]
        report.append("调度策略性能分析报告")
        report.append("=" * 80)
        report.append("")

        # 总体统计
        report.append("📊 总体统计:")
        report.append(f"  总调度次数: {self.total_schedules}")
        report.append(f"  策略切换次数: {self.total_switches}")
        report.append(f"  性能断崖检测次数: {self.cliff_detected}")
        report.append("")

        # 按队列长度分析
        report.append("📈 按队列长度分析:")
        report.append(f"{'队列长度':<12} {'样本数':<8} {'平均时间':<12} "
                     f"{'P95时间':<12} {'主策略':<12} {'策略分布'}")
        report.append("-" * 80)

        for bucket in sorted(analysis.keys()):
            stats = analysis[bucket]
            policy_dist = ', '.join(
                f"{p}:{v:.0%}" for p, v in stats['policy_distribution'].items()
            )
            report.append(
                f"{bucket:<12} {stats['count']:<8} "
                f"{stats['avg_time_ms']:<12.2f} "
                f"{stats['p95_time_ms']:<12.2f} "
                f"{stats['dominant_policy']:<12} {policy_dist}"
            )
        report.append("")

        # 问题报告
        if issues:
            report.append("⚠️  检测到的问题:")
            for i, issue in enumerate(issues, 1):
                report.append(f"\n问题 {i}: {issue['type']} (严重性: {issue['severity']})")
                for key, value in issue.items():
                    if key not in ['type', 'severity']:
                        report.append(f"  {key}: {value}")
        else:
            report.append("✅ 未检测到明显的性能问题")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def export_metrics_csv(self, filepath: str):
        """导出指标到CSV"""
        import csv

        with open(filepath, 'w', newline='') as f:
            if not self.metrics_history:
                return

            fieldnames = [
                'timestamp', 'queue_length', 'active_policy',
                'prefix_compute_time_ms', 'sort_time_ms',
                'total_schedule_time_ms', 'cache_hit_rate',
                'avg_prefix_length', 'policy_switched'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for metrics in self.metrics_history:
                writer.writerow({
                    'timestamp': metrics.timestamp,
                    'queue_length': metrics.queue_length,
                    'active_policy': metrics.active_policy,
                    'prefix_compute_time_ms': metrics.prefix_compute_time_ms,
                    'sort_time_ms': metrics.sort_time_ms,
                    'total_schedule_time_ms': metrics.total_schedule_time_ms,
                    'cache_hit_rate': metrics.cache_hit_rate,
                    'avg_prefix_length': metrics.avg_prefix_length,
                    'policy_switched': metrics.policy_switched,
                })


# 全局监控器实例
_global_monitor: PolicyPerformanceMonitor = None


def get_global_monitor() -> PolicyPerformanceMonitor:
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PolicyPerformanceMonitor()
    return _global_monitor


def enable_policy_monitoring():
    """启用策略监控"""
    global _global_monitor
    _global_monitor = PolicyPerformanceMonitor()
    logger.info("Policy performance monitoring enabled")


def disable_policy_monitoring():
    """禁用策略监控"""
    global _global_monitor
    if _global_monitor is not None:
        report = _global_monitor.generate_report()
        logger.info(f"\n{report}")
    _global_monitor = None
    logger.info("Policy performance monitoring disabled")
