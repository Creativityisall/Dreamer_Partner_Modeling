import contextlib
import threading
import time
from collections import defaultdict

import numpy as np


class Timer:
    """
  多线程安全的性能计时器。
  用于测量代码段（sections）的执行时间、频率，并生成详细统计报告。
  """

    def __init__(self, enabled=True):
        # 是否启用计时器
        self.enabled = enabled

        # 线程栈：存储每个线程当前处于的计时器层级（用于处理嵌套计时）
        # key: 线程ID (threading.get_ident()), value: 当前计时器名称列表
        self.stack = defaultdict(list)

        # 记录所有已遇到的完整的计时路径 (如 'main/load', 'main/train/forward')
        self.paths = set()

        # 统计数据字典，key 都是完整的路径 (path)
        self.mins = defaultdict(lambda: np.inf)  # 最小执行时间 (纳秒)
        self.maxs = defaultdict(lambda: 0)  # 最大执行时间 (纳秒)
        self.sums = defaultdict(lambda: 0)  # 总执行时间 (纳秒)
        self.counts = defaultdict(lambda: 0)  # 执行次数

        # 计时器开始时间点 (用于计算时间占比 frac)
        self.start = time.perf_counter_ns()

        # 保护状态：在写入或重置统计数据时设置为 True，防止数据竞争
        self.writing = False

        # 扩展列表：允许外部注册额外的计时上下文管理器
        self.extensions = []

    @contextlib.contextmanager
    def section(self, name):
        """
    上下文管理器：开始和结束一个计时器部分。
    使用方式：with timer.section('name'): ...
    """
        if not self.enabled:
            yield
            return

        # 获取当前线程的计时栈
        stack = self.stack[threading.get_ident()]

        # 检查递归：不允许同一个名字在同一线程的栈中重复出现
        if name in stack:
            raise RuntimeError(
                f"Tried to recursively enter timer section {name} " +
                f"from {'/'.join(stack)}.")

        stack.append(name)  # 入栈
        path = '/'.join(stack)  # 完整的计时路径
        start = time.perf_counter_ns()  # 记录开始时间

        try:
            # 处理扩展：如果定义了扩展，则在扩展的上下文中运行主代码块
            if self.extensions:
                with contextlib.ExitStack() as es:
                    # 依次进入所有注册的扩展上下文
                    [es.enter_context(ext(path)) for ext in self.extensions]
                    yield
            else:
                # 没有扩展时直接运行主代码块
                yield

        finally:
            # 退出时：计算持续时间并更新统计数据
            dur = time.perf_counter_ns() - start
            stack.pop()  # 出栈

            # 仅在非写入状态下更新统计数据，以确保多线程安全
            if not self.writing:
                self.paths.add(path)
                self.sums[path] += dur
                self.mins[path] = min(self.mins[path], dur)
                self.maxs[path] = max(self.maxs[path], dur)
                self.counts[path] += 1

    def wrap(self, name, obj, methods):
        """
    将计时器 section 应用于对象的一组方法。
    """
        for method in methods:
            # 为每个方法创建一个 section 上下文管理器装饰器
            decorator = self.section(f'{name}.{method}')
            # 使用装饰器包装原始方法
            setattr(obj, method, decorator(getattr(obj, method)))

    def stats(self, reset=True, log=False):
        """
    计算并返回详细的计时统计数据。
    Args:
      reset (bool): 是否在生成报告后重置计时器。
      log (bool): 是否将统计摘要打印到控制台。
    Returns:
      dict: 包含所有路径统计数据的字典。
    """
        if not self.enabled:
            return {}

        self.writing = True
        # 短暂等待以确保其他线程在获取锁前能完成当前 section 的退出
        time.sleep(0.001)

        # 计算当前时间点和经过的总时间 (用于计算时间占比 frac)
        now = time.perf_counter_ns()
        passed = now - self.start
        self.start = now  # 更新 start 时间

        mets = {}
        div = lambda x, y: x and x / y  # 安全除法 (避免除以零)
        keys = list(self.paths)

        # 计算详细的指标 (转换为秒和分数)
        for key in keys:
            # 转换为秒 (除以 1e9)
            mets.update({
                f'{key}/sum': self.sums[key] / 1e9,
                f'{key}/min': self.mins[key] / 1e9,
                f'{key}/max': self.maxs[key] / 1e9,
                f'{key}/mean': div(self.sums[key], self.counts[key]) / 1e9,
                # 计算时间占比 (占总流逝时间的比例)
                f'{key}/frac': self.sums[key] / passed,
                f'{key}/count': self.counts[key],
            })

        self.writing = False  # 释放写入锁

        # ------------------ 格式化摘要 ------------------

        lines = []
        longest = max(len(key) for key in keys) if keys else 0
        # 按时间占比 frac 降序排序
        descending = sorted(keys, key=lambda k: -mets[f'{k}/frac'])

        for key in descending:
            count = mets[f'{key}/count']
            if not count:
                continue

            # 格式化输出字符串
            perc = '{:3.0f}'.format(100 * mets[f'{key}/frac'])
            min_ = '{:.1f}'.format(mets[f'{key}/min'])
            mean = '{:.1f}'.format(mets[f'{key}/mean'])
            max_ = '{:.1f}'.format(mets[f'{key}/max'])
            detail = f'min={min_}s, mean={mean}s, max={max_}s, n={count}'
            space = ' ' * (longest - len(key))

            # 格式: - 10% path/to/section   (min=0.1s, mean=0.5s, max=1.0s, n=10)
            lines.append(f'- {perc}% {key} {space} ({detail})')

        mets['summary'] = '\n'.join(lines)

        if log:
            print('Timer:', mets['summary'], sep='\n')

        reset and self.reset()  # 根据 reset 参数决定是否重置
        return mets

    def reset(self):
        """
    重置所有统计数据和计时器。
    """
        if not self.enabled:
            return

        self.writing = True
        time.sleep(0.001)  # 短暂等待

        # 清空所有统计字典
        self.sums.clear()
        self.mins.clear()
        self.maxs.clear()
        self.counts.clear()
        self.paths.clear()  # 也清空路径列表

        self.start = time.perf_counter_ns()
        self.writing = False


# ------------------ 全局实例和别名 ------------------

# 创建全局 Timer 实例，方便在项目各处使用
global_timer = Timer()

# 为全局实例的方法创建简洁别名
section = global_timer.section
wrap = global_timer.wrap
stats = global_timer.stats
reset = global_timer.reset
extensions = global_timer.extensions