import math
import operator
from collections import defaultdict
from functools import partial as bind

import numpy as np


class Agg:
    """
  通用的数据聚合器类。它维护一个字典，其中键是指标名称，值是Reducer对象的列表。
  它根据数据的类型（标量、数组、字符串等）自动选择合适的聚合方式。
  """

    def __init__(self, maxlen=1e6):
        # 存储指标的Reducer对象列表，格式为 {key: [Reducer1, Reducer2, ...]}
        self.reducers = defaultdict(list)
        # 存储当一个key有多个Reducer时的名称，格式为 {key: (name1, name2, ...)}
        self.names = {}
        # 集合操作（如Stack, Concat）的最大长度限制
        self.maxlen = int(maxlen)

    def add(self, key_or_dict, value=None, agg='default', prefix=None):
        """
    添加新的数据点进行聚合。
    可以传入单个键值对，或一个字典。
    """
        if value is not None:
            # 处理单个键值对
            self._add_single(key_or_dict, value, agg, prefix)
            return
        # 处理字典形式的多个键值对
        for key, value in key_or_dict.items():
            self._add_single(key, value, agg, prefix)

    def result(self, reset=True, prefix=None):
        """
    计算并返回所有聚合指标的当前值。
    Args:
      reset (bool): 计算完成后是否清空所有聚合器。
      prefix (str): 可选的前缀，添加到所有指标名称前。
    Returns:
      dict: 包含所有聚合指标值的字典。
    """
        metrics = {}
        for key, reducers in self.reducers.items():
            if len(reducers) == 1:
                # 如果只有一个聚合器，直接使用 key
                metrics[key] = reducers[0].current()
            else:
                # 如果有多个聚合器（如同时计算 avg 和 max），使用 key/name 格式
                for name, reducer in zip(self.names[key], reducers):
                    metrics[f'{key}/{name}'] = reducer.current()
        if prefix:
            # 应用结果前缀
            metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
        # 清空聚合器（如果 reset=True）
        reset and self.reset()
        return metrics

    def reset(self):
        """清空所有存储的 Reducer 对象，重置聚合器状态。"""
        self.reducers.clear()

    def _add_single(self, key, value, agg, prefix):
        """
    处理单个键值对的内部方法。
    """
        # 添加可选的前缀到键名
        key = f'{prefix}/{key}' if prefix else key
        reducers = self.reducers[key]

        # 1. 如果该 key 已经有 Reducer，则直接更新它们
        if reducers:
            for reducer in reducers:
                reducer.update(value)
            return

        # 2. 如果是第一次看到该 key，则根据 agg 参数或数据类型自动选择 Reducer
        if agg == 'default':
            # 默认行为：根据输入数据的维度和类型自动推断聚合方式
            ndim = np.asarray(value).ndim
            if np.issubdtype(np.asarray(value).dtype, str):
                agg = 'last'  # 字符串取最后一个
            elif ndim == 0:
                agg = 'avg'  # 标量（0维）取平均
            elif ndim == 1:  # 分布或一维数组
                agg = 'concat'  # 连接起来 (用于统计分布或短序列)
            elif ndim == 4:  # 视频帧或图像序列 (如 (T, H, W, C))
                agg = 'concat'  # 连接起来
            else:
                agg = 'last'  # 其他情况取最后一个

        # 处理 agg 是一个字符串或一个聚合名称元组的情况
        if isinstance(agg, str):
            aggs = (agg,)
            self.names[key] = None  # 单个聚合时不需要子名称
        else:
            aggs = agg
            self.names[key] = aggs  # 多个聚合时存储子名称

        # 3. 为每个选定的聚合类型创建 Reducer 对象
        for agg in aggs:
            if agg == 'avg':
                reducer = Mean(value)
            elif agg == 'sum':
                reducer = Sum(value)
            elif agg == 'min':
                reducer = Min(value)
            elif agg == 'max':
                reducer = Max(value)
            elif agg == 'stack':
                # 将值堆叠 (Stack)，用于形状一致的数组
                reducer = Stack(value, self.maxlen)
            elif agg == 'concat':
                # 将值连接 (Concatenate)，用于数组
                reducer = Concat(value, self.maxlen)
            elif agg == 'last':
                # 只保留最新的值
                reducer = Last(value)
            else:
                raise ValueError(f"Unsupported aggregation type: {agg}")

            # 将创建的 Reducer 添加到字典中
            reducers.append(reducer)


class Reducer:
    """
  基础 Reducer 类，处理加法、最大值、最小值的聚合逻辑。
  它处理标量和Numpy数组两种类型，并负责管理中间结果 (interm)。
  """

    def __init__(self, scalar_fn, array_fn, initial):
        # 检查初始值是否为标量
        self.is_scalar = isinstance(initial, (int, float))
        # 根据类型选择标量或数组操作函数
        self.fn = scalar_fn if self.is_scalar else array_fn
        # 存储中间结果
        self.interm = self._input(initial)
        # 记录更新次数
        self.count = 1

    def update(self, value):
        """用新的值更新中间结果。"""
        value = self._input(value)
        if self._isnan(value):
            return
        # 如果中间结果是 NaN，则用当前值初始化它
        if self._isnan(self.interm):
            self.interm = value
            return
        # 执行聚合函数 (加法, min, max 等)
        self.interm = self.fn(self.interm, value)
        self.count += 1

    def current(self):
        """返回聚合结果。"""
        return np.array(self.interm)

    def _input(self, value):
        """规范化输入值：标量保持不变，数组转为 float64 numpy 数组。"""
        if self.is_scalar:
            return value
        else:
            # 使用 np.float64 以确保精度
            return np.asarray(value, np.float64)

    def _isnan(self, value):
        """检查值是否为 NaN (Not a Number)。"""
        if self.is_scalar:
            return math.isnan(value)
        else:
            return np.isnan(value).any()


class Mean:
    """
  计算平均值 (Mean) 的聚合器，通过组合 Sum Reducer 实现 (Sum / Count)。
  """

    def __init__(self, initial):
        self.reducer = Sum(initial)

    def update(self, value):
        self.reducer.update(value)

    def current(self):
        """返回总和除以计数，得到平均值。"""
        return self.reducer.current() / self.reducer.count


class Stack:
    """
  将输入值堆叠 (Stack) 起来（要求所有数组形状一致）。
  """

    def __init__(self, initial, maxlen=1e5):
        self.stack = [initial]
        self.maxlen = int(maxlen)

    def update(self, value):
        # 在达到最大长度前进行堆叠
        if len(self.stack) < self.maxlen:
            self.stack.append(value)

    def current(self):
        """返回堆叠后的 NumPy 数组。"""
        return np.stack(self.stack)


class Concat:
    """
  将输入值沿着第一个轴连接 (Concatenate) 起来（常用于收集序列数据）。
  """

    def __init__(self, initial, maxlen=1e5):
        self.values = [initial]
        # 记录当前的总长度
        self.len = len(self.values[-1])
        self.maxlen = int(maxlen)

    def update(self, value):
        # 在达到最大长度前进行连接
        if self.len < self.maxlen:
            # 只添加能容纳的部分，防止超限
            self.values.append(value[:self.maxlen - self.len])
            self.len += len(self.values[-1])

    def current(self):
        """返回连接后的 NumPy 数组。"""
        return np.concatenate(self.values)


class Last:
    """
  只保留最后一次更新的值。
  """

    def __init__(self, initial):
        self.value = initial

    def update(self, value):
        self.value = value

    def current(self):
        return self.value


# 使用 functools.partial (bind) 创建预配置的 Reducer 实例
# 这样可以避免为 Sum, Min, Max 重写完整的 Reducer 类
Sum = bind(
    Reducer,
    operator.add,  # 标量加法函数
    lambda x, y: np.add(x, y, out=x, dtype=np.float64)  # 数组加法函数 (in-place)
)
Min = bind(
    Reducer,
    min,  # 标量最小值函数
    lambda x, y: np.minimum(x, y, out=x, dtype=np.float64)  # 数组最小值函数 (in-place)
)
Max = bind(
    Reducer,
    max,  # 标量最大值函数
    lambda x, y: np.maximum(x, y, out=x, dtype=np.float64)  # 数组最大值函数 (in-place)
)