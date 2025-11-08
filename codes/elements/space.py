import numpy as np


class Space:
    """
  定义和管理数据空间（如强化学习中的观测空间或动作空间）。
  封装了数据的 dtype, shape, low 和 high 边界，并提供采样和范围检查功能。
  """

    def __init__(self, dtype, shape=(), low=None, high=None):
        # 【注意】对于整数类型，high 是允许的最大值再加一（即上界是排他的）。
        # 确保 shape 是一个元组，即使传入的是单个整数
        shape = (shape,) if isinstance(shape, int) else shape

        # 存储 NumPy 数据类型
        self._dtype = np.dtype(dtype)
        assert self._dtype is not object, self._dtype  # 不支持 object 类型
        assert isinstance(shape, tuple), shape

        # 推断下限 (low) 和上限 (high)
        self._low = self._infer_low(dtype, shape, low, high)
        self._high = self._infer_high(dtype, shape, low, high)

        # 推断最终的形状
        self._shape = self._infer_shape(dtype, shape, self._low, self._high)

        # 标记空间是否为离散的 (整数或布尔类型)
        self._discrete = (
                np.issubdtype(self.dtype, np.integer) or self.dtype == bool)

        # 用于采样的随机数生成器
        self._random = np.random.RandomState()

    @property
    def dtype(self):
        """返回数据类型。"""
        return self._dtype

    @property
    def shape(self):
        """返回数据形状。"""
        return self._shape

    @property
    def low(self):
        """返回空间的下限（NumPy 数组）。"""
        return self._low

    @property
    def high(self):
        """返回空间的上限（NumPy 数组）。"""
        return self._high

    @property
    def discrete(self):
        """返回空间是否为离散的布尔值。"""
        return self._discrete

    @property
    def classes(self):
        """
    返回离散空间中类的数量。
    仅适用于离散空间，计算公式为 high - low。
    """
        assert self.discrete
        classes = self._high - self._low
        if not classes.ndim:
            # 如果是标量，转换为 Python int
            classes = int(classes.item())
        return classes

    def __repr__(self):
        """返回 Space 对象的简洁字符串表示。"""
        # 仅显示 low 和 high 数组的最小值，以保持简洁
        low = None if self.low is None else self.low.min()
        high = None if self.high is None else self.high.min()
        return (
            f'Space({self.dtype.name}, '
            f'shape={self.shape}, '
            f'low={low}, '
            f'high={high})')

    def __contains__(self, value):
        """
    实现 `in` 运算符：检查给定值是否在当前空间内。
    """
        value = np.asarray(value)
        # 对于字符串类型，只检查值的数据类型是否为字符串
        if np.issubdtype(self.dtype, str):
            return np.issubdtype(value.dtype, str)

        # 形状检查
        if value.shape != self.shape:
            return False

        # 上限检查
        if (value > self.high).any():
            return False

        # 下限检查
        if (value < self.low).any():
            return False

        # 数据类型检查
        if value.dtype != self.dtype:
            return False

        return True

    def sample(self):
        """
    从空间中均匀采样一个值。
    对于浮点数，会使用 dtype 的实际 min/max 来限制采样范围。
    """
        low, high = self.low, self.high

        # 对于浮点类型，使用 dtype 的实际 min/max 来限制采样范围，避免超出浮点数的极限值
        if np.issubdtype(self.dtype, np.floating):
            low = np.maximum(np.ones(self.shape) * np.finfo(self.dtype).min, low)
            high = np.minimum(np.ones(self.shape) * np.finfo(self.dtype).max, high)

        # 使用 NumPy 的 uniform 在 [low, high) 范围内采样，并转换为目标 dtype
        return self._random.uniform(low, high, self.shape).astype(self.dtype)

    def _infer_low(self, dtype, shape, low, high):
        """
    推断空间的下限数组 (self._low)。
    """
        if np.issubdtype(dtype, str):
            assert low is None, low
            return None

        if low is not None:
            # 如果提供了 low，尝试广播到 shape
            try:
                return np.broadcast_to(low, shape)
            except ValueError:
                raise ValueError(f'Cannot broadcast {low} to shape {shape}')

        # 如果未提供 low，则根据 dtype 推断默认下限
        elif np.issubdtype(dtype, np.floating):
            return -np.inf * np.ones(shape)  # 浮点数默认下限为 -inf
        elif np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).min * np.ones(shape, dtype)  # 整数默认下限为 dtype 的最小值
        elif np.issubdtype(dtype, bool):
            return np.zeros(shape, bool)  # 布尔值默认下限为 False (0)
        else:
            raise ValueError('Cannot infer low bound from shape and dtype.')

    def _infer_high(self, dtype, shape, low, high):
        """
    推断空间的上限数组 (self._high)。
    """
        if np.issubdtype(dtype, str):
            assert high is None, high
            return None

        if high is not None:
            # 如果提供了 high，尝试广播到 shape
            try:
                return np.broadcast_to(high, shape)
            except ValueError:
                raise ValueError(f'Cannot broadcast {high} to shape {shape}')

        # 如果未提供 high，则根据 dtype 推断默认上限
        elif np.issubdtype(dtype, np.floating):
            return np.inf * np.ones(shape)  # 浮点数默认上限为 +inf
        elif np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max * np.ones(shape, dtype)  # 整数默认上限为 dtype 的最大值
        elif np.issubdtype(dtype, bool):
            return np.ones(shape, bool)  # 布尔值默认上限为 True (1)
        else:
            raise ValueError('Cannot infer high bound from shape and dtype.')

    def _infer_shape(self, dtype, shape, low, high):
        """
    推断最终的 shape 元组。
    主要用于处理 shape 为 None 的情况，此时从 low 或 high 的 shape 中获取。
    """
        # 如果 shape 为 None 且 low 或 high 已提供，则从它们的 shape 中获取
        if shape is None and low is not None:
            shape = low.shape
        if shape is None and high is not None:
            shape = high.shape

        # 确保 shape 是一个元组
        if not hasattr(shape, '__len__'):
            shape = (shape,)

        # 确保 shape 中的所有维度都是正整数
        assert all(dim and dim > 0 for dim in shape), shape

        return tuple(shape)