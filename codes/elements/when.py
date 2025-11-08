import time


class Every:
    """
  基于步数 (step) 的周期性触发器。
  每隔 `every` 步触发一次。
  """

    def __init__(self, every, initial=True):
        # 触发间隔步数
        self._every = every
        # 是否在第一次调用时立即触发
        self._initial = initial
        # 上一次触发时的参考步数 (会被初始化到 every 的倍数)
        self._prev = None

    def __call__(self, step):
        """
    检查当前步数 `step` 是否应触发。
    """
        step = int(step)

        # 负值：始终触发
        if self._every < 0:
            return True
        # 零值：永不触发
        if self._every == 0:
            return False

        if self._prev is None:
            # 第一次调用：初始化 _prev 为小于等于当前 step 的 every 倍数
            self._prev = (step // self._every) * self._every
            return self._initial  # 根据设置返回是否立即触发

        if step >= self._prev + self._every:
            # 如果当前步数超过了上次触发步数 + 间隔
            self._prev += self._every  # 更新上次触发步数
            # 【注意】：如果 step 远大于 _prev + _every，此实现只会递增一次 _prev。
            # 更严谨的实现可能会使用 while 循环或直接设置 _prev = (step // self._every) * self._every
            return True

        return False


class Ratio:
    """
  基于步数的重复次数计算器。
  用于确定自上次检查以来，如果以 `ratio` 的频率执行，应该执行多少次。
  """

    def __init__(self, ratio):
        # 执行频率：例如 ratio=0.1 表示每 10 步执行 1 次
        self._ratio = ratio
        # 上一次的参考步数
        self._prev = None

    def __call__(self, step):
        """
    根据步数差和频率计算应该执行的重复次数。

    Returns:
      int: 应执行的次数。
    """
        step = int(step)

        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1  # 负数频率始终返回 1

        if self._prev is None:
            self._prev = step
            return 1  # 第一次调用执行 1 次

        # 计算 (步数差 * 频率) 得到应执行的总次数
        repeats = int((step - self._prev) * self._ratio)

        # 更新参考步数：将执行次数转换为等价的步数，加到 _prev 上
        # 例如：如果 ratio=0.1, repeats=2，则 _prev 增加 2 / 0.1 = 20 步
        self._prev += repeats / self._ratio

        return repeats


class Once:
    """
  只执行一次的触发器。
  """

    def __init__(self):
        # 内部状态，标记是否已触发
        self._once = True

    def __call__(self):
        """
    第一次调用返回 True，之后返回 False。
    """
        if self._once:
            self._once = False
            return True
        return False


class Until:
    """
  基于步数的截止条件。
  在达到截止步数 `until` 之前返回 True。
  """

    def __init__(self, until):
        # 截止步数
        self._until = until

    def __call__(self, step):
        """
    检查当前步数是否小于截止步数。

    Returns:
      bool: 是否仍在截止步数之前。
    """
        step = int(step)

        # 如果 until 为 None 或 0 等 Falsy 值，则始终返回 True
        if not self._until:
            return True

        return step < self._until


class Clock:
    """
  基于时间的周期性触发器。
  每隔 `every` 秒触发一次。
  """

    def __init__(self, every, first=True):
        # 触发间隔（秒）
        self._every = every
        # 上次触发的时间点
        self._prev = None
        # 是否在第一次调用时立即触发
        self._first = first

    def __call__(self, step=None):
        """
    检查自上次触发以来是否已超过 `every` 秒。
    """
        # 负值：永不触发
        if self._every < 0:
            return False
        # 零值：始终触发
        if self._every == 0:
            return True

        now = time.time()  # 获取当前时间

        if self._prev is None:
            self._prev = now  # 初始化上次触发时间
            return self._first  # 根据设置返回是否立即触发

        if now >= self._prev + self._every:
            # 如果当前时间超过了上次触发时间 + 间隔

            # 【注意】：这里将 _prev 直接设置为 now，这意味着计时是相对于 *实际时间* 进行的，
            # 而不是基于严格的周期累加 (`self._prev += self._every`)。
            # 前者避免了时间漂移，但会导致下一个周期的时间间隔小于等于 `every`。
            self._prev = now
            return True

        return False