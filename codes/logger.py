from typing import List, Callable, Dict
import collections
import re
import numpy as np
from elements import printing, timer  # 假设 printing 和 timer 是自定义的工具模块


class Logger:
    """
    核心日志管理器。负责收集、暂存和刷新指标到指定的输出句柄（output_handles）。
    """

    def __init__(self, output_handles: List[Callable]):
        """
        初始化 Logger。
        Args:
            output_handles (List[Callable]): 负责处理和输出指标的函数或对象列表
                                            （例如 TerminalOutput 或 WandBOutput 的实例）。
        """
        self.output_handles = output_handles
        self._metrics = []  # 存储指标的暂存列表：[(step, key, value), ...]

    @timer.section('logger_add')
    def add(self, step: int, metrics: Dict, prefix: str | None = None):
        """
        将一批指标添加到暂存区（_metrics）。

        Args:
            step (int): 当前训练或环境的时间步。
            metrics (Dict): 要记录的 {键: 值} 字典（值通常为 NumPy 数组）。
            prefix (str | None): 可选的前缀（如 'train/', 'eval/'）。
        """
        for key, value in metrics.items():
            # 加上前缀，形成完整的键名
            key = f"{prefix}/{key}" if prefix else key
            # 确保值是 NumPy 数组
            value = np.asarray(value)
            # 检查键和值的有效性
            self._key_value_check(key, value)
            # 存储为三元组 (step, key, value)
            self._metrics.append((step, key, value))

    def _key_value_check(self, key: str, value: np.ndarray):
        """
        检查指标值的数据类型和形状是否有效。
        """
        # 检查数据类型：必须是数字类型
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(
                f"Type {value.dtype} for key '{key}' is not a number."
            )
        # 检查形状：只支持 0 维（标量）到 4 维（视频）的指标
        if len(value.shape) not in (0, 1, 2, 3, 4):
            raise ValueError(
                f"Shape {value.shape} for name '{key}' cannot be "
                "interpreted as scalar, vector, image, or video."
            )

    @timer.section('logger_flush')
    def flush(self):
        """
        将所有暂存的指标发送给所有输出句柄，并清空暂存区。
        """
        if not self._metrics:
            return
        # 遍历所有输出句柄，将暂存指标作为元组传递给它们
        for handle in self.output_handles:
            handle(tuple(self._metrics))
        # 清空暂存区
        self._metrics.clear()

    def close(self):
        """关闭 Logger（执行一次 final flush）。"""
        self.flush()


class TerminalOutput:
    """
    终端输出处理器。负责将标量指标格式化并打印到终端。
    支持通过正则表达式过滤指标，并限制打印数量。
    """

    def __init__(self, pattern=r'.*', name=None, limit=50):
        """
        初始化终端输出句柄。

        Args:
            pattern (str): 正则表达式，用于过滤要显示的指标键。
            name (str | None): 可选的名称，用于在终端页眉中显示。
            limit (int): 终端显示的标量指标的最大数量。
        """
        # 编译正则表达式，如果不是默认的 '.*'
        self._pattern = (pattern != r'.*') and re.compile(pattern)
        self._name = name
        self._limit = limit

    @timer.section('terminal')
    def __call__(self, summaries):
        """
        接收来自 Logger 的指标数据 (step, key, value) 并进行处理和打印。
        """
        # 获取最新的时间步
        step = max(s for s, _, _, in summaries)

        # 提取并过滤标量指标：只保留 0 维的 NumPy 数组
        scalars = {
            k: float(v) for _, k, v in summaries
            if isinstance(v, np.ndarray) and len(v.shape) == 0}

        # 应用正则表达式过滤
        if self._pattern:
            scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
        else:
            # 如果没有过滤，则应用数量限制
            truncated = 0
            if len(scalars) > self._limit:
                truncated = len(scalars) - self._limit
                # 截断字典到指定限制
                scalars = dict(list(scalars.items())[:self._limit])

        # 格式化标量值
        formatted = {k: self._format_value(v) for k, v in scalars.items()}

        # 构建页眉
        if self._name:
            header = f'{"-" * 20}[{self._name} Step {step:_}]{"-" * 20}'
        else:
            header = f'{"-" * 20}[Step {step:_}]{"-" * 20}'

        # 构建内容主体
        content = ''
        if self._pattern:
            content += f"Metrics filtered by: '{self._pattern.pattern}'"
        elif truncated:
            content += f'{truncated} metrics truncated, filter to see specific keys.'
        content += '\n'

        # 拼接格式化的指标
        if formatted:
            content += ' / '.join(f'{k} {v}' for k, v in formatted.items())
        else:
            content += 'No metrics.'

        # 打印到终端
        printing.print_(f'\n{header}\n{content}\n', flush=True)

    def _format_value(self, value):
        """
        格式化标量值：保留两位小数或使用科学记数法。
        """
        value = float(value)
        if value == 0:
            return '0'
        # 常规格式化：0.01 到 10000 之间
        elif 0.01 < abs(value) < 10000:
            value = f'{value:.2f}'
            # 清理尾随零和小数点
            value = value.rstrip('0')
            value = value.rstrip('0')
            value = value.rstrip('.')
            return value
        # 科学记数法格式化
        else:
            value = f'{value:.1e}'
            # 清理科学记数法的格式，例如 '1.0e+05' -> '1e5'
            value = value.replace('.0e', 'e')
            value = value.replace('+0', '')
            value = value.replace('+', '')
            value = value.replace('-0', '-')
        return value


class WandBOutput:
    """
    Weights & Biases (WandB) 输出处理器。
    负责将收集到的指标转换为 WandB 对象（如 Image, Video, Histogram）并上传。
    """

    def __init__(self, name, pattern=r'.*', **kwargs):
        """
        初始化 WandB 输出句柄。

        Args:
            name (str): WandB 运行的名称。
            pattern (str): 正则表达式，用于过滤要上传的指标键。
            **kwargs: 传递给 wandb.init() 的其他参数。
        """
        self._pattern = re.compile(pattern)
        # 导入 wandb 并在初始化时启动新的运行
        import wandb
        wandb.init(name=name, **kwargs)
        self._wandb = wandb

    @timer.section('wandb')
    def __call__(self, summaries):
        """
        接收指标，根据类型转换为 WandB 对象，并调用 wandb.log() 上传。
        """
        # 按时间步分组指标：{step: {key: wandb_value}}
        bystep = collections.defaultdict(dict)
        wandb = self._wandb

        for step, name, value in summaries:
            # 应用正则表达式过滤
            if not self._pattern.search(name):
                continue

            # 根据值的形状和类型转换为 WandB 对象
            if isinstance(value, str):
                bystep[step][name] = value  # 字符串日志
            elif len(value.shape) == 0:
                bystep[step][name] = float(value)  # 标量
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)  # 直方图
            elif len(value.shape) in (2, 3):
                # 图像 (Image): 2D (H, W) 或 3D (H, W, C)
                value = value[..., None] if len(value.shape) == 2 else value
                assert value.shape[3] in [1, 3, 4], value.shape  # 检查通道数 (C)

                # 转换为 uint8 [0, 255] 格式
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)

                # WandB Image 需要 (C, H, W) 格式
                value = np.transpose(value, [2, 0, 1])
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 4:
                # 视频 (Video): 4D (T, H, W, C)
                assert value.shape[3] in [1, 3, 4], value.shape  # 检查通道数 (C)

                # WandB Video 需要 (T, C, H, W) 格式
                value = np.transpose(value, [0, 3, 1, 2])

                # 转换为 uint8 [0, 255] 格式
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)

                bystep[step][name] = wandb.Video(value)

        # 遍历按时间步分组的指标，并调用 wandb.log() 上传
        for step, metrics in bystep.items():
            self._wandb.log(metrics, step=step)