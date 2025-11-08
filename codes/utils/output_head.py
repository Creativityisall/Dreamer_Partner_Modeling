import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools import symlog, symexp, weight_init  # 导入对称对数/指数函数和权重初始化工具
from .networks import TransposeCNN  # 导入转置卷积网络（用于图像重构）

from typing import Callable


class Output:
    """
    抽象基类：定义了所有预测输出必须实现的接口。
    模型通过这个接口计算损失、获取预测值或采样。
    """

    def loss(self, target):
        """计算预测值与真实目标 (target) 之间的损失。"""
        raise NotImplementedError

    def pred(self):
        """返回分布的统计量（例如均值或概率）。"""
        # return the statistics
        raise NotImplementedError

    def sample(self):
        """从分布中采样一个样本。"""
        # generate a sample from the distribution
        raise NotImplementedError

    @property
    def mode(self):
        """返回分布的众数（最有可能的样本）。"""
        # return the most likely sample
        raise NotImplementedError


class MSE(Output):
    """均方误差（Mean Squared Error）：用于确定性连续值预测。"""

    def __init__(self, mean, squash=None):
        self.mean = mean  # 模型的预测均值
        self.squash = squash or (lambda x: x)  # 可选的后处理函数（例如归一化）

    def loss(self, target):
        assert target.shape == self.mean.shape, f"Target shape {target.shape} does not match mean shape {self.mean.shape}"
        # 计算 MSE 损失。注意：目标值 target.detach() 是从计算图中分离的
        return F.mse_loss(self.mean, self.squash(target).detach(), reduction="none")

    def pred(self):
        return self.mean


class SymlogMSE(Output):
    """
    Symlog MSE：在 Symlog 空间中计算 MSE，通常用于预测奖励或价值。
    这使得模型可以稳定地处理大范围的连续值。
    """

    def __init__(self, mean):
        self.mean = mean  # 模型的预测均值（位于 symlog 空间）

    def loss(self, target):
        assert target.shape == self.mean.shape, f"Target shape {target.shape} does not match mean shape {self.mean.shape}"
        # 目标值先经过 symlog 变换，再计算 MSE
        return F.mse_loss(self.mean, symlog(target).detach(), reduction="none")

    def pred(self):
        # 预测结果返回时，需要进行 symexp 逆变换回原空间
        return symexp(self.mean)


class SymexpMSE(Output):
    """
    Symexp MSE：预测值先经过 Symexp 变换，再计算 MSE。
    常用于预测那些必须为正的量，并保证输出在原始空间。
    """

    def __init__(self, mean):
        self.mean = mean

    def loss(self, target):
        assert target.shape == self.mean.shape, f"Target shape {target.shape} does not match mean shape {self.mean.shape}"
        # 模型的输出 symexp(mean) 与目标值 target 计算 MSE
        return F.mse_loss(symexp(self.mean), target.detach(), reduction="none")

    def pred(self):
        return symexp(self.mean)


class Binary(Output):
    """二元分布（Binary）：用于预测二值事件（如回合终止 terminated）。"""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits  # 逻辑值（未经过 sigmoid 激活的输出）
        self.probs = torch.sigmoid(logits)  # 概率值

    def loss(self, target):
        # 使用 BCEWithLogitsLoss，数值更稳定
        return F.binary_cross_entropy_with_logits(self.logits, target.detach(), reduction="none")

    def pred(self):
        return self.probs

    def sample(self):
        # 伯努利采样
        return torch.bernoulli(self.probs)

    @property
    def mode(self):
        # 众数：logits > 0 对应的即是概率 > 0.5 的类别
        return (self.logits > 0).float()


class TwoHot(Output):
    """
    TwoHot/Categorical 分布：用于将连续目标值离散化并进行预测（如 Dreamer V2）。
    通过 TwoHot 编码（将目标值编码为最接近的两个 bin 的加权组合）来近似分布。
    """

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor):
        self.logits = logits  # 分类的逻辑值
        self.bins = bins  # 离散化的中心点（bin 的值）
        self.num_bins = bins.shape[-1]

    def loss(self, target):
        target = target.detach()
        # 1. 找到目标值 target 落在哪个区间（below 和 above）
        below = (self.bins <= target).sum(-1, keepdim=True) - 1  # target 下方的 bin 索引
        above = self.num_bins - (self.bins > target).sum(-1, keepdim=True)  # target 上方的 bin 索引
        below = torch.clamp(below, min=0, max=self.num_bins - 1)
        above = torch.clamp(above, min=0, max=self.num_bins - 1)

        # 2. 计算权重（线性插值）
        equal = (below == above)
        # 如果 below == above，距离设为 1，否则计算距离
        dist_to_below = torch.where(equal, 1.0, torch.abs(self.bins.gather(-1, below) - target))
        dist_to_above = torch.where(equal, 1.0, torch.abs(self.bins.gather(-1, above) - target))
        total = dist_to_below + dist_to_above
        # 权重与距离成反比
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # 3. 计算 Two-Hot 目标
        # 使用 one_hot 向量和插值权重构造目标分布 (target distribution)
        target = (
                F.one_hot(below.squeeze(-1), num_classes=self.num_bins).float() * weight_below +
                F.one_hot(above.squeeze(-1), num_classes=self.num_bins).float() * weight_above
        )

        # 4. 计算损失：交叉熵
        assert self.logits.shape == target.shape, (self.logits.shape, target.shape)
        logprobs = F.log_softmax(self.logits, dim=-1)
        # 交叉熵 = - sum(target * logprobs)
        loss = - (target * logprobs).sum(-1, keepdim=True)
        return loss

    def pred(self):
        # 预测值是 Softmax 后的概率与 bin 值的加权平均（即期望值）
        probs = F.softmax(self.logits, dim=-1)
        return (probs * self.bins).sum(-1, keepdim=True)


# ====================================================================
# 预测头 (Head) 网络模块
# ====================================================================

class Head(nn.Module):
    """
    Head 基类，提供快捷方法来创建具体的 Output 分布实例。
    """

    def mse(self, x: torch.Tensor, squash: Callable | None = None) -> Output:
        return MSE(x, squash)

    def symlog_mse(self, x: torch.Tensor) -> Output:
        return SymlogMSE(x)

    def symexp_mse(self, x: torch.Tensor) -> Output:
        return SymexpMSE(x)

    def symexp_twohot(self, logits: torch.Tensor) -> Output:
        # 特殊处理：TwoHot 用于预测奖励等，这些值通常在 symexp 空间中处理
        return TwoHot(logits, self.bins)

    def binary(self, x: torch.Tensor) -> Output:
        return Binary(x)


class MLPHead(Head):
    """
    多层感知机预测头：用于处理标量或低维输出（如奖励、终止信号、动作的分布参数）。
    """

    def __init__(
            self,
            output: str,  # 指定输出类型（如 'symlog_mse', 'binary' 等）
            in_dim: int,
            hidden_dim: int,
            hidden_layers: int,
            out_dim: int,
            act: str = "SiLU",
            use_layernorm: bool = True,
            use_symlog: bool = False,
            out_scale: float = 1.0,  # 输出层权重初始化缩放
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.output = output
        self.use_symlog = use_symlog

        # 构建 MLP 隐藏层
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, device=device))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, device=device))
            layers.append(getattr(nn, act)())  # 激活函数，如 SiLU
            in_dim = hidden_dim
        self._mlp = nn.Sequential(*layers)
        weight_init(self._mlp)  # 初始化 MLP 权重

        # 构建输出层
        if output == "symexp_twohot":
            assert out_dim == 1  # TwoHot 通常用于预测单个连续值
            self.num_bins = 255
            # 定义 TwoHot 的 bin 中心点，并将其映射到 Symexp 空间
            bins = torch.linspace(start=-20, end=20, steps=self.num_bins, device=device)
            self.bins = symexp(bins)
            self._out = nn.Linear(in_dim, self.num_bins, device=device)  # 输出维度是 bin 的数量
        else:
            self._out = nn.Linear(in_dim, out_dim, device=device)

        weight_init(self._out, scale=out_scale)  # 初始化输出层权重

    def __call__(self, x: torch.Tensor) -> Output:
        # 如果配置了，先对输入进行 symlog 变换
        if self.use_symlog:
            x = symlog(x)
        # 通过 MLP 层
        x = self._mlp(x)
        # 通过输出层
        x = self._out(x)
        # 根据配置的输出类型创建并返回 Output 实例
        x = getattr(self, self.output)(x)
        return x


class FigureHead(Head):
    """
    图像预测头（解码器）：用于将隐状态解码为图像观测。
    """

    def __init__(
            self,
            in_dim: int,
            out_shape: tuple[int, int, int],  # 图像输出形状 (C, H, W)
            depth: int,  # CNN 深度
            mults: list[int],
            kernel: int,
            act: str = "SiLU",
            use_layernorm: bool = True,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        # 使用 TransposeCNN (转置卷积) 作为图像解码器
        self._transpose_cnn = TransposeCNN(
            in_dim=in_dim,
            out_shape=out_shape,
            depth=depth,
            mults=mults,
            kernel=kernel,
            act=act,
            use_layernorm=use_layernorm,
            device=device,
        )

    def __call__(self, x: torch.Tensor) -> Output:
        # 1. 隐状态通过 TransposeCNN 解码为图像
        x = self._transpose_cnn(x)
        # 2. 图像重构使用 MSE 损失。
        # 图像数据通常在 0-255 范围内，这里将目标值 squash 到 0-1 范围以匹配解码器输出
        x = self.mse(x, squash=lambda x: x / 255)
        return x