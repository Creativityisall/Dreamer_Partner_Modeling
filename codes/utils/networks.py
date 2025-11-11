import torch
import torch.nn as nn

from utils.tools import symlog, weight_init
from functools import partial

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        out_dim: int | None = None,
        act: str = "SiLU",
        use_layernorm: bool = True,
        use_symlog: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.use_symlog = use_symlog

        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, device=device))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, device=device))
            layers.append(getattr(nn, act)())
            in_dim = hidden_dim

        if out_dim is not None:
            layers.append(nn.Linear(in_dim, out_dim, device=device))

        self._mlp = nn.Sequential(*layers)
        weight_init(self._mlp)

    def forward(self, x: torch.Tensor):
        if self.use_symlog:
            x = symlog(x)
        x = self._mlp(x)
        return x

class ConvLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, device=device)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class CNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        depth: int,
        mults: list[int],
        kernel: int,
        output_dim: int,
        act: str = "SiLU",
        use_layernorm: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        H, W, C = input_shape
        self.depths = [C] + [depth * m for m in mults]
        self.cnn_layers = []
        for i in range(1, len(self.depths)):
            pad_h, pad_w = 1, 1
            self.cnn_layers.append(
                nn.Conv2d(
                    in_channels=self.depths[i-1],
                    out_channels=self.depths[i],
                    padding=(pad_h, pad_w),
                    kernel_size=kernel,
                    stride=2,
                    device=device,
                )
            )
            if use_layernorm:
                self.cnn_layers.append(ConvLayerNorm(self.depths[i], device=device))
            self.cnn_layers.append(getattr(nn, act)())
            H = (H + 2 * pad_h - kernel) // 2 + 1
            W = (W + 2 * pad_w - kernel) // 2 + 1
            C = self.depths[i]
        self.cnn_layers = nn.Sequential(*self.cnn_layers)
        self.cnn_layers.apply(weight_init)

        self.mlp_layers = MLP(
            in_dim=C * H * W,
            hidden_dim=output_dim,
            hidden_layers=1,
            use_layernorm=use_layernorm,
            device=device,
        )

    def forward(self, x: torch.Tensor):
        # flatten the batch dimensions
        batch_shape = x.shape[:-3]
        x = x.flatten(0, -4)

        # transpose and normalize
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0 - 0.5

        # forward pass
        x = self.cnn_layers(x)
        x = x.flatten(-3)
        x = self.mlp_layers(x)

        # unflatten the batch dimensions
        x = x.unflatten(0, batch_shape)
        return x

class TransposeCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: tuple[int, int, int],
        depth: int,
        mults: list[int],
        kernel: int,
        act: str = "SiLU",
        use_layernorm: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.depths = [depth * m for m in mults] + [out_shape[-1]]

        H, W, _ = out_shape
        layer_shapes = [(H, W)]
        pad_h, pad_w = 1, 1
        for _ in range(1, len(self.depths)):
            H = (H + 2 * pad_h - kernel) // 2 + 1
            W = (W + 2 * pad_w - kernel) // 2 + 1
            layer_shapes.append((H, W))
        layer_shapes.reverse()
        C = self.depths[0]
        self._init_shape = (C, H, W)

        self._mlp = MLP(
            in_dim=in_dim,
            hidden_dim=C * H * W,
            hidden_layers=1,
            act=act,
            use_layernorm=use_layernorm,
            device=device,
        )

        self.cnn_layers = []
        for i in range(1, len(self.depths)):
            target_h, target_w = layer_shapes[i]
            out_pad_h = target_h - ((H - 1) * 2 + kernel - 2 * pad_h)
            out_pad_w = target_w - ((W - 1) * 2 + kernel - 2 * pad_w)
            self.cnn_layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.depths[i-1],
                    out_channels=self.depths[i],
                    padding=(pad_h, pad_w),
                    output_padding=(out_pad_h, out_pad_w),
                    kernel_size=kernel,
                    stride=2,
                    device=device,
                )
            )
            if i < len(self.depths) - 1:
                if use_layernorm:
                    self.cnn_layers.append(ConvLayerNorm(self.depths[i], device=device))
                self.cnn_layers.append(getattr(nn, act)())
                H = (H - 1) * 2 + kernel - 2 * pad_h + out_pad_h
                W = (W - 1) * 2 + kernel - 2 * pad_w + out_pad_w
                assert (H, W) == layer_shapes[i]
        self.cnn_layers.append(nn.Sigmoid())
        self.cnn_layers = nn.Sequential(*self.cnn_layers)
        self.cnn_layers.apply(weight_init)

    def forward(self, x: torch.Tensor):
        # flatten the batch dimensions
        batch_shape = x.shape[:-1]
        x = x.flatten(0, -2)

        # forward pass
        x = self._mlp(x)
        x = x.unflatten(-1, self._init_shape)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 3, 1)

        # unflatten the batch dimensions
        x = x.unflatten(0, batch_shape)
        return x

class GRU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        use_layernorm: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Linear(in_dim+hidden_dim, 3*hidden_dim, device=device))
        if use_layernorm:
            layers.append(nn.LayerNorm(3*hidden_dim, device=device))
        self._mlp = nn.Sequential(*layers)
        weight_init(self._mlp)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input, hidden], dim=-1)
        x = self._mlp(x)
        update, reset, candidate = torch.split(x, [self.hidden_dim] * 3, dim=-1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update)
        hidden = (1 - update) * hidden + update * candidate
        return hidden

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        out_dim: int | None = None,
        act: str = "SiLU",
        norm_first: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation=act,
            norm_first=norm_first,
            batch_first=batch_first,
            dropout=dropout,
            device=device,
        )

        # init transformer
        ln = nn.LayerNorm(d_model, device=device)
        self._transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=ln,
        )
        weight_init(self._transformer)

        # init output layer
        if out_dim is not None:
            self._output_layer = nn.Linear(d_model, out_dim, device=device)
            weight_init(self._output_layer)
        else:
            self._output_layer = None

    def forward(
            self,
            x: torch.Tensor,
            num_batch_dims: int | None = None,
        ):
        if num_batch_dims is not None:
            batch_shape = x.shape[:num_batch_dims]
            x = x.flatten(0, num_batch_dims - 1)

        assert x.dim() == 3, x.shape  # (batch, seq_len, d_model)
        x = self._transformer(x)

        if num_batch_dims is not None:
            x = x.unflatten(0, batch_shape)
        if self._output_layer is not None:
            x = self._output_layer(x)
        return x

# Partner_model的所有操作都封装在这里
class Partner_Model(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        out_dim: int | None = None,
        act: str = "SiLU",
        use_layernorm: bool = True,
        use_symlog: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.use_symlog = use_symlog

        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, device=device))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, device=device))
            layers.append(getattr(nn, act)())
            in_dim = hidden_dim

        if out_dim is not None:
            layers.append(nn.Linear(in_dim, out_dim, device=device))

        self._mlp = nn.Sequential(*layers)
        weight_init(self._mlp)

    def forward(self, x: torch.Tensor):
        if self.use_symlog:
            x = symlog(x)
        x = self._mlp(x)
        return x