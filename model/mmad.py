import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import numpy as np
from hilbert import decode, encode
from pyzorder import ZOrderIndexer

from basicsr.archs.arch_util import flow_warp


# ========== Decoder ==========
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False,
                              dilation=dilation)


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class SCANS(nn.Module):
    def __init__(self, size=16, dim=2, scan_type='scan', ):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)
        if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
            locs_flat = indexes
        elif 'scan' == scan_type:
            indexes = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1)
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        elif 'hilbert' == scan_type:
            bit = int(math.log2(size))
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(
            img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(
            img.shape), img)
        return img_decode


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            size=8,
            scan_type='scan',
            num_direction=8,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.num_direction = num_direction

        x_proj_weight = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
                         for _ in range(self.num_direction)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))
        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.num_direction)]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.scans = SCANS(size=size, scan_type=scan_type)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = self.num_direction
        xs = []
        if K >= 2:
            xs.append(self.scans.encode(x.view(B, -1, L)))
        if K >= 4:
            xs.append(self.scans.encode(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)))
        if K >= 8:
            xs.append(self.scans.encode(torch.rot90(x, k=1, dims=(2, 3)).contiguous().view(B, -1, L)))
            xs.append(self.scans.encode(
                torch.transpose(torch.rot90(x, k=1, dims=(2, 3)), dim0=2, dim1=3).contiguous().view(B, -1, L)))
        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y = xs

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []
        if K >= 2:
            ys.append(self.scans.decode(out_y[:, 0]))
            ys.append(self.scans.decode(inv_y[:, 0]))
        if K >= 4:
            ys.append(
                torch.transpose(self.scans.decode(out_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B,
                                                                                                                    -1,
                                                                                                                    L))
            ys.append(
                torch.transpose(self.scans.decode(inv_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B,
                                                                                                                    -1,
                                                                                                                    L))
        if K >= 8:
            ys.append(
                torch.rot90(self.scans.decode(out_y[:, 2]).view(B, -1, W, H), k=3, dims=(2, 3)).contiguous().view(B, -1,
                                                                                                                  L))
            ys.append(
                torch.rot90(self.scans.decode(inv_y[:, 2]).view(B, -1, W, H), k=3, dims=(2, 3)).contiguous().view(B, -1,
                                                                                                                  L))
            ys.append(
                torch.rot90(torch.transpose(self.scans.decode(out_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3,
                            dims=(2, 3)).contiguous().view(B, -1, L))
            ys.append(
                torch.rot90(torch.transpose(self.scans.decode(inv_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3,
                            dims=(2, 3)).contiguous().view(B, -1, L))
        y = sum(ys)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            size: int = 8,
            scan_type='scan',
            num_direction=4,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, size=size,
                                   scan_type=scan_type, num_direction=num_direction, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B * N, C)

    x[idx1.reshape(-1)] = x1.reshape(B * N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B * N2, C)

    x = x.reshape(B, N, C)
    return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Predictor(nn.Module):
    def __init__(self, hidden_dim, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.window_size = 1
        cdim = hidden_dim + 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim // 4, 1),
            LayerNorm(cdim // 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim // 4, cdim // 8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim // 8, 2, 1),
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim // 4, hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(1, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_x):
        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)

        x = torch.mean(x, keepdim=True, dim=1)
        x = rearrange(x, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training:
            return mask, offsets, ca, sa
        else:
            score = pred_score[:, :, 0]
            B, N = score.shape
            r = torch.mean(mask, dim=(0, 1)) * 1.0
            if self.ratio == 1:
                num_keep_node = N  # int(N * r) #int(N * r)
            else:
                num_keep_node = min(int(N * r * 2 * self.ratio), N)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], offsets, ca, sa


class Mixer(torch.nn.Module):
    def __init__(self,
                 hidden_dim: int = 0,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 depth: int = 2,
                 size: int = 8,
                 scan_type: str = 'scan',
                 num_direction: int = 8,
                 **kwargs,
                 ):
        super().__init__()
        self.window_size = 1
        self.route = Predictor(hidden_dim, ratio=0.5)
        self.project_v = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True)
        self.project_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.project_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 7, padding=3, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.project_out = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True)

        self.act = nn.GELU()
        self.ssm_blocks_1 = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zorder', num_direction=8, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zigzag', num_direction=8, **kwargs),
        ])
        self.ssm_blocks_2 = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zorder', num_direction=8, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zigzag', num_direction=8, **kwargs),
        ])
        self.ssm_blocks_3 = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zorder', num_direction=8, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zigzag', num_direction=8, **kwargs),
        ])
        self.ssm_blocks_4 = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zorder', num_direction=8, **kwargs),
            VSSBlock(hidden_dim=hidden_dim // 4, drop_path=drop_path, norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type='zigzag', num_direction=8, **kwargs),
        ])
        self.ssm_pro = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        N, C, H, W = x.shape
        v = self.project_v(x)
        pos = torch.stack(torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')) \
            .type_as(x).unsqueeze(0).repeat(N, 1, 1, 1)
        _condition = torch.cat([x, pos], dim=1)
        mask, offsets, ca, sa = self.route(_condition)

        q = x
        k = x + flow_warp(x, offsets.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border')
        qk = torch.cat([q, k], dim=1)

        vs = v * sa

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training:
            N_ = v.shape[1]
            v1, v2 = v * mask, vs * (1 - mask)
            qk1 = qk * mask
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1, v2 = batch_index_select(v, idx1), batch_index_select(vs, idx2)
            qk1 = batch_index_select(qk, idx1)

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, 'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)', b=N, n=N_, dh=self.window_size,
                           dw=self.window_size)

        if not self.training:
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H // self.window_size, w=W // self.window_size, dh=self.window_size, dw=self.window_size
        )

        out = rearrange(attn_out, 'b c h w -> b h w c')
        o1, o2, o3, o4 = torch.chunk(out, 4, dim=-1)
        for blk in self.ssm_blocks_1:
            o1 = blk(o1)
        for blk in self.ssm_blocks_2:
            o2 = blk(o2)
        for blk in self.ssm_blocks_3:
            o3 = blk(o3)
        for blk in self.ssm_blocks_4:
            o4 = blk(o4)
        out = torch.cat([o1, o2, o3, o4], dim=-1)
        out = rearrange(out, 'b h w c -> b c h w', h=H, w=W)
        out = self.ssm_pro(out)

        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)

        return out


class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()
        self.dim = dim
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class HybridBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            depth: int = 2,
            size: int = 8,
            scan_type: str = 'scan',
            num_direction: int = 8,
            **kwargs,
    ):
        super().__init__()
        self.mixer = Mixer(hidden_dim, drop_path, norm_layer, attn_drop_rate,
                              d_state, depth, size, scan_type, num_direction, **kwargs)
        self.norm1 = LayerNorm(hidden_dim)
        self.ffn = GatedFeedForward(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b h w c -> b c h w')
        res = self.mixer(x)
        x = self.norm1(x + res)
        res = self.ffn(x)
        x = self.norm2(x + res)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class GroupSSMBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int = 0,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 depth: int = 2,
                 size: int = 8,
                 scan_type: str = 'scan',
                 num_direction: int = 8,
                 is_light_sr: bool = False,
                 **kwargs, ):
        super(GroupSSMBlock, self).__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.smm_blocks = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs)
            for i in range(depth)])
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        out_ssm = x
        for blk in self.smm_blocks:
            out_ssm = blk(out_ssm)
        x = input * self.skip_scale + self.drop_path(out_ssm)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                        1).contiguous()
        return x


class HSS(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            size=8,
            scan_type='scan',
            num_direction=4,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        if depth % 3 == 0:
            self.blocks = nn.ModuleList([
                HybridBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=3,
                    num_direction=num_direction,
                )
                for i in range(depth // 3)])
        elif depth % 2 == 0:
            self.blocks = nn.ModuleList([
                HybridBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=2,
                    num_direction=num_direction,
                )
                for i in range(depth // 2)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class HMD(nn.Module):
    def __init__(self, dims_decoder=[512, 256, 128, 64], depths_decoder=[2, 2, 2, 2], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, scan_type='scan', num_direction=4, ):
        super().__init__()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = HSS(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                size=8 * 2 ** (i_layer),
                scan_type=scan_type,
                num_direction=num_direction,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        out_features = []
        for i, layer in enumerate(self.layers_up):
            x = layer(x)
            if i != 0:
                out_features.insert(0, rearrange(x, 'b h w c -> b c h w'))
        return out_features


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class DynamicMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_mask = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(1, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, tensor):
        batch_size, channels, height, width = tensor.shape
        x = torch.mean(tensor, keepdim=True, dim=1)  # B, 1, H, W
        x = rearrange(x, 'b c h w -> b (h w) c')
        pred_score = self.out_mask(x)  # B, (H, W), 2
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
        mask = rearrange(mask, 'b (h w) c -> b c h w', h=height, w=width)
        masked_tensor = tensor * mask.expand(-1, channels, -1, -1)
        return masked_tensor


class MambaFuser(nn.Module):
    def __init__(self, block, layers, width_per_group=64, norm_layer=None, ):
        super(MambaFuser, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 128, layers, stride=2)

        self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
        self.bn1 = norm_layer(32 * block.expansion)
        self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
        self.bn2 = norm_layer(64 * block.expansion)
        self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
        self.bn21 = norm_layer(32 * block.expansion)
        self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bn31 = norm_layer(64 * block.expansion)
        self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bnf = norm_layer(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.dc_1 = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, dilation=1, padding=1, stride=1),
            norm_layer(64 * block.expansion),
            nn.SiLU(),
        )
        self.dc_3 = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, dilation=1, padding=1, stride=1),
            norm_layer(64 * block.expansion),
            nn.SiLU(),
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, dilation=3, padding=3, stride=1),
            norm_layer(64 * block.expansion),
            nn.SiLU(),
        )
        self.dc_5 = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, dilation=1, padding=1, stride=1),
            norm_layer(64 * block.expansion),
            nn.SiLU(),
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, dilation=3, padding=3, stride=1),
            norm_layer(64 * block.expansion),
            nn.SiLU(),
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, dilation=5, padding=5, stride=1),
            norm_layer(64 * block.expansion),
            nn.SiLU()
        )

        self.dc_out = nn.Conv2d(256 * block.expansion, 64 * block.expansion, 1)

        self.ln1 = nn.LayerNorm(64 * block.expansion)

        self.ssm_blocks = VSSBlock(hidden_dim=64 * block.expansion, drop_path=0.1, norm_layer=nn.LayerNorm,
                                   attn_drop_rate=0.1, d_state=16, size=16, scan_type='hilbert', num_direction=8)

        self.ln2 = nn.LayerNorm(64 * block.expansion)

        self.cab = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 64 * block.expansion // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64 * block.expansion // 4, 64 * block.expansion, 3, 1, 1),
            ChannelAttention(64 * block.expansion)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion), )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def random_mask(self, tensor, mask_prob=0.1):
        """
        Apply random mask to a tensor of shape (b, c, h, w).

        Args:
            tensor (torch.Tensor): Input tensor of shape (b, c, h, w).
            mask_prob (float): Probability of each element being masked. Default is 0.1.

        Returns:
            torch.Tensor: Tensor with random mask applied.
        """

        # Generate a random mask with the same shape as the input tensor
        mask = torch.rand_like(tensor) < mask_prob

        # Apply the mask to the input tensor
        masked_tensor = tensor * mask.float()

        return masked_tensor

    def patch_mask(self, tensor, patch_size=2, mask_prob=0.1):
        """
        Apply patch mask to a 4D image tensor (batch_size, channels, height, width).

        Args:
            image (torch.Tensor): Input image tensor.
            patch_size (int): Size of the patches. Default is 16.
            mask_prob (float): Probability of masking each patch. Default is 0.5.

        Returns:
            torch.Tensor: Image tensor with patch mask applied.
        """
        batch_size, channels, height, width = tensor.shape
        mask = torch.ones_like(tensor)

        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                if torch.rand(1).item() < mask_prob:
                    mask[:, :, i:i + patch_size, j:j + patch_size] = 0

        masked_tensor = tensor * mask
        return masked_tensor

    def random_row_mask(self, tensor, mask_prob=0.1, partial_mask=False, mask_length=4):
        """
        Apply random row mask to a 4D tensor (batch_size, channels, height, width).

        Args:
            tensor (torch.Tensor): Input tensor of shape (b, c, h, w).
            mask_prob (float): Probability of masking each row. Default is 0.1.
            partial_mask (bool): If True, mask only part of the row. Default is False.
            mask_length (int): Length of the partial mask if partial_mask is True. Default is 5.

        Returns:
            torch.Tensor: Tensor with random row mask applied.
        """
        batch_size, channels, height, width = tensor.shape

        # Initialize mask with ones
        mask = torch.ones_like(tensor)

        for i in range(height):
            if torch.rand(1).item() < mask_prob:
                if partial_mask:
                    # Apply partial mask to a row
                    start_idx = torch.randint(0, width - mask_length + 1, (1,)).item()
                    mask[:, :, i, start_idx:start_idx + mask_length] = 0
                else:
                    # Apply full row mask
                    mask[:, :, i, :] = 0

        masked_tensor = tensor * mask
        return masked_tensor

    def forward(self, x):

        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))

        # sv_features = self.random_mask(sv_features)
        # sv_features = self.dynamic_mask(sv_features)

        dc_out_1 = self.dc_1(sv_features)
        dc_out_3 = self.dc_3(sv_features)
        dc_out_5 = self.dc_5(sv_features)
        dc_out = torch.cat([dc_out_1, dc_out_3, dc_out_5, sv_features], dim=1)
        sv_features = self.dc_out(dc_out)

        sv_features = rearrange(sv_features, 'b c h w -> b h w c')

        temp = self.ln1(sv_features)
        temp = self.ssm_blocks(temp)

        sv_features = sv_features + temp

        temp = self.ln2(sv_features)
        temp = rearrange(temp, 'b h w c -> b c h w')
        temp = self.cab(temp)
        temp = rearrange(temp, 'b c h w -> b h w c')
        sv_features = sv_features + temp

        sv_features = rearrange(sv_features, 'b h w c -> b c h w')
        sv_features = self.bn_layer(sv_features)
        return sv_features.contiguous()


class MMAD(nn.Module):
    def __init__(self, model_t, model_s):
        super(MMAD, self).__init__()
        self.net_t = get_model(model_t)
        self.mamba_fuser = MambaFuser(Bottleneck, 3)
        self.net_s = HMD(depths_decoder=model_s['depths_decoder'], scan_type=model_s['scan_type'],
                                num_direction=model_s['num_direction'])

        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        feats_s = self.net_s(self.mamba_fuser(feats_t))
        return feats_t, feats_s


@MODEL.register_module
def mmad(pretrained=False, **kwargs):
    model = MMAD(**kwargs)
    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params

    vmunet = HMD([512, 256, 128, 64], [2, 2, 2, 2])
    bs = 1
    reso = 8
    x = torch.randn(bs, 512, reso, reso).cuda()
    net = vmunet.cuda()
    net.eval()
    y = net(x)
    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))
