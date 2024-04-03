# ------------------------------------------------------------
# Copyright (c) VCU, Nanjing University.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by Qing-Long Zhang
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math


class New_GAU(nn.Module):
    def __init__(self, dim = 96, expansion_factor=2., qkv_bias=False,
                sr_ratio=[8, 4, 2, 1], num_heads=2):
        super().__init__()

        hidden_dim = int(expansion_factor * dim)
        query_key_dim = dim // 2
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim)
        )

        self.proj = nn.Linear(dim, dim)
        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sig = self.sigmoid = nn.Sigmoid()

        self.sr_ratio = sr_ratio

        self.kv = nn.Linear(dim, dim, bias=qkv_bias)
        self.gamma = nn.Parameter(torch.ones(2, dim))
        self.beta = nn.Parameter(torch.zeros(2, dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        normed_x = self.norm(x)  # (bs,seq_len,dim)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)  # (bs,seq_len,seq_len)

        Z = self.kv(normed_x)
        print(f"Z: {Z.size()}")
        QK = einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        print(f"q: {q.size()}")
        print(f"v: {v.size()}")

        attn = (q @ k.transpose(-2, -1)) / N
        print(f"attn: {attn.size()}")
        A = F.relu(attn) ** 2
        V = einsum('b i j, b j d -> b i d', A, v)
        print(f"V: {V.size()}")
        value = V * gate
        print(f"value: {value.size()}")
        out = self.proj(value)
        output = out + x

        return output


class Block(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = New_GAU(dim, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # pre_norm
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class Stem(nn.Module):

    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True, alpha=1.67):
        super(Stem, self).__init__()
        self.alpha = alpha
        W = in_ch * alpha

        num_filters_3x3 = int(W * 0.333)
        num_filters_5x5 = int(W * 0.667)
        num_filters_7x7 = int(W)

        # Convolution layers
        self.conv_3x3 = nn.Conv2d(in_ch, num_filters_3x3, kernel_size=3, padding=1, stride=2)
        self.conv_5x5 = nn.Conv2d(in_ch, num_filters_5x5, kernel_size=5, padding=2, stride=2)
        self.conv_7x7 = nn.Conv2d(in_ch, num_filters_7x7, kernel_size=7, padding=3, stride=2)

        # Batch normalization layers
        self.bn_3x3 = nn.BatchNorm2d(num_filters_3x3)
        self.bn_5x5 = nn.BatchNorm2d(num_filters_5x5)
        self.bn_7x7 = nn.BatchNorm2d(num_filters_7x7)

        # Ensure the output has the expected number of channels
        self.adjust_channels = nn.Conv2d(num_filters_3x3 + num_filters_5x5 + num_filters_7x7,
                                          out_ch, kernel_size=1, stride=1)
        self.patch_size = patch_size
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):

        B, C, H, W = x.shape
        x_3x3 = F.relu(self.bn_3x3(self.conv_3x3(x)))
        x_5x5 = F.relu(self.bn_5x5(self.conv_5x5(x)))
        x_7x7 = F.relu(self.bn_7x7(self.conv_7x7(x)))

        x_concat = torch.cat([x_3x3, x_5x5, x_7x7], dim=1)
        x_out = self.adjust_channels(x_concat)

        if self.with_pos:
            x_out = self.pos(x_out)

        x_out = x_out.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x_out = self.norm(x_out)
        H, W = H // 2, W // 2
        return x_out, (H, W)

class ConvStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv2d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class Django_Class_Net(nn.Module):
    def __init__(self, in_chans=3, num_classes=4, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i])
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)  # final norm layer
        # classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        B, _, H, W = x.shape
        x, (H, W) = self.stem(x)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 2
        x, (H, W) = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 3
        x, (H, W) = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 4
        x, (H, W) = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.avg_pool(x).flatten(1)
        x = self.head(x)
        return x



if __name__ == "__main__":
    model = Stem(in_ch=3, out_ch=64)
    dummy_input = torch.randn(4, 3, 256, 512)
    output = model(dummy_input)
    print(output.size())