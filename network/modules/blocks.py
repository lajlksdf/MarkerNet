import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=2, norm=nn.LayerNorm):
        super(LinearBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels * rate, bias=False),
            nn.GELU(),
            nn.Linear(in_channels * rate, out_channels, bias=False),
            norm(out_channels)
        )

    def forward(self, x):
        return self.fc(x)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.c = torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(b, momentum=0.1)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        return x


def b16(n=256, activation=nn.GELU, in_channels=3):
    return torch.nn.Sequential(
        ResnetBlock(in_channels, n // 8, down=True),
        activation(),
        ResnetBlock(n // 8, n // 4, down=True),
        activation(),
        ResnetBlock(n // 4, n // 2, down=True),
        activation(),
        ResnetBlock(n // 2, n, down=True))


def b8(n, activation, in_channels=3):
    dim = max(n, in_channels)
    return torch.nn.Sequential(
        ResnetBlock(in_channels, dim // 2, down=True),
        activation(),
        ResnetBlock(dim // 2, n, down=True),
        activation()
    )


class UpperSample(nn.Module):

    def __init__(self, in_dim, out_dim, hw):
        super(UpperSample, self).__init__()

        self.up_proj = nn.Sequential(nn.Linear(in_dim, in_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(in_dim),
                                     nn.Hardswish(),
                                     Rearrange('b c (h w) -> b c h w', h=hw, w=hw)
                                     )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Hardswish(),
            Rearrange('b c h w -> b (h w) c', c=out_dim)
        )

        self.to_out = nn.Sequential(nn.Linear(out_dim, out_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(out_dim),
                                    nn.Hardswish(),
                                    Rearrange('b c n -> b n c'),
                                    nn.Dropout()
                                    )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.de_conv(x)
        x = self.to_out(x)
        return x


class ResnetBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, down=False):
        super(ResnetBlock, self).__init__()
        self.down = down
        self.conv1 = Conv2d_BN(inplanes, inplanes * self.expansion)
        self.conv2 = Conv2d_BN(inplanes * self.expansion, inplanes * self.expansion, 3, 1, 1)
        self.conv3 = Conv2d_BN(inplanes * self.expansion, inplanes)
        self.stride = stride
        if down:
            self.up = Conv2d_BN(inplanes, planes, 3, 2, 1)
        else:
            self.up = Conv2d_BN(inplanes, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual

        return self.up(out)


class ResnetBlockUp(nn.Module):

    def __init__(self, inplanes, planes, hw, expansion=2):
        super(ResnetBlockUp, self).__init__()
        self.conv1 = Conv2d_BN(inplanes, inplanes // expansion)
        self.conv2 = Conv2d_BN(inplanes // expansion, inplanes // expansion, 3, 1, 1)
        self.conv3 = Conv2d_BN(inplanes // expansion, inplanes)
        self.up = Conv2d_BN(inplanes, planes)
        self.pool = nn.AdaptiveAvgPool2d((hw, hw))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.up(out)

        return self.pool(out)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.layers(x)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights.expand_as(x)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3, hw=14, padding=1):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=hw, w=hw)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=padding, groups=scale_dim, bias=False),
            nn.MaxPool2d(kernel_size=depth_kernel, stride=1, padding=padding),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=hw, w=hw)
        )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
