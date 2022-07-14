import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from config import ImageSplicingMaskConfig
from network.hrtransformer import SpatialFrequencyPyramidTransformer
from network.modules.blocks import Conv2d_BN, UpperSample
from network.spatial_ocr_block import PyramidSpatialGather_Module, SpatialOCR_Module


class MaskFeatures(nn.Module):
    def __init__(self, cfg, dim):
        super(MaskFeatures, self).__init__()
        self.cfg = cfg
        self.num_classes = self.cfg.NUM_CLASSES
        self.backbone = SpatialFrequencyPyramidTransformer(cfg)

        group_channel = math.gcd(self.cfg.in_channels, self.cfg.hidden_dim)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, self.cfg.hidden_dim, kernel_size=7, stride=1, padding=3,
                      groups=group_channel),
            nn.BatchNorm2d(self.cfg.hidden_dim),
            nn.ReLU()
        )
        self.ocr_gather_head = PyramidSpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.cfg.hidden_dim, key_channels=self.cfg.hidden_dim // 2,
                                                 out_channels=self.cfg.hidden_dim, scale=1, dropout=0.05, )

        self.mask_head = Conv2d_BN(self.cfg.hidden_dim, dim)
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, self.cfg.hidden_dim, kernel_size=7, stride=1, padding=3,
                      groups=group_channel, ),
            nn.BatchNorm2d(self.cfg.hidden_dim), nn.GELU(),
            nn.Conv2d(self.cfg.hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True, ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.mask_head(feats)
        return out, out_aux


class MaskNet(nn.Module):
    def __init__(self, cfg, dim=256, image_size=56):
        super().__init__()
        self.cfg = cfg
        # features
        self.extra_feats = MaskFeatures(cfg=self.cfg, dim=dim)
        # uppers
        self.upper_samples = nn.Sequential(
            UpperSample(dim, dim // 4, hw=image_size),
            UpperSample(dim // 4, self.cfg.NUM_CLASSES, hw=image_size * 2)
        )
        # to 1 channels
        self.to_img = nn.Sequential(
            Rearrange('(b t) (h w) c -> b t c h w', t=self.cfg.NUM_FRAMES, h=self.cfg.IMAGE_SIZE,
                      w=self.cfg.IMAGE_SIZE),
            nn.Tanh()
        )

        self.to_aux = nn.Sequential(
            Rearrange('(b t) c h w-> b t c h w', t=self.cfg.NUM_FRAMES),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = rearrange(x, 'b t ... -> (b t) ...')
        mask, out_aux = self.extra_feats(x)
        mask = rearrange(mask, 'b c h w -> b (h w) c')
        mask = self.upper_samples(mask)
        mask = (self.to_img(mask) + 1) / 2
        out_aux = self.to_aux(out_aux)
        return mask, out_aux


class MaskImageNet(nn.Module):
    def __init__(self, cfg, dim=256, image_size=56):
        super().__init__()
        self.cfg = cfg
        # features
        self.extra_feats = MaskFeatures(cfg=self.cfg, dim=dim)
        # uppers
        self.upper_samples = nn.Sequential(
            UpperSample(dim, dim // 4, hw=image_size),
            UpperSample(dim // 4, self.cfg.NUM_CLASSES, hw=image_size * 2)
        )
        # to 1 channels
        self.to_img = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=self.cfg.IMAGE_SIZE,
                      w=self.cfg.IMAGE_SIZE),
            nn.Tanh()
        )

        self.to_aux = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        mask, out_aux = self.extra_feats(x)
        mask = rearrange(mask, 'b c h w -> b (h w) c')
        mask = self.upper_samples(mask)
        mask = (self.to_img(mask) + 1) / 2
        out_aux = self.to_aux(out_aux)
        return mask, out_aux


if __name__ == '__main__':
    model = MaskImageNet(cfg=ImageSplicingMaskConfig)
    # img = Image.open('../images/1.jpg')
    # x = helper.torch_resize(img)
    x = torch.randn((4, 3, 224, 224))
    x = model(x)
    print(x)
