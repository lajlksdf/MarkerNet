import torch
import torch.nn as nn
from einops import rearrange

from network.modules.blocks import b8, Conv2d_BN, ResnetBlock
from .helper import tensor_resize
from .modules.frequency_filter import FrequencyFilter
from .modules.spatial_filter import SpatialFilter
from .modules.transformer_block import GeneralTransformerBlock
from .poolformer import PoolFormerBlock

blocks_dict = {
    "BOTTLENECK": ResnetBlock,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
    "POOL_FORMER_BLOCK": PoolFormerBlock,
}

BN_MOMENTUM = 0.1


class HighResolutionTransformerModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, num_heads, num_window_sizes,
                 num_mlp_ratios, multi_scale_output=True, drop_path=0.0, ):
        """Based on Local-Attention & FFN-DW-BN
        num_heads: the number of head witin each MHSA
        num_window_sizes: the window size for the local self-attention
        num_halo_sizes: the halo size around the local window
            - reference: ``Scaling Local Self-Attention for Parameter Efficient Visual Backbones''
        num_sr_ratios: the spatial reduction ratios of PVT/SRA scheme.
            - reference: ``Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions''
        """
        super(HighResolutionTransformerModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            drop_path,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, num_heads, num_window_sizes,
                         num_mlp_ratios, drop_paths, stride=1, ):
        layers = []
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, num_heads, num_window_sizes, num_mlp_ratios,
                       drop_paths, ):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i, block, num_blocks, num_channels, num_heads, num_window_sizes, num_mlp_ratios,
                    drop_paths=[_ * (2 ** i) for _ in drop_paths], )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        Conv2d_BN(num_inchannels[j], num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        num_outchannels_conv3x3 = num_inchannels[i] if k == i - j - 1 else num_inchannels[j]
                        conv3x3s.append(nn.Sequential(
                            Conv2d_BN(num_inchannels[j], num_inchannels[j], 3, 2, 1, groups=num_inchannels[j]),
                            Conv2d_BN(num_inchannels[j], num_outchannels_conv3x3),
                        ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        y = []
        for i in range(self.num_branches):
            y.append(self.branches[i](x[i]))

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + tensor_resize(self.fuse_layers[i][j](x[j]), size=[height_output, width_output])
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class SpatialFrequencyPyramidTransformer(nn.Module):

    def __init__(self, cfg):
        super(SpatialFrequencyPyramidTransformer, self).__init__()
        self.cfg = cfg
        self.spacial_feats = nn.Sequential(
            SpatialFilter(cfg.IMAGE_SIZE),
            b8(256, nn.GELU, in_channels=12),
            Conv2d_BN(256, 64, 3, 2, 1),
        )

        self.dct_feats = nn.Sequential(
            FrequencyFilter(cfg.IMAGE_SIZE),
            b8(256, nn.GELU, in_channels=12),
            Conv2d_BN(256, 32, 3, 1, 1),
        )

        # stochastic depth
        depth_s2 = cfg.STAGE2.NUM_BLOCKS[0] * cfg.STAGE2.NUM_MODULES
        depth_s3 = cfg.STAGE3.NUM_BLOCKS[0] * cfg.STAGE3.NUM_MODULES
        depth_s4 = cfg.STAGE4.NUM_BLOCKS[0] * cfg.STAGE4.NUM_MODULES
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = cfg.DROP_PATH_RATE
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stage3_cfg = cfg.STAGE3
        self._make_token(self.stage3_cfg)
        num_channels = self.stage3_cfg.NUM_CHANNELS
        self.transition2 = self._make_transition_layer([32, 96], num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels,
                                                           drop_path=dpr[depth_s2: depth_s2 + depth_s3])

        self.stage4_cfg = cfg.STAGE4
        self._make_token(self.stage4_cfg)
        num_channels = self.stage4_cfg.NUM_CHANNELS
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, drop_path=dpr[depth_s2 + depth_s3:],
        )

    def _make_token(self, cfg):
        for i in range(cfg.NUM_BRANCHES):
            h, w = cfg.NUM_RESOLUTIONS[i][0], cfg.NUM_RESOLUTIONS[i][1]
            if self.cfg.image_based:
                pos_token = nn.Parameter(torch.randn(1, cfg.NUM_CHANNELS[i], h, w))
            else:
                pos_token = nn.Parameter(torch.randn(1, self.cfg.NUM_FRAMES, cfg.NUM_CHANNELS[i], h, w))
            setattr(self, f"pos_token{cfg.NUM_BRANCHES}-{i}", pos_token)

    def _fill_token(self, xs, cfg):
        for i in range(cfg.NUM_BRANCHES):
            pos_token = getattr(self, f"pos_token{cfg.NUM_BRANCHES}-{i}")
            if self.cfg.image_based:
                xs[i] = xs[i] + pos_token
            else:
                x = rearrange(xs[i], '(b t) ... -> b t ...', t=self.cfg.NUM_FRAMES)
                xs[i] = rearrange(x + pos_token, 'b t ... -> (b t) ...')

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(Conv2d_BN(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels)
                    conv3x3s.append(Conv2d_BN(inchannels, outchannels, 3, 2, 1))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True, drop_path=0.0):

        modules = []
        for i in range(layer_config.NUM_MODULES):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == layer_config.NUM_MODULES - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionTransformerModule(
                    layer_config.NUM_BRANCHES,
                    blocks_dict[layer_config.BLOCK],
                    layer_config.NUM_BLOCKS,
                    num_inchannels,
                    layer_config.NUM_CHANNELS,
                    layer_config.NUM_HEADS,
                    layer_config.NUM_WINDOW_SIZES,
                    layer_config.NUM_MLP_RATIOS,
                    reset_multi_scale_output,
                    drop_path=drop_path[layer_config.NUM_BLOCKS[0] * i: layer_config.NUM_BLOCKS[0] * (i + 1)],
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x1 = self.dct_feats(x)
        x2 = self.spacial_feats(x)
        x3 = torch.cat([tensor_resize(x1, size=28), x2], dim=1)
        y_list = [x1, x2, x3]

        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        self._fill_token(x_list, self.stage3_cfg)
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        self._fill_token(x_list, self.stage4_cfg)
        y_list = self.stage4(x_list)

        return y_list
