# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: Largely reusing the code from the reference VAN implementation
# see https://github.com/Visual-Attention-Network

# import math
from dataclasses import dataclass
from typing import Optional

import nn

# from xformers.components import Activation, build_activation
from xformers.components.feedforward import FeedforwardConfig

from . import register_feedforward

# from xformers.factory.weight_init import trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


@dataclass
class MlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int
    in_features: int
    out_features = (None,)
    act_layer = (nn.GELU,)
    drop = (0.0,)


@register_feedforward("ConvMLP", MlpConfig)
class ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layer_multiplier: int = 1,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_layer_multiplier * in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    #     self.apply(self.init_weights)

    # def init_weights(self, m: nn.Module):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
