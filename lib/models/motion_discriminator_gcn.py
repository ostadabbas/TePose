import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.utils import import_class
from lib.models.ms_gcn import MultiScale_GraphConv as MS_GCN
from lib.models.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from lib.models.mlp import MLP
from lib.models.activation import activation_factory


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size=3,
                 window_stride=1,
                 window_dilation=1,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MotionDiscriminatorGCN(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):
        super(MotionDiscriminatorGCN, self).__init__()

        self.num_point = num_point
        self.in_channels = in_channels

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 64
        c2 = c1 * 2     # 128
        c3 = c2 * 2     # 256

        # r=3 STGC blocks
        self.gcn3d1 = MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True)
        self.residual_1 = TemporalConv(in_channels, c1, kernel_size=residual_kernel_size, stride=1)
        self.act_1 = activation_factory(activation)

        self.gcn3d2 = MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn2 = MS_GCN(num_gcn_scales, c1, c2, A_binary, disentangled_agg=True)
        self.residual_2 = TemporalConv(c1, c2, kernel_size=residual_kernel_size, stride=1)
        self.act_2 = activation_factory(activation)

        self.gcn3d3 = MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn3 = MS_GCN(num_gcn_scales, c2, c3, A_binary, disentangled_agg=True)
        self.residual_3 = TemporalConv(c2, c3, kernel_size=residual_kernel_size, stride=1)
        self.act_3 = activation_factory(activation)

        self.fc = nn.Linear(c3, num_class)
        self.sfm = nn.Softmax(dim=1)
           

    def forward(self, x):
        N, T, _ = x.size()
        x = x.permute(0, 2, 1).contiguous() # N, V * C, T
        x = self.data_bn(x)

        x = x.view(N, self.num_point, self.in_channels, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        res = self.residual_1(x)
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = x.clone() + res
        x = self.act_1(x).clone()

        res = self.residual_2(x)
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = x.clone() + res
        x = self.act_2(x).clone()

        res = self.residual_3(x)
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = x.clone() + res
        x = self.act_3(x).clone()

        out = x
        out_channels = out.size(1)
        out = out.view(N, out_channels, -1)
        out = out.mean(2)   # Global Average Pooling (Spatial+Temporal)

        out = self.sfm(self.fc(out))
        return out[:,0]


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    print('Model total # params:', count_params(model))
