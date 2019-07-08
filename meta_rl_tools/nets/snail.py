import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

from machina.utils import get_device

from .utils import mini_weight_init, weight_init
from .blocks import AttentionBlock, TCBlock


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_pos_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+ i

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table


class PolNetSNAILConstant(nn.Module):
    def __init__(self, ob_space, ac_space, timestep, num_channels, num_keys=32, num_tc_fils=16, no_attention=False, use_pe=False):
        super(PolNetSNAILConstant, self).__init__()
        self.timestep = timestep
        self.no_attention = no_attention
        self.use_pe = use_pe

        if isinstance(ac_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(ac_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        num_filters = int(math.ceil(math.log(timestep)))
        self.fc1 = nn.Linear(ob_space.shape[-1], num_channels)
        self.fc2 = nn.Linear(num_channels, num_channels)
        if not self.no_attention:
            if self.use_pe:
                position_enc = get_sinusoid_encoding_table(ob_space.shape[0], num_channels, padding_idx=0)
                self.register_buffer('position_enc1', torch.tensor(position_enc, dtype=torch.float))
            self.attention1 = AttentionBlock(num_channels, num_keys, num_keys)
            num_channels += num_keys
        self.tc1 = TCBlock(num_channels, timestep, num_tc_fils)
        num_channels += num_filters * num_tc_fils
        self.tc2 = TCBlock(num_channels, timestep, num_tc_fils)
        num_channels += num_filters * num_tc_fils
        if not self.no_attention:
            if self.use_pe:
                position_enc = get_sinusoid_encoding_table(ob_space.shape[0], num_channels, padding_idx=0)
                self.register_buffer('position_enc2', torch.tensor(position_enc, dtype=torch.float))
            self.attention2 = AttentionBlock(num_channels, num_keys, num_keys)
            num_channels += num_keys

        if not self.discrete:
            self.mean_layer = nn.Linear(num_channels, ac_space.shape[0])
            self.log_std_param = nn.Parameter(
                torch.randn(ac_space.shape[0])*1e-10 - 1)

            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(num_channels, vec) for vec in ac_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(num_channels, ac_space.n)
                self.output_layer.apply(mini_weight_init)

    def forward(self, xs):
        xs = torch.flip(xs, dims=[1])
        hs = torch.tanh(self.fc1(xs))
        hs = torch.tanh(self.fc2(hs))
        if not self.no_attention:
            if self.use_pe:
                hs = hs + self.position_enc1
            hs = self.attention1(hs)
        hs = self.tc1(hs)
        hs = self.tc2(hs)
        if not self.no_attention:
            if self.use_pe:
                hs = hs + self.position_enc2
            hs = self.attention2(hs)

        h = hs[:, -1]
        if not self.discrete:
            mean = torch.tanh(self.mean_layer(h))
            log_std = self.log_std_param.expand_as(mean)
            return mean, log_std
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2)
            else:
                return torch.softmax(self.output_layer(h), dim=-1)


class VNetSNAILConstant(nn.Module):
    def __init__(self, ob_space, timestep, num_channels, num_keys=32, num_tc_fils=16, no_attention=False, use_pe=False):
        super(VNetSNAILConstant, self).__init__()
        self.timestep = timestep
        self.no_attention = no_attention
        self.use_pe = use_pe

        num_filters = int(math.ceil(math.log(timestep)))
        self.fc1 = nn.Linear(ob_space.shape[-1], num_channels)
        self.fc2 = nn.Linear(num_channels, num_channels)
        if not self.no_attention:
            if self.use_pe:
                position_enc = get_sinusoid_encoding_table(ob_space.shape[0], num_channels, padding_idx=0)
                self.register_buffer('position_enc1', torch.tensor(position_enc, dtype=torch.float))
            self.attention1 = AttentionBlock(num_channels, num_keys, num_keys)
            num_channels += num_keys
        self.tc1 = TCBlock(num_channels, timestep, num_tc_fils)
        num_channels += num_filters * num_tc_fils
        self.tc2 = TCBlock(num_channels, timestep, num_tc_fils)
        num_channels += num_filters * num_tc_fils
        if not self.no_attention:
            if self.use_pe:
                position_enc = get_sinusoid_encoding_table(ob_space.shape[0], num_channels, padding_idx=0)
                self.register_buffer('position_enc2', torch.tensor(position_enc, dtype=torch.float))
            self.attention2 = AttentionBlock(num_channels, num_keys, num_keys)
            num_channels += num_keys

        self.output_layer = nn.Linear(num_channels, 1)

        self.output_layer.apply(mini_weight_init)

    def forward(self, xs):
        xs = torch.flip(xs, dims=[1])
        hs = torch.tanh(self.fc1(xs))
        hs = torch.tanh(self.fc2(hs))
        if not self.no_attention:
            if self.use_pe:
                hs = hs + self.position_enc1
            hs = self.attention1(hs)
        hs = self.tc1(hs)
        hs = self.tc2(hs)
        if not self.no_attention:
            if self.use_pe:
                hs = hs + self.position_enc2
            hs = self.attention2(hs)

        return self.output_layer(hs[:, -1])

