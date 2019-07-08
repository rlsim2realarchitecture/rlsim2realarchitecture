import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

from .utils import mini_weight_init, weight_init


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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).reshape(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).reshape(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).reshape(-1, len_v, d_v)  # (n*b) x lv x dv

        output = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).reshape(-1, len_q, n_head * d_v)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class PolNetCNP(nn.Module):
    """
    This class is expectd to be used with ob_ac_env
    """
    def __init__(self, ob_space, ac_space, h_size=512, r_size=512, n_head=8, dropout=0.1, aggregation='sum', use_pe=False):
        super(PolNetCNP, self).__init__()
        self.h_size = h_size
        self.r_size = r_size
        self.aggregation = aggregation
        self.use_pe = use_pe

        if not len(ob_space.shape) == 2:
            raise ValueError("observation's shape must be (timestep, ob_dim)")

        if isinstance(ac_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(ac_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        if self.use_pe:
            position_enc = get_sinusoid_encoding_table(ob_space.shape[0], h_size, padding_idx=0)
            self.register_buffer('position_enc', torch.tensor(position_enc, dtype=torch.float))

        ob_dim = ob_space.shape[1]

        self.encode_layer1 = nn.Sequential(
            nn.Linear(ob_dim * 2, self.h_size),
            nn.ReLU())
        self.encode_layer2 = nn.Sequential(
            nn.Linear(self.h_size, self.r_size),
            nn.ReLU())

        if self.aggregation == 'attention':
            self.cast_key_and_query = nn.Sequential(
                nn.Linear(ob_dim, self.r_size),
                nn.ReLU())
            self.attention = MultiHeadCrossAttention(n_head, self.r_size, self.r_size // n_head, self.r_size // n_head, dropout)

        self.decode_layer = nn.Sequential(
            nn.Linear(self.r_size + ob_dim, self.h_size),
            nn.ReLU(),
            nn.Linear(self.h_size, self.h_size),
            nn.ReLU())

        if not self.discrete:
            self.mean_layer = nn.Linear(self.h_size, ac_space.shape[0])
            self.log_std_param = nn.Parameter(
                torch.randn(ac_space.shape[0])*1e-10 - 1)

            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(self.h_size, vec) for vec in ac_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(self.h_size, ac_space.n)
                self.output_layer.apply(mini_weight_init)

    def forward(self, xs):
        xs_prev = torch.cat([xs[:, 1:], torch.zeros_like(xs[:, -1:])], dim=-2)

        enc_inp = torch.cat([xs_prev, xs], dim=-1)

        hs = self.encode_layer1(enc_inp)
        if self.use_pe:
            hs = hs + self.position_enc
        rs = self.encode_layer2(hs)

        if self.aggregation == 'sum':
            r = torch.sum(rs, dim=1)
        elif self.aggregation == 'mean':
            r = torch.mean(rs, dim=1)
        elif self.aggregation == 'attention':
            query = self.cast_key_and_query(xs[:, 0:1])
            key = self.cast_key_and_query(xs_prev)
            value = rs
            r = self.attention(query, key, value).squeeze(1)
        else:
            raise ValueError('aggregation method must be sum or mean or attention')

        inp = torch.cat([xs[:, 0], r], dim=-1)
        h = self.decode_layer(inp)

        if not self.discrete:
            means = torch.tanh(self.mean_layer(h))
            log_std = self.log_std_param.expand_as(means)
            return means, log_std
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2)
            else:
                return torch.softmax(self.output_layer(decoded), dim=-1)


class VNetCNP(nn.Module):
    def __init__(self, ob_space, h_size=512, r_size=512, n_head=8, dropout=0.1, aggregation='sum', use_pe=False):
        super(VNetCNP, self).__init__()
        self.h_size = h_size
        self.r_size = r_size
        self.aggregation = aggregation
        self.use_pe = use_pe

        if not len(ob_space.shape) == 2:
            raise ValueError("observation's shape must be (timestep, ob_dim)")

        if self.use_pe:
            position_enc = get_sinusoid_encoding_table(ob_space.shape[0], h_size, padding_idx=0)
            self.register_buffer('position_enc', torch.tensor(position_enc, dtype=torch.float))

        ob_dim = ob_space.shape[1]

        self.encode_layer1 = nn.Sequential(
            nn.Linear(ob_dim * 2, self.h_size),
            nn.ReLU())
        self.encode_layer2 = nn.Sequential(
            nn.Linear(self.h_size, self.r_size),
            nn.ReLU())

        if self.aggregation == 'attention':
            self.cast_key_and_query = nn.Sequential(
                nn.Linear(ob_dim, self.r_size),
                nn.ReLU())
            self.attention = MultiHeadCrossAttention(n_head, self.r_size, self.r_size // n_head, self.r_size // n_head, dropout)

        self.decode_layer = nn.Sequential(
            nn.Linear(self.r_size + ob_dim, self.h_size),
            nn.ReLU(),
            nn.Linear(self.h_size, self.h_size),
            nn.ReLU())

        self.output_layer = nn.Linear(self.h_size, 1)
        self.output_layer.apply(mini_weight_init)

    def forward(self, xs):
        xs_prev = torch.cat([xs[:, 1:], torch.zeros_like(xs[:, -1:])], dim=-2)

        enc_inp = torch.cat([xs_prev, xs], dim=-1)

        hs = self.encode_layer1(enc_inp)
        if self.use_pe:
            hs = hs + self.position_enc
        rs = self.encode_layer2(hs)

        if self.aggregation == 'sum':
            r = torch.sum(rs, dim=1)
        elif self.aggregation == 'mean':
            r = torch.mean(rs, dim=1)
        elif self.aggregation == 'attention':
            query = self.cast_key_and_query(xs[:, 0:1])
            key = self.cast_key_and_query(xs_prev)
            value = rs
            r = self.attention(query, key, value).squeeze(1)
        else:
            raise ValueError('aggregation method must be sum or mean or attention')

        inp = torch.cat([xs[:, 0], r], dim=-1)
        h = self.decode_layer(inp)

        out = self.output_layer(h)

        return out
