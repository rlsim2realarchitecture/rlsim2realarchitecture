import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from .utils import mini_weight_init, weight_init

class PolNet(nn.Module):
    def __init__(self, ob_space, ac_space, h1=200, h2=100, deterministic=False):
        super(PolNet, self).__init__()

        self.deterministic = deterministic

        if isinstance(ac_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(ac_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        self.fc1 = nn.Linear(np.prod(ob_space.shape), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)

        if not self.discrete:
            self.mean_layer = nn.Linear(h2, ac_space.shape[0])
            if not self.deterministic:
                self.log_std_param = nn.Parameter(
                    torch.randn(ac_space.shape[0])*1e-10 - 1)
            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(h2, vec) for vec in ac_space.nvec])
                map(lambda x: x.apply(mini_weight_init), self.output_layers)
            else:
                self.output_layer = nn.Linear(h2, ac_space.n)
                self.output_layer.apply(mini_weight_init)

    def forward(self, ob):
        ob = ob.reshape(ob.size(0), -1)
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        if not self.discrete:
            mean = torch.tanh(self.mean_layer(h))
            log_std = self.log_std_param.expand_as(mean)
            return mean, log_std
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2)
            else:
                return torch.softmax(self.output_layer(h), dim=-1)


class VNet(nn.Module):
    def __init__(self, ob_space, h1=200, h2=100):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(np.prod(ob_space.shape), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.apply(weight_init)

    def forward(self, ob):
        ob = ob.reshape(ob.size(0), -1)
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


class QNet(nn.Module):
    def __init__(self, ob_space, ac_space, h1=300, h2=400):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(np.prod(ob_space.shape), h1)
        self.fc2 = nn.Linear(ac_space.shape[0] + h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        ob = ob.reshape(ob.size(0), -1)
        h = F.relu(self.fc1(ob))
        h = torch.cat([h, ac], dim=-1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)

