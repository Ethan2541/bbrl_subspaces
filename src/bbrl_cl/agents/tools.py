# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet


# Creation of distributions for sampling the policies (alphas)
def create_dist(dist_type, n_anchors):
    n_anchors = max(1, n_anchors)
    if dist_type == "flat":
        dist = Dirichlet(torch.ones(n_anchors))
    if dist_type == "peaked":
        dist = Dirichlet(torch.Tensor([1.] * (n_anchors-1) + [n_anchors ** 2]))
    elif dist_type == "categorical":
        dist = Categorical(torch.ones(n_anchors))
    elif dist_type == "last_anchor":
        dist = Categorical(torch.Tensor([0] * (n_anchors-1) + [1]))
    return dist



# Group of policies (networks)
class LinearSubspace(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias = True, same_init = False, freeze_anchors = True):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.freeze_anchors = freeze_anchors

        # Same weights for every anchor
        if same_init:
            anchor = nn.Linear(in_channels,out_channels,bias = self.is_bias)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        # Random weights
        else:
            anchors = [nn.Linear(in_channels, out_channels, bias=self.is_bias) for _ in range(n_anchors)]
        
        self.anchors = nn.ModuleList(anchors)


    def forward(self, x, alpha):
        # print("---anchor:",max(x.abs().max() for x in self.anchors.parameters()))
        # check = (not torch.is_grad_enabled()) and (alpha[0].max() == 1.)

        # Output of each policy
        xs = [anchor(x) for anchor in self.anchors]
        
        # if check:
        #    copy_xs = xs
        #    argmax = alpha[0].argmax()
        xs = torch.stack(xs, dim=-1)

        # Weighted linear combination of these outputs
        alpha = torch.stack([alpha] * self.out_channels, dim=-2)
        xs = (xs * alpha).sum(-1)
        # if check:
        #    print("sanity check:",(copy_xs[argmax] - xs).sum().item())
        return xs


    def add_anchor(self, alpha=None):
        if self.freeze_anchors:
            for param in self.parameters():
                param.requires_grad = False

        # Midpoint by default
        if alpha is None:
            alpha = torch.ones((self.n_anchors,)) / self.n_anchors

        # Add a new policy (anchor) to the existing set of neural networks
        # The weights of the new anchor are a linear combination of the existing weights
        new_anchor = nn.Linear(self.in_channels, self.out_channels, bias=self.is_bias)
        new_weight = torch.stack([a * anchor.weight.data for a,anchor in zip(alpha, self.anchors)], dim=0).sum(0)
        new_anchor.weight.data.copy_(new_weight)

        if self.is_bias:
            new_bias = torch.stack([a * anchor.bias.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
            new_anchor.bias.data.copy_(new_bias)
        
        self.anchors.append(new_anchor)
        self.n_anchors +=1



class Sequential(nn.Sequential):
    def forward(self, input, alpha):
        for module in self:
            input = module(input,alpha) if isinstance(module,LinearSubspace) else module(input)
        return input