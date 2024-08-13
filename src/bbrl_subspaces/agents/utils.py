# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import os
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from functools import partial

from bbrl.agents import Agents
from bbrl.agents.agent import Agent
from bbrl import get_class
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

assets_path = os.getcwd() + "/../../assets/"


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



def get_env_agents(cfg, *, autoreset=True, include_last_state=True, alpha_search=False):
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    # Returns an additional environment to estimate the best sampled distribution of a subspace if alpha_search is True
    
    if "xml_file" in cfg.keys() and cfg.xml_file is not None:
        xml_file = assets_path + cfg.xml_file
        print("loading:", xml_file)
    else:
        xml_file = None

    if "wrappers" in cfg.keys():
        print("using wrappers:", cfg.wrappers)
        # wrappers_name_list = cfg.gym_env.wrappers.split(',')
        wrappers_list = []
        wr = get_class(cfg.wrappers)
        # for i in range(len(wrappers_name_list)):
        wrappers_list.append(wr)
        wrappers = wrappers_list
        print(wrappers)
    else:
        wrappers = []

    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(
            make_env, cfg.env_name, autoreset=autoreset, wrappers=wrappers, **cfg.kwargs
        ),
        cfg.n_envs,
        include_last_state=include_last_state,
        seed=cfg.seed.train,
    )

    # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.env_name, wrappers=wrappers, **cfg.kwargs),
        cfg.nb_evals,
        include_last_state=include_last_state,
        seed=cfg.seed.eval,
    )

    # Test environment to estimate the best sampled distribution of a subspace
    if alpha_search:
        alpha_env_agent = ParallelGymAgent(
            partial(make_env, cfg.env_name, wrappers=wrappers, **cfg.kwargs),
            cfg.n_rollouts,
            include_last_state=include_last_state,
            seed=cfg.seed.alpha_search,
        )

    if alpha_search:
        return train_env_agent, eval_env_agent, alpha_env_agent
    else:
        return train_env_agent, eval_env_agent



class SubspaceAgents(Agents):
    def add_anchor(self, **kwargs):
        for agent in self:
            agent.add_anchor(**kwargs)
    def remove_anchor(self, **kwargs):
        for agent in self:
            agent.remove_anchor(**kwargs)
    def set_best_alpha(self, **kwargs):
        for agent in self:
            agent.set_best_alpha(**kwargs)
    def set_task(self, task_id=None, **kwargs):
        for agent in self:
            agent.set_task(task_id)
    def cosine_similarities(self):
        for agent in self:
            cosine_similarities = agent.cosine_similarities()
            if cosine_similarities is not None:
                return cosine_similarities
        return None
    def euclidean_distances(self, **kwargs):
        for agent in self:
            euclidean_distances = agent.euclidean_distances()
            if euclidean_distances is not None:
                return euclidean_distances
        return None
    def subspace_area(self, **kwargs):
        for agent in self:
            subspace_area = agent.subspace_area()
            if subspace_area is not None:
                return subspace_area
        return None
    def get_similarities(self):
        for agent in self:
            similarities = agent.get_similarities()
            if similarities is not None:
                return similarities
        return None



class SubspaceAgent(Agent):
    def add_anchor(self, **kwargs):
        pass
    def remove_anchor(self, **kwargs):
        pass
    def set_best_alpha(self, **kwargs):
        pass
    def set_task(self, task_id=None, **kwargs):
        pass
    def cosine_similarities(self, **kwargs):
        return None
    def euclidean_distances(self, **kwargs):
        return None
    def subspace_area(self, **kwargs):
        return None
    def get_similarities(self, **kwargs):
        return None



# Layer for a group of policies
class LinearSubspace(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias=True, same_init=False, freeze_anchors=True):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.freeze_anchors = freeze_anchors

        # Same weights for every anchor
        if same_init:
            anchor = nn.Linear(in_channels, out_channels, bias=self.is_bias)
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
        new_weight = torch.stack([a * anchor.weight.data for a, anchor in zip(alpha, self.anchors)], dim=0).sum(0)
        new_anchor.weight.data.copy_(new_weight)

        if self.is_bias:
            new_bias = torch.stack([a * anchor.bias.data for a,anchor in zip(alpha, self.anchors)], dim = 0).sum(0)
            new_anchor.bias.data.copy_(new_bias)
        
        self.anchors.append(new_anchor)
        self.n_anchors +=1


    # Anticollapse penalty term as a dictionary
    def cosine_similarities(self):
        cosine_similarities = {}
        # The policies should be pairwise distinct to prevent the subspace from collapsing
        for i in range(self.n_anchors):
            for j in range(i+1, self.n_anchors):
                w1 = self.anchors[i].weight
                w2 = self.anchors[j].weight
                p1 = ((w1 * w2).sum() / max(((w1**2).sum().sqrt() * (w2**2).sum().sqrt()), 1e-8))**2

                b1 = self.anchors[i].bias
                b2 = self.anchors[j].bias
                p2 = ((b1 * b2).sum() / max(((b1**2).sum().sqrt() * (b2**2).sum().sqrt()), 1e-8))**2

                cosine_similarities["θ" + str(i+1) + ", θ" + str(j+1)] = (p1 + p2)
        return cosine_similarities



class Sequential(nn.Sequential):
    def forward(self, input, alpha):
        for module in self:
            input = module(input,alpha) if isinstance(module,LinearSubspace) else module(input)
        return input