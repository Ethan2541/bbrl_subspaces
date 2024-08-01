# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from .utils import *


class AlphaAgent(SubspaceAgent):
    """ An agent to sample a distribution on the current and former subspaces to weight multiple anchors / policies.

    Parameters
    ----------
    n_initial_anchors: number of policies that define the initial subspace
    dist_type: distribution used to sample the weights
    refresh_rate: proportion of the distributions that should be changed
    resampling_policy: whether a new policy should be sampled to update the actor
    repeat_alpha: number of consecutive steps a sampled alphas vector should be used
    """

    def __init__(self, n_initial_anchors, dist_type="flat", refresh_rate=1., resampling_policy=True, repeat_alpha=1000, **kwargs):
        super().__init__()
        self.n_anchors = n_initial_anchors
        self.dist_type = dist_type
        self.refresh_rate = refresh_rate
        self.resampling_policy = resampling_policy
        self.repeat_alpha = repeat_alpha

        self.dist = create_dist(self.dist_type, self.n_anchors)
        # For the initial subspace, we still consider a "former" subspace that has one less anchor
        # Without it, the performances of the overall subspace decrease drastically
        self.dist2 = create_dist("flat", self.n_anchors)

        self.best_alpha = None
        self.best_alphas = torch.Tensor([])

        self.id = nn.Parameter(torch.randn(1,1))


    # Reward tracking: used to compute the cumulated reward for a given sampled policy. It is reset each time a new vector alphas is sampled
    # When it is too low (below a specific threshold), a new policy is sampled
    def track_reward(self, t=None):
        if t is not None:
            if t == 0:
                reward = self.get(("env/reward", t))
                self.set(("tracking_reward", t), reward)
            elif t > 0:
                r = self.get(("env/reward", t))
                old_tracking_reward = self.get(("tracking_reward", t-1))
                refresh_timestep = ((self.get(("env/timestep", t-1)) % self.repeat_alpha) == 0).float()
                tracking_reward = r + old_tracking_reward * (1 - refresh_timestep)
                self.set(("tracking_reward", t),tracking_reward)


    def forward(self, t=None, force_random=False, policy_update=False, alphas=None, mute_alpha=False, **kwargs):
        if mute_alpha:
            pass
        self.track_reward(t)

        if alphas is not None:
            self.set(("alphas", t), alphas)

        # Get the weights with the best performances outside of training
        elif (not self.training) and (not force_random):
            B = self.workspace.batch_size()
            alphas = self.best_alpha.unsqueeze(0).repeat(B,1)
            self.set(("alphas", t), alphas)

        elif t is not None:
            B = self.workspace.batch_size()
            # Sampling in the new subspace and the former subspace
            # Without using the former subspace, the chances to get a good policy are very low
            alphas1 = self.dist.sample(torch.Size([B//2]))
            alphas2 = self.dist2.sample(torch.Size([B - B//2]))

            # Padding the former subspace sampled distribution
            if alphas2.shape[-1] < alphas1.shape[-1]:
                alphas2 = torch.cat([alphas2, torch.zeros(*alphas2.shape[:-1], 1)], dim=-1)
            alphas = torch.cat([alphas1, alphas2], dim=0)

            if isinstance(self.dist, Categorical):
                alphas = F.one_hot(alphas, num_classes=self.n_anchors).float()

            # When an episode starts, a new distribution must be sampled
            # Otherwise, the previous distribution can be used {repeat_alpha} times
            if t > 0 and self.repeat_alpha > 1:
                done = self.get(("env/done", t)).float().unsqueeze(-1)
                refresh_timestep = ((self.get(("env/timestep", t)) % self.repeat_alpha) == 0).float().unsqueeze(-1)
                # Workspace examples where the distribution should be refreshed
                refresh = torch.max(done, refresh_timestep)
                # Refresh the distributions whose cumulated rewards are not in the top k
                if (refresh.sum() > 0) and (self.refresh_rate < 1.):
                    alphas_cumulated_reward = self.get(("tracking_reward", t))
                    k = max(int(len(alphas_cumulated_reward) * (1 - self.refresh_rate)) - 1, 0)
                    threshold = sorted(alphas_cumulated_reward, reverse=True)[k]
                    refresh_condition = (alphas_cumulated_reward < threshold).float().unsqueeze(-1)
                    refresh *= refresh_condition
                alphas_old = self.get(("alphas", t-1))
                alphas = alphas * refresh + alphas_old * (1 - refresh)
            self.set(("alphas", t), alphas)
        
        # Either use the stored distributions or a new resampled policy for the actor loss
        elif policy_update:
            if self.resampling_policy:
                T = self.workspace.time_size()
                B = self.workspace.batch_size()
                alphas = self.dist.sample(torch.Size([T,B]))
                if isinstance(self.dist,Categorical):
                    alphas = F.one_hot(alphas, num_classes=self.n_anchors).float()
                self.set("alphas_policy_update", alphas)
            else:
                self.set("alphas_policy_update", self.get("alphas"))


    def set_best_alpha(self, alpha=None, logger=None, **kwargs):
        # The last policy added as an anchor is the best, if alpha is None
        if alpha is None:
            alpha = torch.Tensor([0.] * (self.n_anchors - 1) + [1.])

        self.best_alpha = alpha
        self.best_alphas = torch.cat([self.best_alphas, alpha.unsqueeze(0)], dim=0)

        if logger is not None:
            logger = logger.get_logger(type(self).__name__ + str("/"))
            if alpha is None:
                logger.message("Set best_alpha = None")
            else:
                logger.message("Set best_alpha = " + str(list(map(lambda x: round(x,2), alpha.tolist()))))
    

    def add_anchor(self, logger=None, **kwargs):
        self.n_anchors += 1
        self.best_alpha = torch.cat([self.best_alpha, torch.zeros(1)], dim=-1)
        self.best_alphas = torch.cat([self.best_alphas, torch.zeros(self.best_alphas.shape[0], 1)], dim=-1)
        self.dist = create_dist(self.dist_type, self.n_anchors)
        self.dist2 = create_dist("flat", self.n_anchors - 1)
        if logger is not None:
            logger = logger.get_logger(type(self).__name__ + str("/"))
            logger.message("Increasing alpha size to " + str(self.n_anchors))


    def remove_anchor(self, logger = None,**kwargs):
        self.n_anchors -= 1
        self.best_alpha = self.best_alpha[:-1]
        self.best_alphas = self.best_alphas[:,:-1]
        self.dist = create_dist(self.dist_type, self.n_anchors)
        self.dist2 = create_dist("flat", self.n_anchors-1)
        if not logger is None:
            logger = logger.get_logger(type(self).__name__ + str("/"))
            logger.message("Decreasing alpha size to " + str(self.n_anchors))



class SubspaceAction(SubspaceAgent):
    """ An agent used to act based on multiple policies, given their respective weights.

    Parameters
    ----------
    n_initial_anchors: number of policies that define the initial subspace
    input_dimension
    output_dimension
    hidden_size: size of the networks' hidden layers
    start_steps: number of steps before the beginning of the actual training
    input_name: name of the variable with the observations
    only_head: the model only has the last layer of the usual networks
    """

    def __init__(self, n_initial_anchors, input_dimension, output_dimension, hidden_size, start_steps=0, input_name="env/env_obs", only_head=False, **kwargs):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.n_anchors = n_initial_anchors
        self.input_size = input_dimension
        self.output_dimension = output_dimension
        self.hs = hidden_size
        
        # Creation of the n_anchors networks with two hidden layers each
        # LeakyReLU -> page 91
        if only_head:
            self.model = Sequential(LinearSubspace(self.n_anchors, self.hs, self.output_dimension * 2)) 
        else:
            self.model = Sequential(
                LinearSubspace(self.n_anchors, self.input_size, self.hs),
                nn.LeakyReLU(negative_slope=0.2),
                LinearSubspace(self.n_anchors, self.hs, self.hs),
                nn.LeakyReLU(negative_slope=0.2),
                LinearSubspace(self.n_anchors, self.hs, self.hs),
                nn.LeakyReLU(negative_slope=0.2),
                LinearSubspace(self.n_anchors, self.hs, 2*self.output_dimension),
            )


    def forward(self, t=None, policy_update=False, predict_proba=False, **kwargs):
        # Outside of training, the output is the linear combination of the anchors' outputs, given their respective weights alphas
        if not self.training:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas", t))
            mu, _ = self.model(x, alphas).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)

        # When not evaluated, the logarithmic probabilities are unnecessary
        elif not predict_proba:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas", t))
            # Before threshold, action is random
            if self.counter <= self.start_steps:
                action = torch.rand(x.shape[0], self.output_dimension)*2 - 1
            else:
                mu, log_std = self.model(x, alphas).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1

        else:
            input = self.get(self.iname)
            # Choosing the sampled policy based on the situation
            if policy_update:
                alphas = self.get("alphas_policy_update")
            else:
                alphas = self.get("alphas")

            mu, log_std = self.model(input,alphas).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))).sum(-1, keepdim=True)
            log_prob -= (2*np.log(2) - action - F.softplus(-2*action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)


    def add_anchor(self, alpha=None, logger=None, **kwargs):
        i = 0
        alphas = [alpha] * (self.hs + 2)
        if logger is not None:
            logger = logger.get_logger(type(self).__name__+str("/"))
            if alpha is None:
                logger.message("Adding one anchor with alpha = None")
            else:
                logger.message("Adding one anchor with alpha = " + str(list(map(lambda x: round(x,2), alpha.tolist()))))

        # Adding an anchor to each subspace layer, with the proper weight alpha
        for module in self.model:
            if isinstance(module, LinearSubspace):
                module.add_anchor(alphas[i])
                # ### Sanity check
                # if i == 0:
                #     for j,anchor in enumerate(module.anchors):
                #         print("--- anchor",j,":",anchor.weight[0].data[:4])
                i += 1
        self.n_anchors += 1


    def remove_anchor(self, logger = None, **kwargs):
        if logger is not None:
            logger = logger.get_logger(type(self).__name__ + str("/"))
            logger.message("Removing last anchor")
        for module in self.model:
            if isinstance(module, LinearSubspace):
                module.anchors = module.anchors[:-1]
                module.n_anchors -= 1
        self.n_anchors -= 1


    # This function is inspired by what is used for the computation of the Line of Policies' penalty term 
    def cosine_similarities(self, **kwargs):
        n = 0
        cosine_similarities = {}
        # The cosine similarities are the respective mean of each network layer's cosine similarities
        for subspace in self.model:
            if isinstance(subspace, LinearSubspace):
                subspace_cosine_similarities = subspace.cosine_similarities()
                for key, value in subspace_cosine_similarities.items():
                    cosine_similarities[key] = cosine_similarities.get(key, 0) + value
                # Bias + Weight
                n += 2
        return {key: similarity/n for key, similarity in cosine_similarities.items()}
    

    def get_subspace_anchors(self, **kwargs):
        anchors = {}
        for anchor_id in range(self.n_anchors):
            layers = []
            for module in self.model:
                if isinstance(module, LinearSubspace):
                    layers.append(copy.deepcopy(module.anchors[anchor_id]))
                else:
                    layers.append(copy.deepcopy(module))
            anchors[anchor_id] = Sequential(*layers)
        return anchors
    

    def euclidean_distances(self, **kwargs):
        euclidean_distances = {}
        anchors = self.get_subspace_anchors()
        for i in range(self.n_anchors):
            for j in range(i+1, self.n_anchors):
                policy_i = torch.nn.utils.parameters_to_vector(anchors[i].parameters())
                policy_j = torch.nn.utils.parameters_to_vector(anchors[j].parameters())
                euclidean_distances[f"π{i+1}, π{j+1}"] = torch.norm(policy_i - policy_j, p=2)
        return euclidean_distances
    

    def subspace_area(self, **kwargs):
        anchors = self.get_subspace_anchors()
        if len(anchors) != 3:
            return None
        with torch.no_grad():
            anchors_euclidean_distances = self.euclidean_distances()
        x, y, z = anchors_euclidean_distances["π1, π3"].item(), anchors_euclidean_distances["π2, π3"].item(), anchors_euclidean_distances["π1, π2"].item()
        d = (x + y + z) / 2
        subspace_area = np.sqrt(d*(d-x)*(d-y)*(d-z))
        return subspace_area
    

    def get_similarities(self, **kwargs):
        with torch.no_grad():
            similarities = "Similarities of the anchors:"
            for key, similarity in self.cosine_similarities().items():
                similarities += f"\ncos({key}) = {round(similarity.item(), 2)}"
            for key, distance in self.euclidean_distances().items():
                similarities += f"\nL2({key}) = {round(distance.item(), 2)}"
            return similarities
    


class AlphaCritic(SubspaceAgent):
    """ An agent used as a critic for the SAC algorithm.

    Parameters
    ----------
    n_anchors: number of policies that define the current subspace
    obs_dimension
    action_dimension
    hidden_size: size of the networks' hidden layers
    input_name: name of the variable with the observations
    output_name: name of the variable with the Q values
    """
    def __init__(self, n_anchors, obs_dimension, action_dimension, hidden_size, input_name="env/env_obs", output_name="q_values", **kwargs):
        super().__init__()
        self.iname = input_name
        self.n_anchors = n_anchors
        self.obs_dimension = obs_dimension
        self.action_dimension= action_dimension
        self.input_size = self.obs_dimension + self.action_dimension + self.n_anchors
        self.hs = hidden_size
        self.output_name = output_name

        # The critic is used to evaluate the future averaged return on (s,a) pairs
        # Adding the sampled distribution allows to consider an infinity of policies
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs, self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs, self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs, self.hs),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hs, 1),
        )


    def forward(self, policy_update=False, **kwargs):
        input = self.get(self.iname).detach()
        action = self.get(("action"))
        # Choosing the appropriate sampled policy
        if policy_update:
            alphas = self.get("alphas_policy_update")
        else:
            alphas = self.get("alphas")

        # Filling to match the dimension (number of anchors)
        if alphas.shape[-1] < self.n_anchors:
            alphas = torch.cat([alphas, torch.zeros(*alphas.shape[:-1], self.n_anchors - alphas.shape[-1])], dim=-1)

        input = torch.cat([input, action, alphas], dim=-1)
        critic = self.model(input).squeeze(-1)
        self.set(f"{self.name}/{self.output_name}", critic)


    def add_anchor(self, n_anchors=None, logger=None, **kwargs):
        self.__init__(self.n_anchors if n_anchors is None else n_anchors, self.obs_dimension, self.action_dimension, self.hs, input_name=self.iname, output_name=self.output_name)
        if logger is not None:
            logger = logger.get_logger(type(self).__name__ + str("/"))
            logger.message("Setting input size to " + str(self.input_size) + " and reinitializing network")




from bbrl_algos.models.stochastic_actors import SquashedGaussianActor
from torch.nn import CosineSimilarity

class IntuitiveSubspaceAction(SubspaceAgent):
    """ An intuitive agent used to act based on multiple policies, given their respective weights.

    Parameters
    ----------
    n_initial_anchors: number of policies that define the initial subspace
    input_dimension
    output_dimension
    hidden_size: size of the networks' hidden layers
    start_steps: number of steps before the beginning of the actual training
    input_name: name of the variable with the observations
    only_head: the model only has the last layer of the usual networks
    """

    def __init__(self, n_initial_anchors, input_dimension, output_dimension, hidden_size, start_steps=0, input_name="env/env_obs", **kwargs):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.n_anchors = n_initial_anchors
        self.input_size = input_dimension
        self.output_dimension = output_dimension
        self.hs = [hidden_size] * 2
        self.anchors = [SquashedGaussianActor(self.input_size, self.hs, self.output_dimension) for _ in range(self.n_anchors)]



    def forward(self, t=None, policy_update=False, predict_proba=False, **kwargs):
        # Outside of training, the output is the linear combination of the anchors' outputs, given their respective weights alphas
        if not self.training:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas", t))
            mu, _ = self.model(x, alphas).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)

        else:
            x = self.get((self.iname, t))
            alphas = self.get(("alphas", t))
            # Before threshold, action is random
            if self.counter <= self.start_steps:
                action = torch.rand(x.shape[0], self.output_dimension)*2 - 1
            else:
                # In SAC, predict_proba is used conversely with stochastic
                xs = [anchor.forward(t, stochastic=predict_proba) for anchor in self.anchors]
                xs = torch.stack(xs, dim=-1)
                alpha = torch.stack([alpha] * self.out_channels, dim=-2)
                action = (xs * alpha).sum(-1)
            self.set(("action", t), action)
            self.counter += 1


    def add_anchor(self, alpha=None, logger=None, **kwargs):
        i = 0
        alphas = [alpha] * (self.hs + 2)
        if logger is not None:
            logger = logger.get_logger(type(self).__name__+str("/"))
            if alpha is None:
                logger.message("Adding one anchor with alpha = None")
            else:
                logger.message("Adding one anchor with alpha = " + str(list(map(lambda x: round(x,2), alpha.tolist()))))

        # Adding an anchor to each subspace layer, with the proper weight alpha
        self.anchors.append(SquashedGaussianActor(self.input_size, self.hs, self.output_dimension))
        self.n_anchors += 1


    def remove_anchor(self, logger = None, **kwargs):
        if logger is not None:
            logger = logger.get_logger(type(self).__name__ + str("/"))
            logger.message("Removing last anchor")
        
        del self.anchors[-1]
        self.n_anchors -= 1


    def cosine_similarities(self, **kwargs):
        n = 0
        cosine_similarities = {}
        cosine_similarity_function = CosineSimilarity()
        anchors = self.get_subspace_anchors()
        # The cosine similarities are the respective mean of each network layer's cosine similarities
        for i in range(self.n_anchors):
            for j in range(i+1, self.n_anchors):
                policy_i = torch.nn.utils.parameters_to_vector(anchors[i].parameters())
                policy_j = torch.nn.utils.parameters_to_vector(anchors[j].parameters())
                cosine_similarities[f"π{i+1}, π{j+1}"] = cosine_similarity_function(policy_i, policy_j)
        return {key: similarity/n for key, similarity in cosine_similarities.items()}
    

    def get_subspace_anchors(self, **kwargs):
        return self.anchors
    

    def euclidean_distances(self, **kwargs):
        euclidean_distances = {}
        anchors = self.get_subspace_anchors()
        for i in range(self.n_anchors):
            for j in range(i+1, self.n_anchors):
                policy_i = torch.nn.utils.parameters_to_vector(anchors[i].parameters())
                policy_j = torch.nn.utils.parameters_to_vector(anchors[j].parameters())
                euclidean_distances[f"π{i+1}, π{j+1}"] = torch.norm(policy_i - policy_j, p=2)
        return euclidean_distances
    

    def subspace_area(self, **kwargs):
        with torch.no_grad():
            anchors = self.get_subspace_anchors()
            if len(anchors) != 3:
                return None
            anchors_euclidean_distances = self.euclidean_distances()
            x, y, z = anchors_euclidean_distances["π1, π3"].item(), anchors_euclidean_distances["π2, π3"].item(), anchors_euclidean_distances["π1, π2"].item()
            d = (x + y + z) / 2
            subspace_area = np.sqrt(d*(d-x)*(d-y)*(d-z))
            return subspace_area
    

    def get_similarities(self, **kwargs):
        with torch.no_grad():
            similarities = "Similarities of the anchors:"
            for key, similarity in self.cosine_similarities().items():
                similarities += f"\ncos({key}) = {round(similarity.item(), 2)}"
            for key, distance in self.euclidean_distances().items():
                similarities += f"\nL2({key}) = {round(distance.item(), 2)}"
            return similarities