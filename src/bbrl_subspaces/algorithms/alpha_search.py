# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import torch
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from ternary.helpers import simplex_iterator
from torch.distributions.dirichlet import Dirichlet

from bbrl_subspaces.agents.utils import LinearSubspace


# After removing an anchor, the last added anchor should still be frozen (optimal subspace reached)
def remove_anchor(model):
    model.agents[1].n_anchors -= 1
    for nn_module in model[1].model:
        if isinstance(nn_module,LinearSubspace):
            nn_module.anchors = nn_module.anchors[:-1]
            nn_module.n_anchors -= 1
    return model



class AlphaSearch:
    def __init__(self, n_estimations, n_rollouts, seed):
        self.n_estimations = n_estimations
        self.n_rollouts = n_rollouts
        self.seed = seed

    def run(self, action_agent, critic_agent, logger, info={}):
        torch.manual_seed(self.seed)
        logger = logger.get_logger(type(self).__name__ + "/")
        n_anchors = action_agent[0].n_anchors
        
        # Subspace of policies (several anchors)
        if (n_anchors > 1):
            replay_buffer = info["replay_buffer"]
            n_rollouts = self.n_rollouts

            # Estimating best alphas in the subspace using K-shot adaptation with K the number of rollouts
            alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([n_rollouts]))
            alphas = torch.stack([alphas for _ in range(2)], dim=0)
            values = []

            # Get a list of n_estimations elements, which are Q-values tensors of size n_rollouts
            logger.message("Starting value estimation in the subspace")
            _training_start_time = time.time()
            for _ in range(self.n_estimations):
                replay_workspace = replay_buffer.get_shuffled(alphas.shape[1])
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["critic-1/q_values"].mean(0))

            # Get the average Q-values for each rollout and sort them
            values = torch.stack(values, dim=0).mean(0)
            sorted_values_indices = values.argsort(descending=True)

            best_alphas = alphas[0, sorted_values_indices]
            info["best_alphas"] = best_alphas
            logger.message("Estimated best alpha in the subspace is : " + str(list(map(lambda x: round(x,2), best_alphas[0].tolist()))))
            logger.message("Time elapsed: " + str(round(time.time() - _training_start_time, 0)) + " sec")
            
            del replay_workspace
            del alphas
            del replay_buffer

            r = {"n_epochs": 0, "training_time": time.time() - _training_start_time}
            

        # There is nothing to do if there is only a single policy: the only weight is set to 1 by default
        else:
            best_alpha = None
            r = {"n_epochs": 0, "training_time": 0}
            action_agent.set_best_alpha(alpha=best_alpha, logger=logger)
            info["best_alpha"] = best_alpha

        return r, action_agent, critic_agent, info