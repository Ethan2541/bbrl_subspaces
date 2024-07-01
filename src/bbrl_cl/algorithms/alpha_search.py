# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import torch
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from ternary.helpers import simplex_iterator
from torch.distributions.dirichlet import Dirichlet

from bbrl_cl.agents.utils import LinearSubspace


# After removing an anchor, the last added anchor should still be frozen (optimal subspace reached)
def remove_anchor(model):
    model.agents[1].n_anchors -= 1
    for nn_module in model[1].model:
        if isinstance(nn_module,LinearSubspace):
            nn_module.anchors = nn_module.anchors[:-1]
            nn_module.n_anchors -= 1
    return model



class AlphaSearch:
    def __init__(self, params):
        self.cfg = params

    def run(self, action_agent, critic_agent, task, logger, seed, info={}):
        logger = logger.get_logger(type(self).__name__ + str("/"))
        n_anchors = action_agent[0].n_anchors
        
        # Subspace of policies (several anchors)
        if (n_anchors > 1):
            replay_buffer = info["replay_buffer"]
            n_samples = self.cfg.n_samples
            n_rollouts = self.cfg.n_rollouts
            n_steps = self.cfg.n_validation_steps

            # Estimating best alphas in the subspace
            alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([n_samples]))
            alphas = torch.stack([alphas for _ in range(self.cfg.time_size)], dim=0)
            values = []

            logger.message("Starting value estimation in the subspace")
            _training_start_time = time.time()
            # K-shot adaptation where K is the number of estimations
            for _ in range(self.cfg.n_estimations):
                replay_workspace = replay_buffer.get_shuffled(alphas.shape[1])
                replay_workspace.set_full("alphas",alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["critic-1/q_values"].mean(0))

            values = torch.stack(values, dim=0).mean(0)
            best_alphas = alphas[0, values.topk(n_rollouts).indices]
            info["best_alphas"] = best_alphas
            logger.message("Estimated best alpha in the subspace is : " + str(list(map(lambda x: round(x,2), best_alphas[0].tolist()))))
            logger.message("Time elapsed: " + str(round(time.time() - _training_start_time, 0)) + " sec")
            
            del replay_workspace
            del alphas
            del replay_buffer
            
            # Validating best alphas through rollout using some budget
            logger.message("Evaluating the two best alphas...")
            B = self.cfg.n_rollouts
            task._env_agent_cfg["n_envs"] = B
            _, env_agent = task.make()

            action_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent))
            acquisition_agent.seed(seed)
            w = Workspace()
            with torch.no_grad():
                acquisition_agent(w, t=0, n_steps=n_steps, alphas=best_alphas)

            logger.message("Acquisition ended")
            cumulative_rewards =  w["env/cumulated_reward"][-1]
            best_reward = cumulative_rewards.max()
            best_alpha = best_alphas[cumulative_rewards.argmax()]

            del w

        # There is nothing to do if there is only a single policy: the only weight is set to 1 by default
        else:
            best_alpha = None
            r = {"n_epochs":0, "training_time":0}
            action_agent.set_best_alpha(alpha=best_alpha, logger=logger)
            info["best_alpha"] = best_alpha

        return r, action_agent, critic_agent, info