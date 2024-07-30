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
    def __init__(self, n_rollouts, n_samples, n_validation_steps, seed, prune_subspace=True, is_initial_task=True):
        self.is_initial_task = is_initial_task
        self.n_rollouts = n_rollouts
        self.n_samples = n_samples
        self.n_validation_steps = n_validation_steps
        self.prune_subspace = prune_subspace    # If False, the anchors are always kept in the subspace
        self.seed = seed

    def run(self, env_agent, action_agent, critic_agent, logger, info={}):
        torch.manual_seed(self.seed)
        logger = logger.get_logger(type(self).__name__ + "/")
        n_anchors = action_agent[0].n_anchors

        # In the initial subspace, the former and new subspaces should be the same
        if self.is_initial_task:
            former_n_anchors = n_anchors
        else:
            former_n_anchors = n_anchors - 1
        
        # Subspace of policies (several anchors)
        if n_anchors > 1:
            replay_buffer = info["replay_buffer"]
            n_rollouts = self.n_rollouts
            n_samples = self.n_samples
            n_steps = self.n_validation_steps

            # Estimate best alphas in the current subspace using n_samples sampled policies
            alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([n_samples]))
            alphas = torch.stack([alphas for _ in range(2)], dim=0)
            values = []

            # Get a list of n_samples elements, which are Q-values tensors of size n_rollouts
            logger.message("Starting value estimation in the current subspace")
            _training_start_time = time.time()
            for _ in range(self.n_samples):
                replay_workspace = replay_buffer.get_shuffled(alphas.shape[1])
                replay_workspace.set_full("alphas", alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["critic-1/q_values"].mean(0))

            # Get the average Q-values for each rollout and sort them to keep the best n_rollouts // 2
            values = torch.stack(values, dim=0).mean(0)
            topk_values_indices = values.topk(n_rollouts // 2).indices

            best_alphas = alphas[0, topk_values_indices]
            info["best_alphas"] = best_alphas
            logger.message("Estimated best alpha in the current subspace is : " + str(list(map(lambda x: round(x,2), best_alphas[0].tolist()))))
            logger.message("Time elapsed: " + str(round(time.time() - _training_start_time, 0)) + " sec")
            

            # Estimate best alphas in the former subspace using n_samples sampled policies
            alphas = Dirichlet(torch.ones(former_n_anchors)).sample(torch.Size([n_samples]))
            # Zero padding to match the size of the current subspace
            alphas = torch.cat([alphas, torch.zeros(*alphas.shape[:-1], 1)], dim=-1)
            alphas = torch.stack([alphas for _ in range(2)], dim=0)
            values = []
            
            # Get a list of n_samples elements, which are Q-values tensors of size n_rollouts
            logger.message("Starting value estimation in the former subspace")
            _training_start_time = time.time()
            for _ in range(self.n_samples):
                replay_workspace = replay_buffer.get_shuffled(alphas.shape[1])
                replay_workspace.set_full("alphas", alphas)
                with torch.no_grad():
                    critic_agent(replay_workspace)
                values.append(replay_workspace["critic-1/q_values"].mean(0))

            # Get the average Q-values for each rollout and sort them to keep the best n_rollouts - n_rollouts // 2
            values = torch.stack(values, dim=0).mean(0)
            topk_values_indices = values.topk(n_rollouts - n_rollouts // 2).indices

            best_alphas_before_training = alphas[0, topk_values_indices]
            info["best_alphas_before_training"] = best_alphas_before_training
            logger.message("Estimated best alpha in the former subspace is : " + str(list(map(lambda x: round(x,2), best_alphas_before_training[0].tolist()))))
            logger.message("Time elapsed: " + str(round(time.time() - _training_start_time, 0)) + " sec")


            del replay_workspace
            del alphas
            del replay_buffer


            # Validating best alphas through rollout using some budget
            logger.message("Evaluating the two best alphas...")
            _validation_start_time = time.time()

            alphas = torch.cat([best_alphas, best_alphas_before_training], dim=0)

            action_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent, action_agent))
            w = Workspace()
            with torch.no_grad():
                acquisition_agent(w, t=0, n_steps=n_steps, alphas=alphas)
            logger.message("Acquisition ended")

            cumulative_rewards, cumulative_rewards_before_training =  w["env/cumulated_reward"][-1].chunk(2)
            best_reward = cumulative_rewards.max().item()
            best_reward_before_training = cumulative_rewards_before_training.max().item()
            
            if best_reward > best_reward_before_training:
                best_alpha = best_alphas[cumulative_rewards.argmax()]
                action_agent.set_best_alpha(alpha=best_alpha, logger=logger)
                info["best_alpha_reward"] = best_reward
                logger.message("Best reward is with the current subspace: " + str(round(best_reward, 2)))
                info["best_alpha"] = best_alpha
            else:
                best_alpha = best_alphas_before_training[cumulative_rewards_before_training.argmax()]
                action_agent.set_best_alpha(alpha=best_alpha, logger=logger)
                info["best_alpha_reward"] = best_reward_before_training
                logger.message("Best reward is with the former subspace: " + str(round(best_reward_before_training, 2)))
                
                if self.prune_subspace:
                    logger.message("Pruning the subspace")
                    action_agent.remove_anchor(logger=logger)

                info["best_alpha"] = best_alpha[:-1]
            
            logger.message("Best distribution after validation: " + str(list(map(lambda x: round(x,2), best_alpha.tolist()))))
            logger.message("Time elapsed: " + str(round(time.time() - _training_start_time, 0)) + " sec")

            r = {"n_epochs": 0, "training_time": time.time() - _validation_start_time}
            del w


        # There is nothing to do if there is only a single policy: the only weight is set to 1 by default
        else:
            best_alpha = None
            r = {"n_epochs": 0, "training_time": 0}
            action_agent.set_best_alpha(alpha=best_alpha, logger=logger)
            info["best_alpha"] = best_alpha

        return r, action_agent, critic_agent, info