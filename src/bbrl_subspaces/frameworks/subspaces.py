# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import numpy as np
import torch

from bbrl import instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from torch.distributions.dirichlet import Dirichlet

from .core import Framework


class Subspace(Framework):
    """
    Model for the subspace method.
    """ 

    def __init__(self, seed, train_algorithm, alpha_search, evaluation, visualization):
        super().__init__(seed)
        torch.manual_seed(self.seed)
        self.train_algorithm_cfg = train_algorithm
        self.train_algorithm = instantiate_class(self.train_algorithm_cfg)
        self.alpha_search = instantiate_class(alpha_search)
        self.evaluation_cfg = evaluation
        self.visualizer = instantiate_class(visualization)
        self.policy_agent = None
        self.critic_agent_1 = None
        self.critic_agent_2 = None


    def _train(self, task, logger):
        task_id = task.task_id()
        info = {"task_id": task_id}
        if task_id > 0:
            self.alpha_search.is_initial_task = False
            self.policy_agent.add_anchor(logger=logger)
            self.critic_agent_1.add_anchor(n_anchors=self.policy_agent[0].n_anchors, logger=logger)
            self.critic_agent_2.add_anchor(n_anchors=self.policy_agent[0].n_anchors, logger=logger)

        train_env_agent, eval_env_agent, alpha_env_agent = task.make()
        r1, self.policy_agent, self.critic_agent_1, self.critic_agent_2, info = self.train_algorithm.run(train_env_agent, eval_env_agent, self.policy_agent, self.critic_agent_1, self.critic_agent_2, logger, visualizer=self.visualizer)
        r2, self.policy_agent, self.critic_agent_1, info = self.alpha_search.run(alpha_env_agent, self.policy_agent, self.critic_agent_1, logger, info)
        self.visualizer.plot_subspace(TemporalAgent(Agents(eval_env_agent, self.policy_agent)), logger, info)
        self.visualizer.reset()
        return r1
    

    def test_anticollapse_coefficients(self, task, logger, anticollapse_coefficients=[0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0], train_seeds=[123, 456, 789, 101112, 131415], eval_seeds=[161718, 192021, 222324, 252627, 282930]):
        task_id = task.task_id()
        info = {"task_id": task_id}
        if task_id > 0:
            self.alpha_search.is_initial_task = False
            self.policy_agent.add_anchor(logger=logger)
            self.critic_agent_1.add_anchor(n_anchors=self.policy_agent[0].n_anchors, logger=logger)
            self.critic_agent_2.add_anchor(n_anchors=self.policy_agent[0].n_anchors, logger=logger)

        for train_seed, eval_seed in zip(train_seeds, eval_seeds):
            task._env_agent_cfg.seed.train = train_seed
            task._env_agent_cfg.seed.eval = eval_seed
            for anticollapse_coefficient in anticollapse_coefficients:
                logger.message(f"Setting the anticollapse coefficient to {anticollapse_coefficient}")
                cfg = self.train_algorithm_cfg
                cfg.params.algorithm.anticollapse_coef = float(anticollapse_coefficient)
                train_algorithm = instantiate_class(cfg)
                self.reset_agents()
                train_env_agent, eval_env_agent, alpha_env_agent = task.make()
                r1, self.policy_agent, self.critic_agent_1, self.critic_agent_2, info = train_algorithm.run(train_env_agent, eval_env_agent, self.policy_agent, self.critic_agent_1, self.critic_agent_2, logger, visualizer=self.visualizer)
                self.visualizer.reset()
        return r1
    

    def reset_agents(self):
        self.policy_agent = None
        self.critic_agent_1 = None
        self.critic_agent_2 = None


    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.parameters())
        return {"n_parameters": pytorch_total_params}


    def get_evaluation_agent(self, task_id):
        self.policy_agent.set_task(task_id)
        return copy.deepcopy(self.policy_agent)


    def _evaluate_single_task(self, task):
        _, env_agent, _ = task.make()
        actor = self.get_evaluation_agent(task.task_id())
        actor.eval()
        eval_agent = TemporalAgent(Agents(env_agent, actor))

        # Evaluating best alpha
        rewards = []
        w = Workspace()
        for _ in range(self.evaluation_cfg.n_rollouts):
            with torch.no_grad():
                eval_agent(w, t=0, stop_variable="env/done")
            ep_lengths= w["env/done"].max(0)[1] + 1
            B = ep_lengths.size()[0]
            arange = torch.arange(B)
            cr = w["env/cumulated_reward"][ep_lengths-1, arange]
            rewards.append(cr)
        rewards = torch.stack(rewards, dim=0).mean()
        metrics = {"avg_reward" : rewards.item()}
        del w

        # Evaluating oracle
        if self.evaluation_cfg.oracle_rollouts > 0:
            rewards = []
            w = Workspace()
            n_anchors = actor[0].n_anchors
            alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([B]))
            for _ in range(self.evaluation_cfg.oracle_rollouts):
                with torch.no_grad():
                    eval_agent(w, t=0, alphas=alphas, stop_variable="env/done")
                ep_lengths = w["env/done"].max(0)[1] + 1
                B = ep_lengths.size()[0]
                arange = torch.arange(B)
                cr = w["env/cumulated_reward"][ep_lengths-1, arange]
                rewards.append(cr)
            rewards = torch.stack(rewards, dim=0).mean(0).max()
            metrics["oracle_reward"] = rewards.item()
            del w
        return metrics