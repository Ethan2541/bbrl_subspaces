# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import torch
from bbrl import instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from torch.distributions.dirichlet import Dirichlet

from bbrl_cl.core import Framework


class Subspace(Framework):
    """
    Model for the subspace method.
    """
    def __init__(self, seed, params):
        super().__init__(seed, params)
        self.train_algorithm = instantiate_class(self.cfg.train_algorithm)
        self.alpha_search = instantiate_class(self.cfg.alpha_search)
        self.policy_agent = None
        self.critic_agent = None

    def _train(self,task,logger):
        task_id = task.task_id()
        info = {"task_id": task_id}
        if task_id > 0:
            self.policy_agent.add_anchor(logger=logger)
            self.critic_agent.add_anchor(n_anchors = self.policy_agent[0].n_anchors,logger = logger)

        train_env_agent, eval_env_agent = task.make()
        r1, self.policy_agent, self.critic_agent, info = self.train_algorithm.run(train_env_agent, eval_env_agent, logger, self.seed, info=info)
        r2, self.policy_agent, self.critic_agent, info = self.alpha_search.run(self.policy_agent, self.critic_agent, task, logger, self.seed, info=info)
        return r1

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.parameters())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        self.policy_agent.set_task(task_id)
        return copy.deepcopy(self.policy_agent), self.critic_agent

    def evaluate(self,test_tasks,logger):
        """ Evaluate a model over a set of test tasks
        Args:
            test_tasks: The set of tasks to evaluate on
            logger
        Returns:
            evaluation: Some statistics about the evaluation (i.e metrics)
        """
        logger.message("Starting evaluation...")
        with torch.no_grad():
            evaluation={}
            for k,task in enumerate(test_tasks):
                metrics=self._evaluate_single_task(task,logger)
                evaluation[task.task_id()]=metrics
                logger.message("Evaluation over task "+str(k)+":"+str(metrics))

        logger.message("-- End evaluation...")
        return evaluation

    def _evaluate_single_task(self,task,logger):
        _, env_agent = task.make()
        actor, _ = self.get_evaluation_agent(task.task_id())
        actor.eval()
        
        ev_agent = Agents(env_agent, actor)
        eval_agent = TemporalAgent(ev_agent)

        # Evaluating best alpha
        rewards = []
        w = Workspace()
        for i in range(self.cfg.evaluation.n_rollouts):
            with torch.no_grad():
                eval_agent(w, t=0, stop_variable="env/done") # stochastic=False?
            ep_lengths= w["env/done"].max(0)[1]+1
            B = ep_lengths.size()[0]
            arange = torch.arange(B)
            cr = w["env/cumulated_reward"][ep_lengths-1,arange]
            rewards.append(cr)
        rewards = torch.stack(rewards, dim = 0).mean()
        metrics={"avg_reward" : rewards.item()}
        del w

        # Evaluating oracle
        if self.cfg.evaluation.oracle_rollouts>0:
            rewards = []
            w = Workspace()
            n_anchors = actor[0].n_anchors
            alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([B]))
            for i in range(self.cfg.evaluation.oracle_rollouts):
                with torch.no_grad():
                    eval_agent(w, t=0, alphas=alphas, stop_variable="env/done")
                ep_lengths= w["env/done"].max(0)[1]+1
                B = ep_lengths.size()[0]
                arange = torch.arange(B)
                cr = w["env/cumulated_reward"][ep_lengths-1,arange]
                rewards.append(cr)
            rewards = torch.stack(rewards, dim=0).mean(0).max()
            metrics["oracle_reward"] = rewards.item()
            del w
        return metrics