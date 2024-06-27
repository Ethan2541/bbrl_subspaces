# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import typing as tp

import bbrl
import torch
import torch.utils.data
from bbrl import get_class, instantiate_class
from bbrl.agents.agent import Agent
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from functools import partial

assets_path = os.getcwd() + "../../assets/"

class Task:
    """A Reinforcement Learning task defined as a BBRL agent. Use make() method
    to instantiate the bbrl agent corresponding to the task. 

    Parameters
    ----------
    env_agent_cfg   : The OmegaConf (or dict) that allows to configure the BBRL agent
    task_id         : An identifier of the task
    input_dimension : The input dimension of the observations
    output_dimension: The output dimension of the actions (i.e size of the output tensor, or number of actions if discrete actions)
    """
    def __init__(self,env_agent_cfg: dict,
                      task_id: int,
                      is_training_task,
                      input_dimension: tp.Union[None,int] = None,
                      output_dimension: tp.Union[None,int] = None,
                      )  -> None:
        self._task_id = task_id
        self._env_agent_cfg = env_agent_cfg
        self._is_training_task = is_training_task

        if input_dimension is None or output_dimension is None:
            env = self.make()[1]
            obs_size, action_dim = env.get_obs_and_actions_sizes()
            self._input_dimension = obs_size
            self._output_dimension = action_dim
        else:
            self._input_dimension = input_dimension
            self._output_dimension = output_dimension


    def input_dimension(self) -> int:
        return self._input_dimension

    def output_dimension(self) -> int:
        return self._output_dimension

    def task_id(self) -> int:
        return self._task_id

    def env_cfg(self) -> dict:
        return self._env_agent_cfg

    def make(self) -> tp.Tuple[GymAgent, GymAgent]:
        # Returns a pair of environments (train / evaluation) based on a configuration `cfg`

        cfg = self.env_cfg()
        autoreset=True
        include_last_state=True

        train_env_agent, eval_env_agent = None, None
    
        if "xml_file" in cfg.keys():
            xml_file = assets_path + cfg.xml_file
            print("loading:", xml_file)
        else:
            xml_file = None

        if "wrappers" in cfg.keys():
            print("using wrappers:", cfg.wrappers)
            # wrappers_name_list = cfg.wrappers.split(',')
            wrappers_list = []
            wr = get_class(cfg.wrappers)
            # for i in range(len(wrappers_name_list)):
            wrappers_list.append(wr)
            wrappers = wrappers_list
            print(wrappers)
        else:
            wrappers = []

        # Train environment
        if xml_file is None:
            if self._is_training_task:
                train_env_agent = ParallelGymAgent(
                    partial(
                        make_env, cfg.env_name, autoreset=autoreset, wrappers=wrappers
                    ),
                    cfg.n_envs,
                    include_last_state=include_last_state,
                    seed=cfg.seed.train,
                )

            # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
            eval_env_agent = ParallelGymAgent(
                partial(make_env, cfg.env_name, wrappers=wrappers),
                cfg.nb_evals,
                include_last_state=include_last_state,
                seed=cfg.seed.eval,
            )
        else:
            if self._is_training_task:
                train_env_agent = ParallelGymAgent(
                    partial(
                        make_env, cfg.env_name, autoreset=autoreset, wrappers=wrappers
                    ),
                    cfg.n_envs,
                    include_last_state=include_last_state,
                    seed=cfg.seed.train,
                )

            # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
            eval_env_agent = ParallelGymAgent(
                partial(make_env, cfg.env_name, wrappers=wrappers),
                cfg.nb_evals,
                include_last_state=include_last_state,
                seed=cfg.seed.eval,
            )

        return train_env_agent, eval_env_agent
    

class Scenario:
    """ 
    A scenario is a sequence of train tasks and a sequence of test tasks.
    """

    def __init__(self) -> None:
        self._train_tasks = []
        self._test_tasks = []

    def train_tasks(self) -> tp.List[Task]:
        return self._train_tasks

    def test_tasks(self) -> tp.List[Task]:
        return self._test_tasks



class Framework:
    """A (CRL) Model can be updated over one new task, and evaluated over any task
    
    Parameters
    ----------
    seed 
    params : The OmegaConf (or dict) that allows to configure the model
    """
    def __init__(self,seed: int,params: dict) -> None:
        self.seed=seed
        self.cfg=params
        self._stage=0

    def memory_size(self) -> dict:    
        raise NotImplementedError

    def get_stage(self) -> int:
        return self._stage

    def train(self,task: Task,logger: tp.Any, **extra_args) -> None:
        """ Update a model over a particular task.

        Parameters
        ----------
        task: The task to train on
        logger: a bbrl logger to log metrics and messages
        """
        logger.message("-- Train stage "+str(self._stage))
        output=self._train(task,logger.get_logger("stage_"+str(self._stage)+"/"))
        [logger.add_scalar("monitor_per_stage/"+k,output[k],self._stage) for k in output]
        self._stage+=1

    def evaluate(self,test_tasks: tp.List[Task], logger: tp.Any) -> dict:
        """ Evaluate a model over a set of test tasks
        
        Parameters
        ----------
        test_tasks: The set of tasks to evaluate on
        logger: a bbrl logger

        Returns
        ----------
        evaluation: A dict containing some evaluation metrics
        """
        logger.message("Starting evaluation...")
        with torch.no_grad():
            evaluation={}
            for k,task in enumerate(test_tasks):
                metrics=self._evaluate_single_task(task)
                evaluation[task.task_id()]=metrics
                logger.message("Evaluation over task "+str(k)+":"+str(metrics))

        logger.message("-- End evaluation...")
        return evaluation

    def _train(self,task: Task,logger: tp.Any) -> None:
        raise NotImplementedError

    def get_evaluation_agent(self,task_id: int) -> bbrl.agents.agent.Agent:
        raise NotImplementedError

    def _evaluate_single_task(self,task: Task) -> dict:
        metrics={}
        env_agent=task.make()
        policy_agent=self.get_evaluation_agent(task.task_id())

        if not policy_agent is None:
            policy_agent.eval()
            acquisition_agent = TemporalAgent(Agents(env_agent,policy_agent))
            acquisition_agent.seed(self.seed*13+self._stage*100)

            avg_reward=0.0
            n=0
            avg_success=0.0
            for r in range(self.cfg.evaluation.n_rollouts):
                workspace=Workspace()
                acquisition_agent(workspace,t=0,stop_variable="env/done")
                ep_lengths=workspace["env/done"].max(0)[1]+1
                B=ep_lengths.size()[0]
                arange=torch.arange(B)
                cr=workspace["env/cumulated_reward"][ep_lengths-1,arange]
                avg_reward+=cr.sum().item()
                if self.cfg.evaluation.evaluate_success:
                    cr=workspace["env/success"][ep_lengths-1,arange]
                    avg_success+=cr.sum().item()
                n+=B
            avg_reward /= n
            metrics["avg_reward"] = avg_reward

            if self.cfg.evaluation.evaluate_success:
                avg_success/=n
                metrics["success_rate"]=avg_success
        return metrics


class CRLAgent(Agent):
    """A bbrl Agent that is able to apply set_task() and add_regularizer() methods
    """
    def set_task(self,task_id: tp.Union[None,int] = None) -> None:
        pass

    def add_regularizer(self, *args) -> torch.Tensor:
        return torch.Tensor([0.])

class CRLAgents(Agents):
    """A batch of CRL Agents called sequentially.
    """
    def set_task(self,task_id: tp.Union[None,int] = None) -> None:
        for agent in self:
            agent.set_task(task_id)

    def add_regularizer(self, *args) -> torch.Tensor:
        return torch.cat([agent.add_regularizer(*args) for agent in self]).sum()


class Logger:
    def __init__(self, logger):
        self.logger = logger

    def add_log(self, log_string, log_item, steps):
        if isinstance(log_item, torch.Tensor) and log_item.dim() == 0:
            log_item = log_item.item()
        self.logger.add_scalar(log_string, log_item, steps)

    # A specific function for RL algorithms having a critic, an actor and an entropy losses
    def log_losses(self, critic_loss, entropy_loss, actor_loss, steps):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)
        self.add_log("reward/std", rewards.std(), nb_steps)

    def close(self) -> None:
        self.logger.close()