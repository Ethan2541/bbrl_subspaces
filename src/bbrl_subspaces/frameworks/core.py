# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import typing as tp

import bbrl
import torch
import torch.utils.data
from bbrl.agents.gymnasium import GymAgent
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_subspaces.agents.utils import get_env_agents


class Task:
    """A Reinforcement Learning task defined as a BBRL agent. Use make() method
    to instantiate the BBRL agent corresponding to the task. 
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
        train_env_agent, eval_env_agent, alpha_env_agent = get_env_agents(cfg, alpha_search=True)
        return train_env_agent, eval_env_agent, alpha_env_agent



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
        self.seed = seed
        self.cfg = params
        self._stage = 0


    def memory_size(self) -> dict:    
        raise NotImplementedError


    def get_stage(self) -> int:
        return self._stage


    def train(self,task: Task,logger: tp.Any, **kwargs) -> None:
        """ Update a model over a particular task.
        Parameters
        ----------
        task: The task to train on
        logger: a bbrl logger to log metrics and messages
        """

        logger.message("-- Train stage " + str(self._stage))
        output = self._train(task, logger.get_logger("stage_" + str(self._stage) + "/"))
        [logger.add_scalar("monitor_per_stage/" + k, output[k], self._stage) for k in output]
        self._stage += 1


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
            evaluation = {}
            for k, task in enumerate(test_tasks):
                metrics = self._evaluate_single_task(task)
                evaluation[task.task_id()] = metrics
                logger.message("Evaluation over task " + str(k) + ":" + str(metrics))
        logger.message("-- End evaluation...")
        return evaluation


    def _train(self, task: Task, logger: tp.Any) -> None:
        raise NotImplementedError


    def get_evaluation_agent(self, task_id: int) -> bbrl.agents.agent.Agent:
        raise NotImplementedError


    def _evaluate_single_task(self, task: Task) -> dict:
        raise NotImplementedError