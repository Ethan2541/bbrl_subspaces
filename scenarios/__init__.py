# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import typing as tp

from bbrl.agents.gymnasium import make_env
from bbrl_cl.core import Scenario, Task

class GymScenario(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,n_steps,domain,tasks, repeat_scenario, **kwargs):
        super().__init__()
        tasks = list(tasks) * repeat_scenario
        print("Domain:",domain)
        print("Scenario:",tasks)
        for k,task in enumerate(tasks):
            agent_cfg={
                "classname": "bbrl.agents.gymnasium.ParallelGymAgent",
                "make_env_fn": make_env,
                "num_envs": n_train_envs,
                "make_env_args": {
                                  "env_name": domain,
                                  "autoreset": True,
                                 }
            }
            self._train_tasks.append(Task(agent_cfg,k,n_steps))
            test_cfg={
                "classname": "bbrl.agents.gymnasium.ParallelGymAgent",
                "make_env_fn": make_env,
                "num_envs": n_evaluation_envs,
                "make_env_args": {
                                  "env_name": domain,
                                  "autoreset": True,
                                 }
            }
            self._test_tasks.append(Task(test_cfg,k))