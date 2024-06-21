# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import typing as tp

from bbrl_cl.core import Scenario, Task

class GymScenario(Scenario):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        tasks = list(tasks) * cfg.repeat_scenario
        print("Domain:", cfg.domain)
        print("Scenario:", cfg.tasks)
        for k, task_cfg in enumerate(cfg.tasks_cfgs):
            self._train_tasks.append(Task(task_cfg.training, k))
            self._test_tasks.append(Task(task_cfg.test, k))