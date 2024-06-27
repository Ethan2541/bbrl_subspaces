import typing as tp

from omegaconf import OmegaConf

from bbrl_cl.core import Scenario, Task

configs_path = "../../../../configs/"

class GymScenario(Scenario):
    def __init__(self, domain, tasks, tasks_cfgs, repeat_scenario, **kwargs):
        super().__init__()
        tasks = list(tasks) * repeat_scenario
        print("Domain:", domain)
        print("Scenario:", tasks)
        for k, task_path in enumerate(tasks_cfgs):
            task_cfg = OmegaConf.load(configs_path + task_path)
            self._train_tasks.append(Task(task_cfg.training, k, True))
            self._test_tasks.append(Task(task_cfg.test, k, False))