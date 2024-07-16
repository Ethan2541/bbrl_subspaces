import typing as tp

from omegaconf import open_dict

from .core import Scenario, Task

configs_path = "../../../../configs/"

class GymScenario(Scenario):
    def __init__(self, domain, repeat_scenario, tasks, base_env, **kwargs):
        super().__init__()
        tasks = list(tasks) * repeat_scenario
        print("Domain:", domain)
        print("Scenario:", tasks)
        for k, task in enumerate(tasks):
            with open_dict(base_env):
                base_env.xml_file = self.get_xml_from_task(task)
            self._train_tasks.append(Task(base_env, k))
            self._test_tasks.append(Task(base_env, k))

    
    def get_xml_from_task(self, task):
        match task:
            case "normal":
                return None