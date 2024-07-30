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
                base_env["kwargs"] = self.get_environment_configuration_from_task(domain, task)
            self._train_tasks.append(Task(base_env, k))
            self._test_tasks.append(Task(base_env, k))


    def get_environment_configuration_from_task(self, domain, task):
        match domain:
            case "CartPoleContinuousSubspace-v0" | "CartPoleContinuousSubspace-v1":
                # Environment parameters:
                #     gravity: acceleration due to gravity
                #     masscart: mass of the cart
                #     masspole: mass of the pole
                #     length: length of the pole
                #     force_mag
                #     tau: seconds between state updates
                match task:
                    case "moon":
                        return {
                            "gravity": 1.6
                        }
                    case "normal" | _:
                        return {}
                    
            case "PendulumSubspace-v0" | "PendulumSubspace-v1":
                match task:
                    case "normal" | _:
                        return {}
                    
            case "AcrobotContinuousSubspace-v0" | "AcrobotContinuousSubspace-v1":
                match task:
                    case "normal" | _:
                        return {}
            
            # Environment not supported -> Use default configuration
            case _:
                return {}