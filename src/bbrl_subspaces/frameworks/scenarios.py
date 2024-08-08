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
                #     force_mag: intensity of the force applied to the cart
                #     tau: seconds between state updates
                match task:
                    case "moon":
                        return {
                            "gravity": 1.6
                        }
                    case "hugecart":
                        return {
                            "masscart": 10.0
                        }
                    case "hugecart_moon":
                        return {
                            "masscart": 10.0,
                            "gravity": 1.6
                        }
                    case "tinycart":
                        return {
                            "masscart": 0.1
                        }
                    case "tinycart_moon":
                        return {
                            "masscart": 0.1,
                            "gravity": 1.6
                        }
                    case "shortpole":
                        return {
                            "length": 0.25
                        }
                    case "longpole":
                        return {
                            "length": 1.0
                        }
                    case "normal" | _:
                        return {}
                    
            case "PendulumSubspace-v0" | "PendulumSubspace-v1":
                # Environment parameters:
                #     max_speed: maximum angular speed
                #     max_torque: maximum torque
                #     dt: seconds between state updates
                #     g: acceleration due to gravity
                #     m: mass of the pendulum
                #     l: length of the pendulum
                match task:
                    case "normal" | _:
                        return {}
                    
            case "AcrobotContinuousSubspace-v0" | "AcrobotContinuousSubspace-v1":
                # Environment parameters:
                #     dt: seconds between state updates
                #     link_length_1: length of link 1
                #     link_length_2: length of link 2
                #     link_mass_1: mass of link 1
                #     link_mass_2: mass of link 2
                #     link_com_pos_1: position of the center of mass of link 1
                #     link_com_pos_2: position of the center of mass of link 2
                #     link_moi: moments of inertia for both links
                #     g: acceleration due to gravity
                #     max_vel_1: maximum angular velocity of link 1
                #     max_vel_2: maximum angular velocity of link 2
                #     torque_noise_max: maximum noise to be added to the torque
                match task:
                    case "normal" | _:
                        return {}
            
            # Unsupported environment: use default configuration
            case _:
                return {}