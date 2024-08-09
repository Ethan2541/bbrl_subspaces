# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import hydra
import time

from bbrl import instantiate_class
from bbrl.agents import Agents, TemporalAgent

from bbrl_subspaces.agents.utils import get_env_agents



@hydra.main(
    config_path="./configs/",
    config_name="cartpole.yaml",
    # config_name="lunar_lander.yaml",
    # config_name="pendulum.yaml",
    # config_name="rocket_lander.yaml",
    # config_name="swimmer.yaml",
    version_base="1.3",
)
def main(cfg):
    _start = time.time()
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg, verbose=False)
    framework = instantiate_class(cfg.framework)
    scenario = instantiate_class(cfg.scenario)
    stage = framework.get_stage()
    for train_task in scenario.train_tasks()[stage:]:
        # framework.test_anticollapse_coefficients(train_task, logger, anticollapse_coefficients=[-0.001])
        framework.train(train_task, logger)
        evaluation = framework.evaluate(scenario.test_tasks(), logger)
        metrics = {}
        for tid in evaluation:
            for k,v in evaluation[tid].items():
                logger.add_scalar("evaluation/" + str(tid) + "_" + k, v, stage)
                metrics[k] = v + metrics.get(k, 0)
        for k,v in metrics.items():
            logger.add_scalar("evaluation/aggregate_" + k, v / len(evaluation), stage)
        m_size = framework.memory_size()
        for k, v in m_size.items():
            logger.add_scalar("memory/" + k, v, stage)
        stage += 1
    logger.close()
    logger.message("Time elapsed: " + str(round((time.time() - _start), 0)) + " sec")


if __name__ == "__main__":
    main()