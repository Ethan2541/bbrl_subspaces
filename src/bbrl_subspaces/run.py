# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import hydra

from bbrl import instantiate_class
from bbrl.agents import Agents, TemporalAgent

from bbrl_algos.models.envs import get_env_agents



@hydra.main(
    config_path="./configs/",
    # config_name="sac_cartpole.yaml",
    config_name="sac_pendulum.yaml",
    version_base="1.3",
)
def main(cfg):
    logger = instantiate_class(cfg.logger)
    sac = instantiate_class(cfg.subspace_algorithm)
    alpha_search = instantiate_class(cfg.alpha_search)
    visualizer = instantiate_class(cfg.visualization)

    train_env_agent, eval_env_agent = get_env_agents(cfg.subspace_algorithm.params)
    r, action_agent, critic_agent, info = sac.run(train_env_agent, eval_env_agent, logger)
    r, action_agent, critic_agent, info = alpha_search.run(action_agent, critic_agent, logger, info)

    visualizer.plot_subspace(TemporalAgent(Agents(eval_env_agent, action_agent)), logger)

if __name__ == "__main__":
    main()