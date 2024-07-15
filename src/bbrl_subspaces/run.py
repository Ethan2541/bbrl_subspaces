# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import hydra

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
    logger = instantiate_class(cfg.logger)
    sac = instantiate_class(cfg.subspace_algorithm)
    alpha_search = instantiate_class(cfg.alpha_search)
    visualizer = instantiate_class(cfg.visualization)

    train_env_agent, eval_env_agent, alpha_env_agent = get_env_agents(cfg, alpha_search=True)
    r, action_agent, critic_agent, info = sac.run(train_env_agent, eval_env_agent, logger, visualizer=visualizer)
    r, action_agent, critic_agent, info = alpha_search.run(alpha_env_agent, action_agent, critic_agent, logger, info)
    visualizer.plot_subspace(TemporalAgent(Agents(eval_env_agent, action_agent)), logger, info)

if __name__ == "__main__":
    main()