# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import hydra

from bbrl import instantiate_class
from bbrl.agents import Agents, TemporalAgent

from bbrl_algos.models.envs import get_env_agents

from bbrl_cl.algorithms.sac import SAC
from bbrl_cl.visualization.subspace_visualizer import SubspaceVisualizer



@hydra.main(
    config_path="./configs/",
    # config_name="sac_cartpole.yaml",
    config_name="sac_pendulum.yaml",
    version_base="1.3",
)
def main(cfg):
    train_env_agent, eval_env_agent = get_env_agents(cfg)
    logger = instantiate_class(cfg.logger)

    r, action_agent, critic_agent, info = SAC(cfg).run(train_env_agent, eval_env_agent, logger, cfg.algorithm.seed.torch)
    a_s = instantiate_class(cfg.alpha_search)
    r, action_agent, critic_agent, info = a_s.run(action_agent, critic_agent, logger, info)

    SubspaceVisualizer(cfg.visualization).plot_subspace(TemporalAgent(Agents(eval_env_agent, action_agent)), logger)

if __name__ == "__main__":
    main()