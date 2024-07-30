# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.utils.data


class Logger:
    def __init__(self, logger):
        self.logger = logger

    def add_log(self, log_string, log_item, steps):
        if isinstance(log_item, torch.Tensor) and log_item.dim() == 0:
            log_item = log_item.item()
        self.logger.add_scalar(log_string, log_item, steps)

    # A specific function for RL algorithms having a critic, an actor and an entropy losses
    def log_losses(self, critic_loss, entropy_loss, actor_loss, steps):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)
        self.add_log("reward/std", rewards.std(), nb_steps)

    def close(self) -> None:
        self.logger.close()