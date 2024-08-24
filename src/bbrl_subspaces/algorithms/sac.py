# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.dirichlet import Dirichlet

from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.shared_models import soft_update_params
from bbrl_algos.models.utils import save_best

from bbrl_subspaces.agents.utils import SubspaceAgents
from bbrl_subspaces.logger import Logger

import matplotlib
import time

matplotlib.use("TkAgg")


class SAC:
    def __init__(self, name, env_name, params, **kwargs):
        self.name = name
        self.env_name = env_name
        self.cfg = params


    # Create the SAC Agent
    def create_sac_agent(self, train_env_agent, eval_env_agent, actor, critic_1, critic_2):
        obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
        assert (
            train_env_agent.is_continuous_action()
        ), "SAC code dedicated to continuous actions"
        
        policy_agent_cfg = self.cfg.policy_agent
        policy_agent_cfg.input_dimension = obs_size
        policy_agent_cfg.output_dimension = act_size

        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.obs_dimension = obs_size
        critic_agent_cfg.action_dimension = act_size
        
        if actor is None:
            actor = instantiate_class(policy_agent_cfg)

        tr_agent = Agents(train_env_agent, actor)
        ev_agent = Agents(eval_env_agent, actor)

        if critic_1 is None:
            critic_1 = instantiate_class(critic_agent_cfg)
        critic_1 = critic_1.set_name("critic-1")
        target_critic_1 = copy.deepcopy(critic_1).set_name("target-critic-1")

        if critic_2 is None:
            critic_2 = instantiate_class(critic_agent_cfg)
        critic_2 = critic_2.set_name("critic-2")
        target_critic_2 = copy.deepcopy(critic_2).set_name("target-critic-2")

        train_agent = TemporalAgent(tr_agent)
        eval_agent = TemporalAgent(ev_agent)
        return (
            train_agent,
            eval_agent,
            actor,
            critic_1,
            target_critic_1,
            critic_2,
            target_critic_2,
        )


    # Configure the actor and critic optimizers
    def setup_optimizers(self, actor, critic_1, critic_2):
        actor_optimizer_args = get_arguments(self.cfg.actor_optimizer)
        parameters = actor.parameters()
        actor_optimizer = get_class(self.cfg.actor_optimizer)(parameters, **actor_optimizer_args)
        critic_optimizer_args = get_arguments(self.cfg.critic_optimizer)
        parameters = nn.Sequential(critic_1, critic_2).parameters()
        critic_optimizer = get_class(self.cfg.critic_optimizer)(
            parameters, **critic_optimizer_args
        )
        return actor_optimizer, critic_optimizer


    def setup_entropy_optimizers(self):
        if self.cfg.algorithm.entropy_mode == "auto":
            entropy_coef_optimizer_args = get_arguments(self.cfg.entropy_coef_optimizer)
            # Note: we optimize the log of the entropy coef which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            # Comment and code taken from the SB3 version of SAC
            log_entropy_coef = torch.log(
                torch.ones(1) * self.cfg.algorithm.init_entropy_coef
            ).requires_grad_(True)
            entropy_coef_optimizer = get_class(self.cfg.entropy_coef_optimizer)(
                [log_entropy_coef], **entropy_coef_optimizer_args
            )
            return entropy_coef_optimizer, log_entropy_coef
        else:
            return None, None


    def compute_critic_loss(
        self,
        reward,
        must_bootstrap,
        current_actor,
        q_agents,
        target_q_agents,
        rb_workspace,
        ent_coef,
    ):
        """
        Computes the critic loss, using the sampled policy, for a set of $S$ transition samples

        Args:
            cfg: The experimental configuration
            reward: Tensor (2xS) of rewards
            must_bootstrap: Tensor (S) of indicators
            current_actor: The actor agent (as a TemporalAgent)
            q_agents: The critics (as a TemporalAgent)
            target_q_agents: The target of the critics (as a TemporalAgent)
            rb_workspace: The transition workspace
            ent_coef: The entropy coefficient

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The two critic losses (scalars)
        """

        # Compute q_values from both critics with the actions present in the buffer:
        # at t, we have Q(s,a) from the (s,a) in the RB
        q_agents(rb_workspace, t=0, n_steps=1)
        
        with torch.no_grad():
            # Replay the current actor on the replay buffer to get actions of the
            # current actor
            current_actor(rb_workspace, t=1, n_steps=1, predict_proba=True)

            # Compute target q_values from both target critics: at t+1, we have
            # Q(s+1,a+1) from the (s+1,a+1) where a+1 has been replaced in the RB
            target_q_agents(rb_workspace, t=1, n_steps=1)

            action_logprobs_next = rb_workspace["action_logprobs"]

        q_values_rb_1, q_values_rb_2, post_q_values_1, post_q_values_2 = rb_workspace[
            "critic-1/q_values",
            "critic-2/q_values",
            "target-critic-1/q_values",
            "target-critic-2/q_values",
        ]

        q_next = torch.min(post_q_values_1[1], post_q_values_2[1]).squeeze(-1)
        v_phi = q_next - ent_coef * action_logprobs_next[1]

        target = reward[-1] + self.cfg.algorithm.discount_factor * v_phi * must_bootstrap.int()
        td_1 = target - q_values_rb_1[0].squeeze(-1)
        td_2 = target - q_values_rb_2[0].squeeze(-1)
        td_error_1 = td_1**2
        td_error_2 = td_2**2
        critic_loss_1 = td_error_1.mean()
        critic_loss_2 = td_error_2.mean()

        return critic_loss_1, critic_loss_2


    def compute_actor_loss(self, ent_coef, current_actor, q_agents, rb_workspace, logger, nb_steps):
        """
        Actor loss computation
        :param ent_coef: The entropy coefficient $\alpha$
        :param current_actor: The actor agent (temporal agent)
        :param q_agents: The critics (as temporal agent)
        :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
        """

        # Recompute the q_values from the current actor, not from the actions in the buffer

        current_actor(rb_workspace, t=0, n_steps=1, predict_proba=True)
        action_logprobs_new = rb_workspace["action_logprobs"]

        q_agents(rb_workspace, t=0, n_steps=1)
        q_values_1, q_values_2 = rb_workspace["critic-1/q_values", "critic-2/q_values"]

        current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)

        actor_loss = ent_coef * action_logprobs_new[0] - current_q_values[0]
        logger.add_log("rl_loss", actor_loss.mean(), nb_steps)

        # Adding a penalty term to ensure that the policies are different enough to prevent the subspace from collapsing
        # A positive anticollapse coefficient increases the distance between the anchors, while a negative coefficient decreases it
        penalty = sum(list(current_actor.agent.euclidean_distances().values()))
        logger.add_log("distance_loss", penalty, nb_steps)
        actor_loss -= self.cfg.algorithm.anticollapse_coef * penalty

        return actor_loss.mean()


    def run(self, train_env_agent, eval_env_agent, policy_agent, critic_agent_1, critic_agent_2, logger, info={}, visualizer=None):
        torch.random.manual_seed(seed=self.cfg.algorithm.seed.torch)
        bbrl_logger = Logger(logger)
        logger = logger.get_logger(type(self).__name__ + "/")
        logger.message("Initialization")
        best_reward = float("-inf")
        n_epochs = 0
        n_steps_average_rewards = []
        subspace_areas = []
        average_subspace_rewards = []
        max_subspace_rewards = []
        best_average_subspace_reward = float("-inf")
        best_max_subspace_reward = float("-inf")

        # init_entropy_coef is the initial value of the entropy coef alpha.
        ent_coef = self.cfg.algorithm.init_entropy_coef
        tau = self.cfg.algorithm.tau_target

        # Create the SAC Agent
        (
            train_agent,
            eval_agent,
            actor,
            critic_1,
            target_critic_1,
            critic_2,
            target_critic_2,
        ) = self.create_sac_agent(train_env_agent, eval_env_agent, policy_agent, critic_agent_1, critic_agent_2)

        actor.train()
        current_actor = TemporalAgent(actor)
        q_agents = TemporalAgent(SubspaceAgents(critic_1, critic_2))
        target_q_agents = TemporalAgent(SubspaceAgents(target_critic_1, target_critic_2))
        train_workspace = Workspace()

        # Creates a replay buffer
        rb = ReplayBuffer(max_size=self.cfg.algorithm.buffer_size)

        # Configure the optimizer
        actor_optimizer, critic_optimizer = self.setup_optimizers(actor, critic_1, critic_2)
        entropy_coef_optimizer, log_entropy_coef = self.setup_entropy_optimizers()
        nb_steps = 0
        tmp_steps = 0
        tmp_steps_avg_reward = 0

        # If entropy_mode is not auto, the entropy coefficient ent_coef will remain fixed
        if self.cfg.algorithm.entropy_mode == "auto":
            # target_entropy is \mathcal{H}_0 in the SAC and aplications paper.
            target_entropy = -np.prod(train_env_agent.action_space.shape).astype(np.float32)

        # Training loop
        logger.message("Training the subspace")
        _training_start_time = time.time()
        while nb_steps < self.cfg.algorithm.n_steps:
            # Execute the agent in the workspace
            if nb_steps > 0:
                train_workspace.zero_grad()
                train_workspace.copy_n_last_steps(1)
                train_agent(
                    train_workspace,
                    t=1,
                    n_steps=self.cfg.algorithm.n_steps_train,
                )
            else:
                train_agent(
                    train_workspace,
                    t=0,
                    n_steps=self.cfg.algorithm.n_steps_train,
                )

            transition_workspace = train_workspace.get_transitions()
            action = transition_workspace["action"]
            nb_steps += action[0].shape[0]
            rb.put(transition_workspace)

            if nb_steps > self.cfg.algorithm.learning_starts:
                # Sample a given number of policies from the subspace
                for _ in range(self.cfg.algorithm.n_policies):
                    # Get a sample from the workspace
                    rb_workspace = rb.get_shuffled(self.cfg.algorithm.batch_size)

                    terminated, reward = rb_workspace["env/terminated", "env/reward"]
                    if entropy_coef_optimizer is not None:
                        ent_coef = torch.exp(log_entropy_coef.detach())

                    # Critic update
                    critic_optimizer.zero_grad()

                    (critic_loss_1, critic_loss_2) = self.compute_critic_loss(
                        reward,
                        ~terminated[1],
                        current_actor,
                        q_agents,
                        target_q_agents,
                        rb_workspace,
                        ent_coef,
                    )

                    bbrl_logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
                    bbrl_logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
                    critic_loss = critic_loss_1 + critic_loss_2
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critic_1.parameters(), self.cfg.algorithm.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        critic_2.parameters(), self.cfg.algorithm.max_grad_norm
                    )
                    critic_optimizer.step()

                    # Actor update
                    if nb_steps % self.cfg.algorithm.policy_update_delay == 0:
                        actor_optimizer.zero_grad()
                        actor_loss = self.compute_actor_loss(
                            ent_coef, current_actor, q_agents, rb_workspace, bbrl_logger, nb_steps
                        )
                        bbrl_logger.add_log("actor_loss", actor_loss, nb_steps)
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(), self.cfg.algorithm.max_grad_norm
                        )
                        actor_optimizer.step()

                        # Entropy coef update part
                        if entropy_coef_optimizer is not None:
                            # See Eq. (17) of the SAC and Applications paper
                            # log. probs have been computed when computing the actor loss
                            action_logprobs_rb = rb_workspace["action_logprobs"].detach()
                            entropy_coef_loss = -(
                                log_entropy_coef.exp() * (action_logprobs_rb + target_entropy)
                            ).mean()
                            entropy_coef_optimizer.zero_grad()
                            entropy_coef_loss.backward()
                            entropy_coef_optimizer.step()
                            bbrl_logger.add_log("entropy_coef_loss", entropy_coef_loss, nb_steps)
                        bbrl_logger.add_log("entropy_coef", ent_coef, nb_steps)

                    # Soft update of target q function
                    if nb_steps % self.cfg.algorithm.target_update_delay == 0:
                        soft_update_params(critic_1, target_critic_1, tau)
                        soft_update_params(critic_2, target_critic_2, tau)
                    # soft_update_params(actor, target_actor, tau)
                n_epochs += 1


            # Evaluating the average subspace reward
            if nb_steps - tmp_steps_avg_reward > self.cfg.algorithm.avg_reward_interval:
                tmp_steps_avg_reward = nb_steps
                n_anchors = eval_agent.agent.agents[1][0].n_anchors
                eval_workspace = Workspace()
                alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([self.cfg.algorithm.n_samples]))
                with torch.no_grad():
                    eval_agent(
                        eval_workspace,
                        t=0,
                        alphas=alphas,
                        stop_variable="env/done",
                    )
                subspace_areas.append(eval_agent.agent.agents[1][1].subspace_area())
                cumulative_rewards =  eval_workspace["env/cumulated_reward"][-1]
                average_cumulative_reward = cumulative_rewards.mean().item()
                max_cumulative_reward = cumulative_rewards.max().item()

                if average_cumulative_reward > best_average_subspace_reward:
                    best_average_subspace_reward = average_cumulative_reward

                if max_cumulative_reward > best_max_subspace_reward:
                    best_max_subspace_reward = max_cumulative_reward

                average_subspace_rewards.append(average_cumulative_reward)
                max_subspace_rewards.append(max_cumulative_reward)
                n_steps_average_rewards.append(nb_steps)
                logger.message(f"After {nb_steps} steps: average subspace reward = {average_cumulative_reward:.02f} (best = {best_average_subspace_reward:.02f}), maximum subspace reward = {max_cumulative_reward:.02f} (best = {best_max_subspace_reward:.02f})")

            # Evaluate
            if nb_steps - tmp_steps > self.cfg.algorithm.eval_interval:
                tmp_steps = nb_steps
                eval_workspace = Workspace()  # Used for evaluation
                eval_agent(
                    eval_workspace,
                    t=0,
                    stop_variable="env/done",
                )
                rewards = eval_workspace["env/cumulated_reward"][-1]
                mean = rewards.mean()
                bbrl_logger.log_reward_losses(rewards, nb_steps)

                if mean > best_reward:
                    best_reward = mean

                logger.message(f"After {nb_steps} steps: reward = {mean:.02f} (best = {best_reward:.02f})")

                if self.cfg.save_best and best_reward == mean:
                    save_best(
                        actor, self.cfg.env_name, mean, "./sac_best_agents/", "sac"
                    )

                if visualizer is not None:
                    info["anchors_similarities"] = current_actor.agent.get_similarities()
                    info["subspace_area"] = current_actor.agent.subspace_area()
                    visualizer.plot_subspace(
                        TemporalAgent(Agents(copy.deepcopy(eval_env_agent), copy.deepcopy(actor))), logger, info, nb_steps
                    )
                    visualizer.plot_reward_curves(
                        logger, self.cfg.algorithm.anticollapse_coef, self.cfg.algorithm.n_samples, n_steps_average_rewards, average_subspace_rewards, max_subspace_rewards, subspace_areas, step=nb_steps
                    )

        logger.message("Training ended")
        logger.message("Time elapsed: " + str(round(time.time() - _training_start_time, 0)) + " sec")

        info["replay_buffer"] = rb
        r = {"n_epochs": n_epochs, "training_time": time.time() - _training_start_time}
    
        # Plot the reward curves at the end of the training
        visualizer.plot_reward_curves(
            logger, self.cfg.algorithm.anticollapse_coef, self.cfg.algorithm.n_samples, n_steps_average_rewards, average_subspace_rewards, max_subspace_rewards, subspace_areas
        )
        return r, actor, critic_1, critic_2, info


    def load_best(self, best_filename):
        best_agent = torch.load(best_filename)
        return best_agent