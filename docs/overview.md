# Overview

## Outlook

This page is the main entry point to the BBRL Subspaces documentation. We expand the BBRL library with the notion of Subspaces of Policies. Behind this core concept, the goal is to improve an agent's performances over a wide range of problems despite only learning a few behaviours: the idea is that a new problem can be seen as a combination of well-known problems, to a certain extent. The main interest of such subspaces is that we get an infinite amount of policies while only storing a few ones.

## Main concepts

### Scenario

A scenario is a succession of tasks that all take place in the same [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment. The goal is to train the subspace to adapt to different configurations. As such, two different tasks shoud have different environment parameters. Monotask scenarios are scenarios that only have one task, as opposed to multitask scenarios.

Let's consider the [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) environment where a 2-dimensional robot has to learn how to run forward. We define the following tasks:

<center>

| Task     | Configuration                        |
|----------|--------------------------------------|
| Normal   | Default environment parameters       |
| Hugefeet | Larger foot size                     |
| Moon     | The gravity intensity is set to 0.15 |
| Rainfall | The friction is decreased to 0.4     |

</center>

A possible scenario using these task is `Normal -> Hugefoot -> Moon -> Rainfall`. Most often, scenarios are carefully design to achieve a specific purpose. Depending on their goal, we can derive a few types of scenarios:

- **Forgetting Scenarios:** a single policy tends to forget the former task when learning a new one.

- **Transfer Scenarios:** a single policy has more difficulties to learn a new task after having learned the former one, rather than learning it from scratch.

- **Robustness Scenarios:** alternate between a normal task and a very different distraction task that disturbs the whole learning process of a single policy. While this challenge looks particularly simple from a human perspective (a simple -1 vector applied on the output is fine to find an optimal policy in a continual setting), we figured out that the Fine- tuning policies struggle to recover good performances (the final average reward actually decreases).

- **Compositional Scenarios:** present two first tasks that will be useful to learn the last one, but a very different distraction task is put at the third place to disturb this forward transfer. The last task is indeed a combination of the two first tasks in the sense that it combines their particularities. For example, if the first task is `Moon` and the second one is `Tinyfeet`, the last one will combine moon’s gravity and feet morphological changes.

- **Humanoid Scenarios:** additional scenarios built with the challenging environment `Humanoid` to test the subspace method in higher dimensions.


### Subspace of Policies

A subspace of policies is defined by $n$ different policies called **anchors**. Using **Dirichlet distributions**, we can combine these anchors to produce a new sampled policy. Hence, a subspace actually represents an infinity of policies. Compared to single policy training where we aim to get the most optimal results with the policy, subspace training strives to get the largest subspace (i.e. the anchors' parameters should be as pairwise different as possible) with almost only well-performing policies.

**NB:** a subspace of policies with 2 anchors is called a **Line of Policies**.


## Algorithm

First, dealing with subspaces of policies requires to implement agents specifically designed to encompass Dirichlet distributions and multiple anchors' outputs. These agents are described [here](./agents.md).

Currently, the only training algorithm that has been implemented is the Soft Actor Critic (SAC) algorithm with two critics. A subspace of policies is trained on a scenario, sequentially, one task at a time. More details can be found [here](./training.md).

When training on a task in a multitask scenario, a new anchor is systematically added to the subspace. To spare memory usage, it is important to remove this anchor if it actually underperforms. It is explained more thoroughly [here](./pruning.md).


## Visualization

In addition to the logs, to get a better understanding of the results produced by the subspace, we propose visualization tools. They allow to plot the average and maximum reward curves over time, and also get a global vision of the performances of the policies sampled in the subspace. More details about these can be found [here](./visualization.md).