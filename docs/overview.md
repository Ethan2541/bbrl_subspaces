# Overview

## Outlook

This page is the main entry point to the BBRL Subspaces documentation. We expand the BBRL library with the notion of Subspaces of Policies. Behind this core concept, we get an infinite amount of policies, that leverage various different features, while only storing a few ones. Therefore, For a given task, we have more chances to find at least one efficient policy at test time.

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

A possible scenario using these task is `Normal -> Hugefoot -> Moon -> Rainfall`. Most often, scenarios are carefully designed to achieve a specific purpose. Depending on their goal, we can derive a few types of scenarios:

- **Forgetting Scenarios:** a single policy tends to forget the former task when learning a new one.

- **Transfer Scenarios:** if the transfer is positive, a single policy has less difficulties to learn a new task after having learnt the former one, rather than learning it from scratch. Instead, if the transfer is negative, a policy has a hard time learning a new task after having learnt the former one.

- **Robustness Scenarios:** alternate between a normal task and a very distractive task that disturbs the whole learning process of a single policy. For example, when inverting the actions (i.e. multiplying by a -1 factor), well-performing policies actually struggle to recover good performances: the final average reward actually decreases.

- **Compositional Scenarios:** present two first tasks that will be useful to learn the last one, which is a combination of the particularities of the two first tasks. For instance, if the first task is `Moon` and the second one is `Tinyfeet`, the last one will combine moon’s gravity and feet morphological changes. Before learning the last task, a distractive task can be set to disturb the forward transfer.

- **Humanoid Scenarios:** additional scenarios built with the challenging environment `Humanoid` to test the subspace method in higher dimensions.


### Subspace of Policies

A $n$-dimensional subspace of policies is defined as a convex hull delimited by $n$ different policies called **anchors**. Following this definition, a policy from the subspace is a linear combination of these anchors weighted by a sampled **Dirichlet distributions**. Hence, a subspace actually represents an infinity of policies. Compared to single policy training where we aim to get the most optimal results with the policy, subspace training strives to get the largest subspace (i.e. the anchors' parameters should be as pairwise different as possible) with almost only well-performing policies.

**NB:** a subspace of policies with 2 anchors is called a **Line of Policies**.


## Algorithm

First, dealing with subspaces of policies requires to implement agents specifically designed to encompass Dirichlet distributions and multiple anchors' outputs. These agents are described [here](./agents.md). Secondly, a subspace of policies is trained on a scenario, sequentially, one task at a time. More details can be found [here](./training.md).


## Visualization

In addition to the logs, to get a better understanding of the results produced by the subspace, we propose visualization tools. They allow to plot the average and maximum reward curves over time, and also get a global vision of the performances of the policies sampled in the subspace. More details about these can be found [here](./visualization.md).