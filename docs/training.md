# Training a Subspace of Policies

Training a subspace of policies is quite different from getting one optimal policy as we strive to get a near infinite amount of well-performing policies.

It also greatly depends on the task at hand. Indeed, during the initial task, we have to find a specific number of anchors at the same time, while the following tasks only require to find a single anchor, added before the training process. In this case, the previously well-defined anchors are frozen.


## Overview

At a given time, the anchors should yield great results, and most policies sampled within the anchors' convex hull should also perform well.

In order to leverage traditional training methods, we thus first sample $n_{anchors}$ weights $(w_i)_i$ using a Dirichlet distribution. A policy resulting from the linear combination of the anchors weighted by the $w_i$ is then used in the same way as a regular training process.

The weights are resampled when an episode is over, or every `repeat_alpha` timesteps. The weights can also be changed only when the current cumulated rewards are not satisfying enough compared to the top K rewards of the replay buffer, where K depends on the `refresh_rate` and `buffer_size` hyperparameters.

The training phase subsequently frequently updates the parameters of the free anchors (as opposed to the frozen ones), which also updates the current sampled policy as seen in Figure 1. It is important to note that if `resampling_policy = True`, then a new policy is sampled specifically for the anchors' updates.

![Subspace of Policies](assets/subspace_of_policies_training.jpg)

**Figure 1.** *Policy update of the whole subspace*

Regarding the use of the actor and critic losses, they are actually almost identical to a regular training process. The only difference is that the actor loss leverages an anticollapse term.


## Preventing the subspace from collapsing

In order to prevent the collapse of the subspace of policies, i.e. the convergence towards a single optimal policy (all anchors are the same), it is mandatory for **the anchors to be different enough**. Let's note $u$ and $v$ the parameters vectors of two anchors. There are two main ways to measure the pairwise similarities of the subspace anchors.

1. **Cosine Similarity** 
$$cos(u,v) = \frac{u\cdot v}{||u|| ||v||}$$

2. **Euclidean Distance**
$$L_2(u,v) = ||u-v||_2$$

Here, we only consider the pairwise cosine similarities, and add their sum as a penalty term for the actor loss. The sum is weighted by an **anticollapse coefficient**: the greater it is, the more different the anchors should be.


## Best policy estimation

In order to evaluate a subspace, it is fundamental to consider how we can find the best performing policy for the current task, among the infinite amount of policies at hand.

The most straightforward method to estimate the best policy of the subspace, is to sample $n_{samples}$ policies (derived from a sampled Dirichlet distribution) and evaluate all of them over an episode, for ranking purposes. As it can be a very long process, we only limit the evaluation environment's rollout to a few steps called validation steps.

A shortcut could be to use the critic's Q values. Given some sampled policies, we estimate a $n\_{samples}$ Q-values with the critic. The sampled policy with the largest Q-values in average is then taken as the best policy of the subspace for the current task.


## Multitask Scenarios

When dealing with a sequence of tasks, a new anchor (policy) is added to the subspace at the beginning of a new task. However, having the subspace grow linearly drastically increases the memory usage, as more policies are stored. Instead, we determine which anchors bring relevant information to our subspace.


### Dirichlet Distributions Weights

In multitask scenarios, the weights of the distributions are equally sampled on the former subspace (without the last anchor added) and on the new subspace. Sampling both on the former and new subspaces allows to find more well-performing policies, as it easier to find good policies on the former, already trained, subspace than it is on the newer subspace.


### Pruning or extending the subspace

To decide whether a new anchors should be kept, it is crucial to separate the initial subspace (before the anchor is added) from the new subspace. Keep in mind that initially, for the first training task, there is no notion of former subspace.

In this case, we seek to estimate the best $\frac{n_{samples}}{2}$ policy of the current subspace using the Q-values method. The same goes for the former subspace. These $n_{samples}$ policies are evaluated over a few validation steps, and ranked based on their average cumulative rewards.

The subspace is pruned, i.e. the last added anchors is removed, when the best sampled policy belong to the former subspace with $n_{anchors} - 1$ anchors.

**NB:** *the removal of the anchors can be prevented by setting the prune_subspace hyperparameter to False.*