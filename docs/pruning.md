# Pruning or Extending the Subspace

When dealing with a sequence of tasks, a new anchor (policy) is added to the subspace at the beginning of a new task. However, having the subspace grow linearly drastically increases the memory usage, as more policies are stored. Instead, we determine which anchors bring relevant information to our subspace.

**NB:** *the removal of the anchors can be prevented by setting the prune_subspace hyperparameter to False.*


## Best policy estimation

First and foremost, to evaluate a subspace, it is fundamental to consider how we can find the most optimal policy for the current task, among the infinite amount of policies at hand.

The most straightforward method to estimate the best policy of the subspace, is to sample $n_{samples}$ policies (derived from a sampled Dirichlet distribution) and evaluate all of them over an episode, for ranking purposes. As it can be a very long process, we only limit the evaluation to a few steps called validation steps.

A shortcut could be to use the critic's Q values. Given some sampled policies, we estimate a $n\_{samples}$ Q-values with the critic. The sampled policy with the largest Q-values in average is then taken as the best policy of the subspace for the current task.


## Deciding to keep the last anchor added or not

To decide whether a new anchors should be kept, it is crucial to separate the initial subspace (before the anchor is added) from the new subspace.

**NB:** initially, for the first training task, there is no notion of former subspace.

In this case, we seek to estimate the best $\frac{n_{samples}}{2}$ policy of the current subspace using the Q-values method. The same goes for the former subspace. These $n_{samples}$ policies are evaluated over a few validation steps, and ranked based on their average cumulative rewards.

The subspace is pruned, i.e. the last added anchors is removed, when the best sampled policy belong to the former subspace with $n_{anchors} - 1$ anchors.