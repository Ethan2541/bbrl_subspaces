# Subspace Agents


## Linear Subspace Module

The `LinearSubspace` module allows to create specific hidden layers that can be subdivided into $n_{anchors}$ independent parts. Their respective outputs are then combined with a sum weighted by a sampled Dirichlet distribution.

[INSERT IMAGE]


## Actor

The actor used to decide the next action, is actually a sequence of two distinct agents: `AlphaAgent` and `SubspaceAction`. This sequence is called `SubspaceActionAgent`.


### AlphaAgent

renew the distribution


### SubspaceAction

resampling policy

[INSERT IMAGE]


## Critic

The `AlphaCritic` agent used is actually very similar to standard critics. The only difference is that the Dirichlet distribution of the evaluated policy is given as an input to the network. This allows to better generalize on every policy of the subspace, rather than on a specific sampled policy.