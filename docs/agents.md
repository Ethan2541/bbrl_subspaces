# Subspace Agents

Effectively implementing a subspace of policies requires very specific networks, actors and critics.


## LinearSubspace Module

The `LinearSubspace` module allows to create specific hidden layers that can be subdivided into $n_{anchors}$ independent parts. Their respective outputs are then combined with a sum weighted by a sampled Dirichlet distribution.

[INSERT IMAGE]


## Actor

The actor used to decide the next action, is actually a sequence of two distinct agents: `AlphaAgent` and `SubspaceAction`. This sequence is called `SubspaceActionAgent`.


### AlphaAgent

### SubspaceAction


## Critic