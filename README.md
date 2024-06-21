# BBRL for Continuous Reinforcement Learning

This library is greatly inspired from [SaLinA](https://github.com/facebookresearch/salina) and its Continuous Reinforcement Learning component [salina_cl](https://github.com/facebookresearch/salina/tree/main/salina_cl). It also heavily relies on [BBRL](https://github.com/osigaud/bbrl) and its algorithms library [bbrl_algos](https://github.com/osigaud/bbrl_algos).

It aims to re-implement the work from Jean-Baptiste GAYA on Policy Subspaces.


## TODO

- Adapt the SAC algorithm from `bbrl_algos`: encapsulation, arguments, return values (number of interactions, number of epochs, replay buffer)
- Adapt the AlphaSearch class to leverage two critics instead of a single one
- Edit the SAC configuration file to match the hyperparameters used in `bbrl`