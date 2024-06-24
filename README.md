# BBRL for Continuous Reinforcement Learning

This library is greatly inspired from [SaLinA](https://github.com/facebookresearch/salina) and its Continuous Reinforcement Learning component [salina_cl](https://github.com/facebookresearch/salina/tree/main/salina_cl). It also heavily relies on [BBRL](https://github.com/osigaud/bbrl) and its algorithms library [bbrl_algos](https://github.com/osigaud/bbrl_algos).

It aims to re-implement the work from Jean-Baptiste GAYA on Policy Subspaces.


## Quick Start

* Create and activate a python environment with your favorite tool (e.g. conda or venv)
* Then clone the repository
* Use `pip` to install it with egg-link

```console
conda create bbrl_cl_env
conda activate bbrl_cl_env
git clone https://github.com/Ethan2541/bbrl_cl
cd bbrl_cl
pip install -e .
```


## Usage

You simply need to go to src/bbrl_cl and execute:
```console
python run.py -cn=csp
```

If you want to add a custom scenario / tasks, you can edit the configuration files in `src/bbrl_cl/configs`. You can also change the value of the command line argument `-cn`, which refers to the configuration filename, for further testing.