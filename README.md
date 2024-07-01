## Cloth Bag Manipulation Learning

It uses NVIDIA-OmniIsaacGymEnvs which is built on NVIDIA Isaac Sim to simulate the environments used in reinforcement learning.

### Getting started

Install NVIDIA-OmniIsaacGymEnvs using the following commands.
```shell
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
cd OmniIsaacGymEnvs
$ISAACSIM_PYTHON_PATH -m pip install -e .
```

Download the deformable tasks.

```shell
git clone https://github.com/harshaguda/cloth-bag-manipulation-learning.git
```

### Run environment
```shell
$ISAACSIM_PYTHON_EXE scripts/random_policy.py task=ClothBagAttach num_envs=2
```
### Train

```shell
$ISAACSIM_PYTHON_EXE scripts/ppo_trainer_headless.py task=ClothBagAttach num_envs=2
```
