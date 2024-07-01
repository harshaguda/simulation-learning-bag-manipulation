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
### Setting up Isaac Sim on Triton Cluster

Read more instructions about setting up the experiments on cluster [here.](https://harshaguda.notion.site/Isaac-Orbit-on-Triton-60762305bbc244eba68d07bed0c715f6?pvs=4)

Running experiments on cluster, navigate to the folder `triton_scripts`
```shell
sbatch run_experiment.sh experiment_franka_ppo.sh <random_seed>
```
