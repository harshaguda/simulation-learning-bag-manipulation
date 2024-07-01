# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import gym
import hydra
from omegaconf import DictConfig
import os
import datetime

import torch
import torch.nn as nn


import omniisaacgymenvs
# from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from utils.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import get_experience
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *  # noqa: F403 F401
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict  # noqa: E501
from utils.task_util import initialize_task


from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
# from skrl.utils import set_seed
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from utils.test import OmniverseIsaacGymWrapper

import sys
sys.stdout = None
import warnings
warnings.filterwarnings("ignore")

# define the model
class CNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20,
                 max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std,
                               max_log_std, reduction)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 )
        self.mean_layer = nn.Linear(32, self.num_actions)
        self.value_layer = nn.Linear(32, 1)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    # def act(self, inputs, role):
    #     return GaussianMixin.act(self, inputs, role)
    def compute(self, inputs, role):
        # permute (samples, width * height * channels) ->
        # (samples, channels, width, height)
        return (self.mean_layer(
            self.net(inputs["states"].view(-1,
                                           *self.observation_space.shape))),
                self.log_std_parameter, {})


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(
            inputs["states"].view(-1,
                                  *self.observation_space.shape)), {}


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    headless = cfg.headless
    # render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras  # noqa: E501
    print(enable_viewport)
    # select kit app file
    experience = get_experience(headless=headless,
                                enable_livestream=cfg.enable_livestream,
                                enable_viewport=enable_viewport,
                                enable_recording=cfg.enable_recording,
                                kit_app=cfg.kit_app)
    episode_length = cfg.task.env.episodeLength
    test = cfg.test
    env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport or cfg.enable_recording,
        experience=experience
    )
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed
    initialize_task(cfg_dict, env)
    # parse experiment directory
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(omniisaacgymenvs.__file__)))
    experiment_dir = os.path.join(module_path, "runs",
                                  cfg.train.params.config.name)
    task_name = cfg.task_name

    directory = "runs/torch/" + task_name
    experiment_name = "{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), "PPO")
    experiment_dir = os.path.join(directory, experiment_name)
    if cfg.enable_recording:
        if cfg.recording_dir == '':
            videos_dir = os.path.join(experiment_dir, "videos")
        else:
            videos_dir = cfg.recording_dir
        video_interval = lambda step: step % episode_length*4 == 0  # noqa: E501 E731
        video_length = episode_length
        env.is_vector_env = True
        if env.metadata is None:
            env.metadata = {"render_modes": ["rgb_array"],
                            "render_fps": cfg.recording_fps}
        else:
            env.metadata["render_modes"] = ["rgb_array"]
            env.metadata["render_fps"] = cfg.recording_fps
        env = OmniverseIsaacGymWrapper(
            env, video_folder=videos_dir, step_trigger=video_interval,
            video_length=video_length
        )
    else:
        env = wrap_env(env)
    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=256,
                          num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    policy = CNN(observation_space=env.observation_space,
                 action_space=env.action_space,
                 device=env.device,
                 clip_actions=True,
                 clip_log_std=True,
                 min_log_std=-20,
                 max_log_std=2,
                 reduction="sum")
    models = {}
    models["policy"] = policy
    models["value"] = Value(env.observation_space, env.action_space, device)

    # configure and instantiate the agent (visit its documentation to see all
    # the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 256#episode_length * 8 #16  # episode_length * 16  # memory_size
    cfg["learning_epochs"] = 16 #episode_length * 4  # episode_length * 8
    cfg["mini_batches"] = 16 #episode_length * 2  # episode_length * 2  # 16 * 512 / 8192
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 30e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 0#episode_length * 512
    cfg["learning_starts"] = 0#(episode_length + 1) * 512
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.01*2
    cfg["value_loss_scale"] = 2.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space,
                                        "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 16
    cfg["experiment"]["checkpoint_interval"] = episode_length * 64
    cfg["experiment"]["directory"] = directory
    cfg["experiment"]["experiment_name"] = experiment_name #"{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), "PPO")
    cfg["experiment"]["wandb"] = True
    cfg["experiment"]["wandb_kwargs"] = {"name": task_name,
                                         "sync_tensorboard": True}
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    
    if test:
        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 1600 * 30, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        agent.load("agent_51200.pt")
        trainer.eval()
    else:
        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 1600 * 40, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        # agent.load("best_agent.pt")
        # start training
        trainer.train()


if __name__ == "__main__":
    parse_hydra_configs()
