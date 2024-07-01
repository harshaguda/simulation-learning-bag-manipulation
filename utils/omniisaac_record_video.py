from typing import Any, Optional, Tuple, Callable

import torch

from skrl.envs.wrappers.torch.base import Wrapper
from gym import logger

import gym
import os

from gym.wrappers.monitoring import video_recorder
from skrl.envs.wrappers.torch import OmniverseIsaacGymWrapper

def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

class OmniverseIsaacGymRecordWrapper(OmniverseIsaacGymWrapper):
    def __init__(self, env: Any,
                 video_folder: str,
                episode_trigger: Callable[[int], bool] = None,
                step_trigger: Callable[[int], bool] = None,
                video_length: int = 0,
                name_prefix: str = "rl-video",) -> None:
        """Omniverse Isaac Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Omniverse Isaac Gym environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None
        
        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def run(self, trainer: Optional["omni.isaac.gym.vec_env.vec_env_mt.TrainerMT"] = None) -> None:
        """Run the simulation in the main thread

        This method is valid only for the Omniverse Isaac Gym multi-threaded environments

        :param trainer: Trainer which should implement a ``run`` method that initiates the RL loop on a new thread
        :type trainer: omni.isaac.gym.vec_env.vec_env_mt.TrainerMT, optional
        """
        self._env.run(trainer)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        
        self._obs_dict, reward, terminated, info = self._env.step(actions)
        truncated = info["time_outs"] if "time_outs" in info else torch.zeros_like(terminated)
        
        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if terminated:
                self.episode_id += 1
        elif terminated[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if terminated:
                        self.close_video_recorder()
                elif terminated[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()
        return self._obs_dict["obs"], reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
        super().reset()

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
        self.close_video_recorder()
        

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self._env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)
        
    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def __del__(self):
        self.close_video_recorder()