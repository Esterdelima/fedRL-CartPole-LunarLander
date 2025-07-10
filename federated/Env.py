import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor(),  # Output will be [C, H, W] in [0, 1]
        ])
        obs_shape = (1, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_shape), dtype=np.float32
        )

    def observation(self, observation):
        observation = self.transform(observation)
        return observation


def create_mario_env(env_name="SuperMarioBros-1-1-v0"):
    env = gym_super_mario_bros.make(env_name, apply_api_compatibility=True, render_mode=None)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = TransformObservation(env)
    env = FrameStack(env, num_stack=4)
    return env
