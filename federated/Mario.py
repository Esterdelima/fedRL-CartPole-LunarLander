import os
import torch
import numpy as np

from Agent import Agent
from single_agent.QNetwork import QNetwork


class Mario(Agent):
    def __init__(self, env_names, env_fn, Qnet=QNetwork, load=False, path=None) -> None:
        self.path = path + "global/"
        self.envs = [env_fn(name) for name in env_names]
        self.n_actions = self.envs[0].action_space.n
        self.state_shape = self.envs[0].observation_space.shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.online_net = Qnet(self.state_shape[0], self.n_actions).to(self.device)
        self.target_net = Qnet(self.state_shape[0], self.n_actions).to(self.device)

        if load:
            self.load()
        else:
            self.update_target_network()


    def load(self):
        self.online_net.load_state_dict(torch.load(self.path + "online_net.pt", 
                                                   map_location=torch.device(self.device)))
        self.target_net.load_state_dict(torch.load(self.path + "target_net.pt", 
                                                   map_location=torch.device(self.device)))


    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(self.online_net.state_dict(), self.path + "online_net.pt")
        torch.save(self.target_net.state_dict(), self.path + "target_net.pt")


    def get_score(self):
        return 1


    def test(self):
        rewards = np.zeros(len(self.envs))
        for i in range(len(self.envs)):
            r = self.evaluate(i)
            rewards[i] = r
        return rewards


    def evaluate(self, i, render=False):
        rewards = 0
        state, _ = self.envs[i].reset()

        while True:
            action = self.greedyPolicy(state)
            state_p, reward, terminated, truncated, _ = self.envs[i].step(action)
            done = terminated or truncated

            if render:
                self.envs[i].render()

            rewards += reward
            if done:
                break
            state = state_p
        return rewards


    def greedyPolicy(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.target_net(state_tensor).argmax().item()
        return action
