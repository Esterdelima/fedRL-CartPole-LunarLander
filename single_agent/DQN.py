import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from federated.QNetwork import FCQ
from federated.ReplayBuffer import ReplayBuffer


class Agent():
    def __init__(self, id, env_fn, Qnet, buffer,
                 net_args,
                 max_epsilon, min_epsilon, decay_steps, gamma,
                 target_update_rate, min_buffer,
                 load=False, path="./",
                 use_double=True) -> None:

        self.id = id
        self.path = os.path.join(path, str(id)) + "/"

        self.env = env_fn()
        self.env_fn = env_fn
        self.n_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.min_buffer = min_buffer
        self.min_epsilon = min_epsilon
        self.epsilon_decay = (max_epsilon - min_epsilon) / decay_steps
        self.gamma = gamma
        self.target_update_rate = target_update_rate
        self.use_double = use_double

        self.buffer = buffer(self.state_shape, self.n_actions,
                             load=load, path=self.path)

        net_args = net_args if net_args is not None else {}

        self.online_net = Qnet(self.state_shape[0], self.n_actions, **net_args).to(self.device)
        self.target_net = Qnet(self.state_shape[0], self.n_actions, **net_args).to(self.device)

        if load:
            self.load()
        else:
            self.update_target_network()
            self.epsilon = max_epsilon
            self.step_count = 0
            self.episode_count = 0
            self.rewards = []

        self.env_state, _ = self.env.reset()
        self.episode_reward = 0


    def step(self, steps):
        for _ in range(steps):
            self.step_count += 1

            action = self.epsilonGreedyPolicy(self.env_state)
            state_p, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.episode_reward += reward

            is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            is_failure = done and not is_truncated
            self.buffer.store(self.env_state, action, reward, state_p, float(is_failure))

            if len(self.buffer) >= self.min_buffer:
                self.update()
                if self.step_count % self.target_update_rate == 0:
                    self.update_target_network()

            self.env_state = state_p
            if done:
                self.episode_count += 1
                self.env_state, _ = self.env.reset()
                self.rewards.append(self.episode_reward)
                self.episode_reward = 0


    def train(self, n_episodes):
        for _ in range(n_episodes):
            self.episode_reward = 0
            state, _ = self.env.reset()

            while True:
                self.step_count += 1
                action = self.epsilonGreedyPolicy(state)
                state_p, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.episode_reward += reward

                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = done and not is_truncated
                self.buffer.store(state, action, reward, state_p, float(is_failure))

                if len(self.buffer) >= self.min_buffer:
                    self.update()
                    if self.step_count % self.target_update_rate == 0:
                        self.update_target_network()

                state = state_p
                if done:
                    self.episode_count += 1
                    self.rewards.append(self.episode_reward)
                    break

        print(f"Agent-{self.id} Episode {self.episode_count} Step {self.step_count} "
              f"Last Reward={self.rewards[-1]}, Average Reward={np.mean(self.rewards)}")


    def evaluate(self):
        rewards = 0
        state, _ = self.env.reset()
        while True:
            action = self.greedyPolicy(state)
            state_p, reward, terminated, truncated, _ = self.env.step(action)
            rewards += reward
            done = terminated or truncated
            if done:
                break
            state = state_p
        return rewards


    def get_score(self):
        return np.mean(self.rewards[-5:]) if len(self.rewards) >= 5 else np.mean(self.rewards)


    def update(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        states, actions, rewards, states_p, is_terminals = self.buffer.sample()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        states_p = torch.tensor(states_p, dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(is_terminals).to(self.device)

        td_estimate = self.online_net(states).gather(1, actions)

        if self.use_double:
            next_actions = self.online_net(states_p).argmax(1, keepdim=True)
            with torch.no_grad():
                q_states_p = self.target_net(states_p)
            q_state_p_action_p = q_states_p.gather(1, next_actions)
        else:
            with torch.no_grad():
                q_state_p_action_p = self.target_net(states_p).max(1, keepdim=True)[0]

        td_target = rewards + (1 - is_terminals) * self.gamma * q_state_p_action_p

        td_error = td_estimate - td_target
        loss = td_error.pow(2).mean()

        self.online_net.optimize(loss)
        self.update_epsilon()


    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon,
                           self.epsilon - self.epsilon_decay)


    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def epsilonGreedyPolicy(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.online_net(state_tensor).argmax().item()
        return action


    def greedyPolicy(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.target_net(state_tensor).argmax().item()
        return action


    def load(self):
        try:
            with open(self.path + "step_count.pkl", 'rb') as f:
                self.step_count = pickle.load(f)
            with open(self.path + "episode_count.pkl", 'rb') as f:
                self.episode_count = pickle.load(f)
            with open(self.path + "rewards.pkl", 'rb') as f:
                self.rewards = pickle.load(f)
            with open(self.path + "epsilon.pkl", 'rb') as f:
                self.epsilon = pickle.load(f)
            self.online_net.load_state_dict(torch.load(self.path + "online_net.pt",
                                                       map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load(self.path + "target_net.pt",
                                                       map_location=torch.device(self.device)))
            self.buffer.load(self.path)
        except FileNotFoundError:
            print(f"No saved state found at {self.path}, starting fresh.")


    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.buffer.save(self.path)
        with open(self.path + "step_count.pkl", "wb") as f:
            pickle.dump(self.step_count, f)
        with open(self.path + "episode_count.pkl", "wb") as f:
            pickle.dump(self.episode_count, f)
        with open(self.path + "rewards.pkl", "wb") as f:
            pickle.dump(self.rewards, f)
        with open(self.path + "epsilon.pkl", "wb") as f:
            pickle.dump(self.epsilon, f)
        torch.save(self.online_net.state_dict(), self.path + "online_net.pt")
        torch.save(self.target_net.state_dict(), self.path + "target_net.pt")
        print(f"Agent-{self.id} state saved to {self.path}")