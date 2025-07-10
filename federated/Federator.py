import os
import numpy as np
import torch
from tqdm import tqdm

from federated.Agent import Agent
from federated.QNetwork import FCQ
from federated.ReplayBuffer import ReplayBuffer


class Federator:
    def __init__(self, n_agents, update_rate, env_fn, Qnet, buffer, net_args,
             max_epsilon, min_epsilon, decay_steps, gamma,
             target_update_rate, min_buffer, path="./",
             aggregation="fedavg", mu=0.01, use_double=False) -> None:


        self.env = env_fn[0]() if isinstance(env_fn, list) else env_fn()
        self.n_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape

        self.update_rate = update_rate
        self.n_agents = n_agents
        self.path = path
        self.aggregation = aggregation
        self.mu = mu

        self.global_agent = Agent(
            id="global",
            env_fn=env_fn[0],
            Qnet=Qnet,
            buffer=buffer,
            net_args=net_args,
            max_epsilon=max_epsilon,
            min_epsilon=min_epsilon,
            decay_steps=decay_steps,
            gamma=gamma,
            target_update_rate=target_update_rate,
            min_buffer=min_buffer,
            path=self.path,
            use_double=use_double
        )

        self.agents = [
            Agent(
                id=i,
                env_fn=env_fn[i] if isinstance(env_fn, list) else env_fn,
                Qnet=Qnet,
                buffer=buffer,
                net_args=net_args,
                max_epsilon=max_epsilon,
                min_epsilon=min_epsilon,
                decay_steps=decay_steps,
                gamma=gamma,
                target_update_rate=target_update_rate,
                min_buffer=min_buffer,
                path=self.path,
                use_double=use_double
            )
            for i in range(n_agents)
        ]

        self.set_local_networks()


    def print_episode_lengths(self):
        for a in self.agents:
            print(f"Agent {a.id}: {a.episode_count} episodes")


    def train(self, n_runs):
        rewards = np.zeros(n_runs)
        for r in tqdm(range(n_runs)):
            scores = []
            for agent in self.agents:
                agent.step(self.update_rate)
                scores.append(agent.get_score() + 1e-5)  # Avoid division by zero

            if self.aggregation == "fedprox":
                self.aggregate_fedprox(scores)
            else:
                self.aggregate_fedavg(scores)

            self.set_local_networks()
            rewards[r] = self.global_agent.evaluate()
        return rewards


    def aggregate_fedavg(self, scores):
        sd_online = self.global_agent.online_net.state_dict()
        sd_target = self.global_agent.target_net.state_dict()

        online_dicts = [agent.online_net.state_dict() for agent in self.agents]
        target_dicts = [agent.target_net.state_dict() for agent in self.agents]

        for key in sd_online:
            sd_online[key] = torch.zeros_like(sd_online[key])
            for i, state_dict in enumerate(online_dicts):
                sd_online[key] += scores[i] * state_dict[key]
            sd_online[key] /= sum(scores)

        for key in sd_target:
            sd_target[key] = torch.zeros_like(sd_target[key])
            for i, state_dict in enumerate(target_dicts):
                sd_target[key] += scores[i] * state_dict[key]
            sd_target[key] /= sum(scores)

        self.global_agent.online_net.load_state_dict(sd_online)
        self.global_agent.target_net.load_state_dict(sd_target)


    def aggregate_fedprox(self, scores):
        sd_online = self.global_agent.online_net.state_dict()
        sd_target = self.global_agent.target_net.state_dict()

        online_dicts = [agent.online_net.state_dict() for agent in self.agents]
        target_dicts = [agent.target_net.state_dict() for agent in self.agents]

        for key in sd_online:
            sd_online[key] = torch.zeros_like(sd_online[key])
            for i, state_dict in enumerate(online_dicts):
                prox_term = self.mu * (state_dict[key] - sd_online[key])
                sd_online[key] += scores[i] * (state_dict[key] - prox_term)
            sd_online[key] /= sum(scores)

        for key in sd_target:
            sd_target[key] = torch.zeros_like(sd_target[key])
            for i, state_dict in enumerate(target_dicts):
                prox_term = self.mu * (state_dict[key] - sd_target[key])
                sd_target[key] += scores[i] * (state_dict[key] - prox_term)
            sd_target[key] /= sum(scores)

        self.global_agent.online_net.load_state_dict(sd_online)
        self.global_agent.target_net.load_state_dict(sd_target)


    def set_local_networks(self):
        for agent in self.agents:
            agent.online_net.load_state_dict(self.global_agent.online_net.state_dict())
            agent.target_net.load_state_dict(self.global_agent.target_net.state_dict())


    def save(self):
        os.makedirs(self.path, exist_ok=True)
        self.global_agent.save()
        for agent in self.agents:
            agent.save()


    def load(self):
        self.global_agent.load()
        for agent in self.agents:
            agent.load()
