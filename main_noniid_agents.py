import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from federated.Federator import Federator
from federated.QNetwork import FCQ
from federated.ReplayBuffer import ReplayBuffer
from visualize_agent_gif import visualize_agent

if __name__ == "__main__":
    results_path = "./results/noniid/"
    os.makedirs(results_path, exist_ok=True)

    # 🔧 Hiperparâmetros leves para teste
    net_args = {
        "hidden_layers": (64, 32),
        "activation_fn": torch.nn.functional.relu,
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.001,
    }

    n_runs = 10
    n_agents = 2
    n_iterations = 2
    update_rate = 50

    # 🔮 Ambientes diferentes (não-IID)
    envs = [
        lambda: gym.make("CartPole-v1"),
        lambda: gym.make("LunarLander-v2")
    ]

    aggregations = ["fedavg", "fedprox"]
    for agg in aggregations:
        fed_rewards = np.zeros(n_runs)
        for i in range(n_iterations):
            fed = Federator(
                n_agents=n_agents,
                update_rate=update_rate,
                env_fn=envs,  # ⬆️ aqui o diferencial
                Qnet=FCQ,
                buffer=ReplayBuffer,
                net_args=net_args,
                max_epsilon=1.0,
                min_epsilon=0.1,
                decay_steps=200,
                gamma=0.99,
                target_update_rate=100,
                min_buffer=16,
                path=os.path.join(results_path, f"{agg}_noniid/"),
                aggregation=agg,
                mu=0.1 if agg == "fedprox" else 0.0,
                use_double=True

            )
            fed_rewards += fed.train(n_runs)
            fed.save()

        fed_rewards /= n_iterations
        np.save(os.path.join(results_path, f"{agg}_rewards.npy"), fed_rewards)

    # 📈 Plot simples
    plt.figure(figsize=(10,6))
    for agg in aggregations:
        rewards = np.load(os.path.join(results_path, f"{agg}_rewards.npy"))
        plt.plot(rewards, label=agg)

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Federated Learning com Ambientes Não-IID")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_path, "comparison_noniid.png"))
    plt.show()
    
    visualize_agent(fed.global_agent, save_path="./results/noniid/global_render.gif", render_mode="rgb_array")

