import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

from federated.Federator import Federator
from single_agent.DQN import Agent
from single_agent.QNetwork import FCQ
from single_agent.ReplayBuffer import ReplayBuffer
from visualize_agent_gif import visualize_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":

    results_path = "./results/lunarlander/"
    os.makedirs(results_path, exist_ok=True)

    # args = {
    #     "env_fn": lambda: gym.make("LunarLander-v2"),
    #     "Qnet": FCQ,
    #     "buffer": ReplayBuffer,

    #     "net_args": {
    #         "hidden_layers": (512, 256, 128),
    #         "activation_fn": torch.nn.functional.relu,
    #         "optimizer": torch.optim.Adam,
    #         "learning_rate": 0.0005,
    #     },

    #     "max_epsilon": 1.0,
    #     "min_epsilon": 0.1,
    #     "decay_steps": 2000,
    #     "gamma": 0.99,
    #     "target_update_rate": 100,
    #     "min_buffer": 64
    # }

    # n_runs = 350
    # n_agents = 3
    # n_iterations = 5
    # update_rate = 300
    
    # -------- Test (leve e rápido) -------------------------
    args = {
        "env_fn": lambda: gym.make("LunarLander-v2", render_mode="rgb_array"),
        "Qnet": FCQ,
        "buffer": ReplayBuffer,
        "net_args": {
            "hidden_layers": (128, 64),
            "activation_fn": torch.nn.functional.relu,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.0003,
        },
        "max_epsilon": 1.0,
        "min_epsilon": 0.1,
        "decay_steps": 8000,
        "gamma": 0.99,
        "target_update_rate": 20,
        "min_buffer": 64,
    }

    n_runs = 600
    n_agents = 3
    n_iterations = 3
    update_rate = 50





    # --------- Federated Training ----------
    all_fed_rewards = {}
    aggregations = ["fedavg", "fedprox"]
    for agg in aggregations:
        fed_rewards = np.zeros(n_runs)
        for i in range(n_iterations):
            fed = Federator(
                n_agents=n_agents,
                update_rate=update_rate,
                path=os.path.join(results_path, f"federated_{agg}/"),
                aggregation=agg,
                mu=0.1 if agg == "fedprox" else 0.0,
                use_double=True, 
                **args
            )
            fed_rewards += fed.train(n_runs)
            fed.save()

        fed_rewards /= n_iterations
        all_fed_rewards[agg] = fed_rewards
        np.save(os.path.join(results_path, f"{agg}_rewards.npy"), fed_rewards)
        print(f"Federated {agg} rewards saved.")


    # --------- Single-Agent Training ----------
    single_rewards = np.zeros(n_runs)
    for i in range(n_iterations):
        ag = Agent(id="single", path=results_path + "single/", use_double=True, **args)
        for r in tqdm(range(n_runs)):
            ag.step(update_rate)
            single_rewards[r] += ag.evaluate()
        ag.save()

    single_rewards /= n_iterations
    np.save(os.path.join(results_path, 'single_rewards.npy'), single_rewards)


    # --------- Plotting ----------
    plt.figure(figsize=(10,6))
    for agg in aggregations:
        plt.plot(all_fed_rewards[agg], label=f"Federated ({agg})")
    plt.plot(single_rewards, label="Single Agent", color="black")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Federated vs Single Agent on LunarLander")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_path, "comparison_plot.png"))
    plt.show()

    # --------- Visualization ----------
    # Exemplo: visualiza o global agent do último federated treinado
    visualize_agent(fed.global_agent, output_path=os.path.join(results_path, "federated_render.gif"))

    # Exemplo: visualiza o single agent
    visualize_agent(ag, output_path=os.path.join(results_path, "single_render.gif"))