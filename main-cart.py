import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

from visualize_agent_gif import visualize_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated.Agent import Agent
from federated.Federator import Federator
from federated.QNetwork import FCQ
from federated.ReplayBuffer import ReplayBuffer


if __name__ == "__main__":

    results_path = "./results/cartpole/"
    os.makedirs(results_path, exist_ok=True)

    # net_args = {
    #     "hidden_layers": (512, 128),
    #     "activation_fn": torch.nn.functional.relu,
    #     "optimizer": torch.optim.Adam,
    #     "learning_rate": 0.0005,
    # }
    
    # n_runs = 2000
    # n_agents = 3
    # n_iterations = 10
    # update_rate = 30
    
    # -------- Test -------------------------
    net_args = {
    "hidden_layers": (64, 32),
    "activation_fn": torch.nn.functional.relu,
    "optimizer": torch.optim.Adam,
    "learning_rate": 0.0005,
    }

    n_runs = 500
    n_iterations = 3
    n_agents = 3
    update_rate = 10
    use_double = True


    # --------- Federated Training ----------
    fed_results = {}
    aggregations = ["fedavg", "fedprox"]

    for agg in aggregations:
        fed_rewards = np.zeros(n_runs)
        for i in range(n_iterations):
            fed = Federator(
                n_agents=n_agents,
                update_rate=update_rate,
                env_fn= lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
                Qnet=FCQ,
                buffer=ReplayBuffer,
                net_args=net_args,
                max_epsilon=1.0,
                min_epsilon=0.1,
                decay_steps=5000,
                gamma=0.99,
                target_update_rate=15,
                min_buffer=64,
                path=os.path.join(results_path, f"federated_{agg}/"),
                aggregation=agg,
                mu=0.1 if agg == "fedprox" else 0.0,
                use_double=use_double,
            )
            fed_rewards += fed.train(n_runs)
            fed.save()

        fed_rewards /= n_iterations
        fed_results[agg] = fed_rewards
        np.save(os.path.join(results_path, f"{agg}_rewards.npy"), fed_rewards)


    # --------- Single-Agent Training ----------
    single_rewards = np.zeros(n_runs)
    for i in range(n_iterations):
        ag = Agent(
            id="single",
            env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
            Qnet=FCQ,
            buffer=ReplayBuffer,
            net_args=net_args,
            max_epsilon=1.0,
            min_epsilon=0.1,
            decay_steps=5000,
            gamma=0.99,
            target_update_rate=15,
            min_buffer=64,
            path=results_path + "single/",
            use_double=use_double,
        )
        for r in tqdm(range(n_runs)):
            ag.step(update_rate)
            single_rewards[r] += ag.evaluate()
        ag.save()

    single_rewards /= n_iterations
    np.save(os.path.join(results_path, 'single_rewards.npy'), single_rewards)


    # --------- Plotting ----------
    plt.figure(figsize=(10,6))
    for agg in fed_results:
        plt.plot(fed_results[agg], label=f"Federated ({agg})")
    plt.plot(single_rewards, color="r", label="Single Agent")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Federated Strategies vs Single Agent on CartPole")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_path, "comparison_plot.png"))
    plt.show()
    
    # --------- Visualization ----------
    # Exemplo: visualiza o global agent do último federated treinado
    visualize_agent(fed.global_agent, output_path=os.path.join(results_path, "federated_render.gif"))

    # Exemplo: visualiza o single agent
    visualize_agent(ag, output_path=os.path.join(results_path, "single_render.gif"))