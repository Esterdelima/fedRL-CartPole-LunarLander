import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from single_agent.DQN import Agent
from single_agent.QNetwork import FCQ
from single_agent.ReplayBuffer import ReplayBuffer

if __name__ == "__main__":

    results_path = "./results/lunarlander/single/"
    os.makedirs(results_path, exist_ok=True)

    # 🔧 Hiperparâmetros de Teste
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

    # 🚀 Inicializa o agente
    agent = Agent(id="single", path=results_path, use_double=True, **args)

    # 🏃‍♂️ Treina por 20 episódios (modo teste)
    agent.train(600)

    # 💾 Salva modelo e buffer
    agent.save()

    print(f"✅ Total episodes treinados: {agent.episode_count}")

    # 💾 Salva recompensas em numpy
    np.save(os.path.join(results_path, 'rewards.npy'), agent.rewards)

    # 📊 Gera o gráfico de recompensas
    plt.figure(figsize=(10, 6))
    plt.plot(agent.rewards, label="Training Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Single Agent - LunarLander")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_path, "training_rewards.png"))
    plt.show()

    print(f"✅ Resultados salvos em {results_path}")
