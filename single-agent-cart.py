import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from single_agent.DQN import Agent
from single_agent.QNetwork import FCQ
from single_agent.ReplayBuffer import ReplayBuffer


if __name__ == "__main__":

    # 📂 Caminho onde os resultados serão salvos
    results_path = "./results/cartpole/single/"
    os.makedirs(results_path, exist_ok=True)
    
    use_double = False  

    # 🔧 Hiperparâmetros
    args = {
        "env_fn": lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
        "Qnet": FCQ,
        "buffer": ReplayBuffer,

        # "net_args": {
        #     "hidden_layers": (64, 64),  # Leve e eficiente
        #     "activation_fn": torch.nn.functional.relu,
        #     "optimizer": torch.optim.Adam,
        #     "learning_rate": 0.0005,
        # },

        # "max_epsilon": 1.0,
        # "min_epsilon": 0.1,
        # "decay_steps": 5000,
        # "gamma": 0.99,
        # "target_update_rate": 15,
        # "min_buffer": 64
        
        # -------------- Test -----------------
        "net_args": {
            "hidden_layers": (64, 32),
            "activation_fn": torch.nn.functional.relu,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.0005,
        },
        
        "max_epsilon": 1.0,
        "min_epsilon": 0.1,
        "decay_steps": 5000,               
        "gamma": 0.99,                      
        "target_update_rate": 15,            
        "min_buffer": 64,
        "use_double": use_double,
            }

    # 🚀 Instancia o agente
    agent = Agent(id="single", path=results_path, **args)



    # 🏃‍♂️ Treina por 300 episódios (ajuste como quiser)
    agent.train(500) 
    
    # 💾 Salva modelo e buffer
    agent.save()

    print(f"✅ Total episodes treinados: {agent.episode_count}")

    # 💾 Salva também as recompensas em numpy
    np.save(os.path.join(results_path, 'rewards.npy'), agent.rewards)

    # 📊 Plotando os resultados
    plt.figure(figsize=(10,6))
    plt.plot(agent.rewards, label="Training Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Single Agent - CartPole")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_path, "training_rewards.png"))
    plt.show()

    print(f"✅ Resultados salvos em {results_path}")
