import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# 📁 Configurações
ROOT_RESULTS = "./results/"
ENVIRONMENTS = ["cartpole", "lunarlander"]
AGGREGATIONS = ["fedavg", "fedprox"]
AGENTS = ['0', '1', 'global']

# 🗂️ Lista para armazenar dados
all_data = []

# 🚀 Percorre todos os ambientes e estratégias
for env in ENVIRONMENTS:
    base_path = os.path.join(ROOT_RESULTS, env)

    for agg in AGGREGATIONS:
        # 1. Recompensas Agregadas (npy)
        reward_file = os.path.join(base_path, f"{agg}_rewards.npy")
        if os.path.exists(reward_file):
            rewards = np.load(reward_file)
            for episode, reward in enumerate(rewards, start=1):
                all_data.append({
                    "Environment": env,
                    "Strategy": agg,
                    "Agent": "Federated_Aggregated",
                    "Episode": episode,
                    "Reward": reward,
                    "RunID": "main",
                    "Timestamp": datetime.now().isoformat()
                })

        # 2. Recompensas dos agentes federados (pkl)
        for agent_id in AGENTS:
            agent_path = os.path.join(base_path, f"federated_{agg}", agent_id, "rewards.pkl")
            if os.path.exists(agent_path):
                with open(agent_path, 'rb') as f:
                    rewards = pickle.load(f)
                for episode, reward in enumerate(rewards, start=1):
                    all_data.append({
                        "Environment": env,
                        "Strategy": agg,
                        "Agent": f"Federated_{agent_id}",
                        "Episode": episode,
                        "Reward": reward,
                        "RunID": f"{env}_{agg}_{agent_id}",
                        "Timestamp": datetime.now().isoformat()
                    })

    # 3. Recompensas do agente single
    single_dir = os.path.join(base_path, "single", "single")  # ← corrigido aqui
    single_rewards_path = os.path.join(single_dir, "rewards.pkl")

    if os.path.exists(single_rewards_path):
        with open(single_rewards_path, 'rb') as f:
            rewards = pickle.load(f)
        for episode, reward in enumerate(rewards, start=1):
            all_data.append({
                "Environment": env,
                "Strategy": "single",
                "Agent": "Single",
                "Episode": episode,
                "Reward": reward,
                "RunID": f"{env}_single",
                "Timestamp": datetime.now().isoformat()
            })
    else:
        print(f"❌ rewards.pkl do agente single não encontrado em: {single_rewards_path}")

# 📄 Converte para DataFrame e exporta
df = pd.DataFrame(all_data)
csv_path = os.path.join(ROOT_RESULTS, "all_rewards.csv")
df.to_csv(csv_path, index=False)
print(f"✅ CSV final salvo em: {csv_path}")
