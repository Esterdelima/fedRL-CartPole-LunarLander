# Federated Deep Reinforcement Learning Framework

This project explores the application of **Federated Learning (FL)** to **Deep Reinforcement Learning (DRL)** by enabling multiple agents to train in parallel across different environments, sharing only model parameters. The aggregation is done using the **Federated Averaging (FedAvg)** algorithm.

The goal is to assess whether federated training improves sample efficiency, performance, and generalization compared to single-agent training.

---

## 🚀 **Environments Supported**
- ✅ **CartPole-v1** (Simple balance control)
- ✅ **LunarLander-v2** (2D landing problem)
- ✅ **Super Mario Bros** (Multi-level platformer — using worlds 1-1 to 1-4)

---

## 🧠 **Deep Reinforcement Learning Algorithms**
- ✔️ **Deep Q-Network (DQN)**
- ✔️ **Double DQN (DDQN)**

---

## 🏗️ **Project Structure**
```plaintext
├── pytorch/               → Core modules (Agent, Federator, QNetwork, Buffer, Env)
├── results/               → Training logs, models, rewards, and plots
├── main-cart.py           → Federated training on CartPole
├── main-lun.py            → Federated training on LunarLander
├── single-agent-cart.py   → Single-agent training on CartPole
├── single-agent-lun.py    → Single-agent training on LunarLander
├── Mario.ipynb            → Testing and evaluating federated models on Mario
├── requirements.txt       → Dependencies
├── README.md              → Project description and usage
