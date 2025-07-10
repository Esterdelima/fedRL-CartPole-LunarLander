
# 🧠 Aprendizado por Reforço Federado com Deep Q-Networks

Este projeto investiga o uso de **Aprendizado Federado (FL)** aplicado ao **Aprendizado por Reforço Profundo (Deep Reinforcement Learning - DRL)**. A proposta é avaliar se múltiplos agentes podem aprender de forma descentralizada, compartilhando apenas os parâmetros dos modelos, sem transmitir experiências ou dados brutos.

---

## 🎯 Objetivo

Avaliar se o uso de estratégias de agregação como **FedAvg** e **FedProx**:

- Aumenta a eficiência amostral  
- Melhora a performance global  
- Promove generalização  
- Permite reaproveitamento da política aprendida  

A comparação é feita com agentes isolados (single-agent), em cenários homogêneos (IID) e heterogêneos (não-IID).

---

## 📦 Estrutura do Projeto

```bash
├── federated/                 # Módulos federados (Agente, Federator, Buffer, Q-Network)
├── single_agent/              # Implementação para agente isolado (não federado)
├── results/                   # Recompensas, modelos, gráficos e vídeos
├── main-cart.py               # Treinamento federado em CartPole-v1
├── main-lun.py                # Treinamento federado em LunarLander-v2
├── main_noniid_agents.py      # Treinamento federado com ambientes diferentes (não-IID)
├── single-agent-cart.py       # Treinamento single-agent no CartPole
├── single-agent-lun.py        # Treinamento single-agent no LunarLander
├── visualize_agent_gif.py     # Script para renderização e visualização dos agentes
├── requirements.txt           # Dependências do projeto
```

---

## 🌍 Ambientes Suportados

- 🟢 `CartPole-v1` — ambiente simples de equilíbrio de haste  
- 🟢 `LunarLander-v2` — pouso 2D com múltiplos sensores e ações  
- 🔄 `SuperMarioBros` — planejado como trabalho futuro  

---

## 🧠 Algoritmos de RL

- ✅ **Deep Q-Network (DQN)**  
- ✅ **Double DQN (DDQN)**  
- 🔄 *Outros algoritmos como PPO e A3C podem ser integrados futuramente*

---

## ⚙️ Estratégias Federadas

- `FedAvg` — média ponderada dos modelos dos agentes locais  
- `FedProx` — variante com regularização para lidar com ambientes heterogêneos  
- `Single-Agent` — baseline com agente isolado (centralizado)  

---

## 🧪 Como Rodar os Experimentos

> Recomendado: Python 3.10+ com ambiente `conda` ou `venv`.

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

Ou via Conda:

```bash
conda create -n fedrl python=3.10
conda activate fedrl
pip install -r requirements.txt
```

---

### 2. Treinamento Federado

#### CartPole

```bash
python main-cart.py
```

#### LunarLander

```bash
python main-lun.py
```

#### Ambientes Diferentes (não-IID)

```bash
python main_noniid_agents.py
```

---

### 3. Treinamento Isolado (Single-Agent)

#### CartPole

```bash
python single-agent-cart.py
```

#### LunarLander

```bash
python single-agent-lun.py
```

---

### 4. Visualização do Comportamento (GIF/Render)

Use o script:

```bash
python visualize_agent_gif.py
```

> O arquivo `visualize_agent_gif.py` gera um `.gif` do comportamento do agente (global ou local) após o treinamento.

---

## 📈 Resultados

Os resultados são salvos automaticamente em `./results/`:

- `rewards.npy`: recompensas por episódio  
- `comparison_plot.png`: gráfico comparando estratégias  
- `render_global_*.gif`: visualização do comportamento do agente global  

---

## 📌 Trabalhos Futuros

- Adicionar suporte ao ambiente `SuperMarioBros`  
- Integrar algoritmos avançados como PPO e A3C  
- Avaliar reuso de políticas globais via Fine-Tuning  
- Analisar robustez em cenários com agentes adversariais ou com falhas  

---

## 📚 Referências Acadêmicas

Este projeto se baseia em estudos recentes da literatura de FRL, como:

- Zhuo et al. — *Federated Deep Reinforcement Learning*  
- Fan et al. — *Fault-Tolerant Federated RL*  
- Jin et al. — *FRL with Environment Heterogeneity*  
- Qi et al. — *Survey on FRL*  
- Neto et al. — *FRL for IoT*  

---

## 👩‍💻 Autoria

Este projeto foi desenvolvido como parte do trabalho final da disciplina **Aprendizado por Reforço**, com foco em reprodutibilidade, leveza computacional e extensão de estudos do estado da arte.
