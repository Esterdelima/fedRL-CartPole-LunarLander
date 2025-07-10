import imageio
import os
import torch
from PIL import Image
import numpy as np

def visualize_agent(agent, output_path="agent_render.gif", max_steps=500):
    env = agent.env_fn()  # Cria uma nova instância do ambiente
    frames = []

    state, _ = env.reset()
    for _ in range(max_steps):
        # Renderiza frame como RGB array
        frame = env.render()
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = Image.fromarray(frame[0])  # Gymnasium pode retornar tupla

        frames.append(img)

        action = agent.greedyPolicy(state)
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"🎥 GIF salvo em: {output_path}")
