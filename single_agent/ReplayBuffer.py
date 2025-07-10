import numpy as np
import pickle
import os


class ReplayBuffer:
    def __init__(self, state_shape, action_space, 
                 batch_size=64, max_size=50000,
                 load=False, path=None):
        self.next = 0
        self.size = 0
        self.max_size = max_size
        self.batch_size = batch_size

        self.states = np.empty((max_size, *state_shape), dtype=np.float32)
        self.actions = np.empty((max_size, 1), dtype=np.int64)
        self.rewards = np.empty((max_size,), dtype=np.float32)
        self.states_p = np.empty((max_size, *state_shape), dtype=np.float32)
        self.is_terminals = np.empty((max_size,), dtype=np.float32)

        if load and path is not None:
            self.load(path)


    def __len__(self):
        return self.size
    
    def store(self, state, action, reward, state_p, is_terminal):
        self.states[self.next] = state
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.states_p[self.next] = state_p
        self.is_terminals[self.next] = is_terminal

        self.next = (self.next + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.states_p[indices],
            self.is_terminals[indices],
        )

    def clear(self):
        self.next = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.states_p.fill(0)
        self.is_terminals.fill(0)

    def save(self, path="./"):
        os.makedirs(path, exist_ok=True)
        data = {
            "next": self.next,
            "size": self.size,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "states_p": self.states_p,
            "is_terminals": self.is_terminals,
        }
        with open(os.path.join(path, "replay_buffer.pkl"), "wb") as f:
            pickle.dump(data, f)

    def load(self, path="./"):
        try:
            with open(os.path.join(path, "replay_buffer.pkl"), "rb") as f:
                data = pickle.load(f)
            self.next = data["next"]
            self.size = data["size"]
            self.states = data["states"]
            self.actions = data["actions"]
            self.rewards = data["rewards"]
            self.states_p = data["states_p"]
            self.is_terminals = data["is_terminals"]
        except FileNotFoundError:
            print(f"No replay buffer found at {path}, starting with empty buffer.")