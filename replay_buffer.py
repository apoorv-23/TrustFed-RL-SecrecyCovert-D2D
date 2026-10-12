import numpy as np


class ReplayBuffer:

    def __init__(self, capacity: int, state_shape, action_shape=(), action_dtype=np.int64):
        self.capacity = int(capacity)
        self.s = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.ns = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.a = np.zeros((self.capacity, *action_shape), dtype=action_dtype)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.d = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s, a, r, ns, terminal):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, float(terminal)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_many(self, s, a, r, ns):
        n = len(s)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx] = s, a, r, ns, 0.0
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]

    def __len__(self):
        return self.size
