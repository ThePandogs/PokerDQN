import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def add(self, priority, data):
        tree_index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_index, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = 2 * parent_index + 2
            if left_child_index >= len(self.tree):
                leaf_index = parent_index - self.capacity + 1
                return leaf_index, self.tree[parent_index], self.data[leaf_index]
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(size)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.size = size
        self.frame = 0

    def add(self, experience, priority):
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        self.frame += 1
        self.beta = min(1.0, self.beta_start + self.frame / self.beta_frames * (1.0 - self.beta_start))
        batch = []
        indices = []
        priorities = []
        segment = self.tree.tree[0] / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(index)
            priorities.append(priority)
        weights = np.array(priorities) ** (-self.beta)
        weights /= weights.max()
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx + self.tree.capacity - 1, priority)

    def get_size(self):
        return self.tree.size  # Devuelve el número de elementos en el buffer
