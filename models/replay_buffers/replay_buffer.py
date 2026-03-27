from skrl.memories.torch import RandomMemory


class ReplayBuffer(RandomMemory):
    def __init__(self, capacity, num_envs, device, **kwargs):
        super().__init__(memory_size=capacity, num_envs=num_envs, device=device, **kwargs)
