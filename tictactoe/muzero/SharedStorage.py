import copy

class SharedStorage:
    def __init__(self):
        self._networks = {}  # Stores models at different training steps
        self.latest_step = 0  # Track latest training step

    def latest_network(self):
        """Return the latest neural network."""
        if self.latest_step in self._networks:
            return self._networks[self.latest_step]
        else:
            return None  # No trained network yet

    def save_network(self, step, network):
        """Save the neural network at a specific training step."""
        self._networks[step] = copy.deepcopy(network)
        self.latest_step = step

    def get_network(self, step):
        """Retrieve a specific checkpointed model."""
        return self._networks.get(step, None)
