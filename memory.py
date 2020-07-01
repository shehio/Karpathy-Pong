class Memory:

    def __init__(self):
        self.dlogps = []
        self.hidden_layers = []
        self.rewards = []
        self.states = []

        # PPO additional fields
        self.entropies = []
        self.state_values = []
        self.actions = []
