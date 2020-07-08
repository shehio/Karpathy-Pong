class Memory:

    def __init__(self):
        self.dlogps = []
        self.hidden_layers = []
        self.actual_rewards = []
        self.states = []

        # PPO additional fields
        self.entropies = []
        self.actions = []
        self.new_dlogps = []
        self.predicted_values = []
        self.episode_complete = []
