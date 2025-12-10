#from typing import list
    
class RolloutBuffer:
    """
    collect data to update the ppo policy agent.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.logprobs = [] # for policy gradient
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    