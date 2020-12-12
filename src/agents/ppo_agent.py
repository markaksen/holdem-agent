"""
Implementation of an RLcard agent using the Proximal Policy Optimization method
"""


class PPOAgent(object):
    """A PPO RL Agent in the RLCard environment"""
    
    def __init__(self):
        raise NotImplementedError

    @property
    def use_raw(self):
        raise NotImplementedError

    def step(self, state: dict):
        ''' Predict the action given the curent state in gerenerating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        raise NotImplementedError
    
    def eval_step(self, state: dict):
        ''' Predict the action given the current state for evaluation.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        raise NotImplementedError