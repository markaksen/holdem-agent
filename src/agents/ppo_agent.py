"""
Implementation of an RLcard agent using the Proximal Policy Optimization method
"""


class PPOAgent(object):
    """A PPO RL Agent in the RLCard environment"""
    
    def __init__(self, train_every):
        self.train_every = train_every
        self.memory = []
        self.total_t = 0

    @property
    def use_raw(self):
        raise NotImplementedError
    
    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def feed_memory(self, state, action, reward, next_state, done):
        '''Store the experience'''
        self.memory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

    def train(self):
        ''' Train the model'''
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