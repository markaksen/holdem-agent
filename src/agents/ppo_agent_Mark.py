"""
Implementation of an RLcard agent using the Proximal Policy Optimization method
"""
import typing
import tensorflow as tf 
from tensorflow.keras impot backend as K

class PPOAgent(object):
    """A PPO RL Agent in the RLCard environment"""
    
    def __init__(self, sess, train_every, action_num, state_shape, learning_rate=0.00005, gamma=0.9, critic_layers=None, actor_layers=None):
        '''
        Build a PPOAgent for playing RLCard games. Implements an advantage actor-critic method that is trained using the PPO surrogate objective
        Args:
            sess (tf.Session)
            train_every (int)
            action_num (int)
            state_shape (list)
            learning_rate (float)
            gamma (float)
            critic_layers (list)
            actor_layers (list)
        '''
        self.train_every = train_every
        self.use_raw = True
        self.memory = []
        self.total_t = 0
        self.policy = PPOPolicy(
            action_num=action_num,
            state_shape=state_shape,
            learning_rate=learning_rate,
            gamma=gamma,
            critic_layers=critic_layers if critic_layers else [100],
            actor_layers=actor_layers if actor_layers else [100],
        )
    
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
        if (tmp >= 0) and (tmp % self.train_every == 0):
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
        self.policy.update(self.memory)

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


class PPOPolicy(object):
    ''' PPO Policy context and data '''
    def __init__(self, action_num, state_shape, learning_rate, gamma=0.9, critic_layers=[100], actor_layers=[100]):
        self.gamma = gamma  # This is the discount factor
        self.sess = tf.Session()
        # This is our critic network, which estimates value of any given state
        # TODO double check shapes of below (SF - changed from arrays to tuples and from state_shape to state_shape[0])
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, state_shape[0]), name="X")
        self.X_next = tf.placeholder(dtype=tf.float32, shape=(None, state_shape[0]), name="next_X")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=(None), name='reward')
        self.actions = tf.placeholder(dtype=tf.float32, shape=(None), name='action')
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        
        self._build_value_net(layers=critic_layers)
        # This is our actor network, which is defining our policy
        self._build_actor_net(action_num=action_num, layers=actor_layers)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='ppo_adam')
        self.train_op = self.optimizer.minimize(self._loss, global_step=tf.contrib.framework.get_global_step())

    def update(self, sess, state_batch, action_batch, rewards_batch, next_state_batch):
        _, loss = sess.run(
            [self.train_op, self._loss], 
            {
                self.X: state_batch,
                self.actions: action_batch,
                self.rewards:rewards_batch,
                self.X_next: next_state_batch
            }
        )
        return loss
    
    def _build_value_net(self, layers):
        '''Build the value network'''
        """reference https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f"""
        train_X = tf.layers.batch_normalization(self.X, training=self.is_train)
        fc = tf.contrib.layers.flatten(train_X)

        train_X_next = tf.layers.batch_normalization(self.X_next, training=self.is_train)
        fc_next = tf.contrib.layers.flatten(train_X_next)
        for layer_width in layers:
            fc = tf.contrib.layers.fully_connected(fc, layer_width, activation_fn=tf.tanh)
            fc_next = tf.contrib.layers.fully_connected(fc_next, layer_width, activation_fn=tf.tanh)
        value_pred = tf.contrib.layers.fully_connected(fc, 1, activation_fn=None)
        next_value_pred = tf.contrib.layers.fully_connected(fc_next, 1, activation_fn=None)
        advantage = self._advantage(self.rewards, next_value_pred, value_pred)
        self._value_loss = tf.pow(advantage, 2)
        #TODO Need to validate this function works as we expect

    def _build_actor_net(self, action_num, layers):
        '''Build the actor network'''
        fc = tf.layers.batch_normalization(self.X, training=self.is_train)
        for layer_width in layers:
            fc = tf.contrib.layers.fully_connected(fc, layer_width, activation_fn=tf.tanh)
        action_prob = tf.contrib.layers.fully_connected(fc, action_num, activation_fn=None)
        # TODO what is the actor loss? this is prediction error on action taken I think
        self._actor_loss = tf.losses.log_loss(self.actions, action_prob)

    def _loss_clip(self, epsilon=0.2):
        '''Compute the clip loss'''
        # Reference https://arxiv.org/pdf/1707.06347.pdf - equation 7
        # TODO implement L^{CLIP}(theta) - see below link
        # epsilon <- hyper param, 0.2, say
        #
        # Ahat_t <- estimated value of advantage function at time t
        # r_t(theta) <- pi_theta(a_t | s_t) / pi_thetaold(a_t | s_t), where pi_theta is stochastic policy i.e. policy ration
        #
        # # Unclipped objective
        # unclipped <- r_t(theta) * Ahat_t
        # # Clipped objective
        # clipped <- max(min(r_t(theta), 1+epsilon), 1-epsilon) * Ahat_t [tf has clip_by_value, which they use in their implementation]
        # # Take their min so that result is lower bound on unclipped objective
        # return min(unclipped, clipped)

        def loss(y_true, y_pred):
            # https://github.com/Zataomm/cs230_catch5/blob/master/c5ppo.py
        
            newpolicy_probs = K.sum(y_true * y_pred,axis=1)
            old_probs = K.sum(y_true * oldpolicy_probs,axis=1)
            
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(old_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            entropy_loss =  K.mean(-K.sum((y_pred * K.log(y_pred + 1e-10)),axis=1))
            total_loss = actor_loss - entropy_beta *entropy_loss
            return total_loss

        return loss
    
    def _loss(self):
        '''Compute the loss'''
        # TODO This should compute the loss for the overall system using the PPO Loss statement
        # which is a function of actor loss, value loss, the clip, the variance penalty, and the entropy bonus
        # which is described in the PPO Paper here: https://arxiv.org/pdf/1707.06347.pdf in equation 9

        # TODO for OpenAI (PPO paper authors') implementation, consult https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py
        # Adjacent files also useful

        raise NotImplementedError

    def _advantage(self, returns, value_preds, next_value_preds):
        '''Advantage estimator
        Parameters
        ----------
        returns: tf.Tensor
        value_preds: tf.Tensor
        next_value_preds: tf.Tensor

        Returns
        -------
        advantage: tf.Tensor 
        '''
        return returns + (1.0 - self.gamma) * next_value_preds - value_preds

