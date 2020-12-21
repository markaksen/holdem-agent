"""
Implementation of an RLcard agent using the Proximal Policy Optimization method
"""
import typing
import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class PPOAgent(object):
    """A PPO RL Agent in the RLCard environment"""
    
    def __init__(self, sess, train_every, action_num, state_shape,
                 learning_rate=0.00005, gamma=0.9, critic_layers=None, actor_layers=None,
                 replay_memory_size=20000, replay_memory_init_size=100,
                 # TODO add more? these necessary? and see ppo2.py in OpenAI for docs on what params are. e.g. gamma vs lam
                 cliprange=0.2, vf_coef=0.5, ent_coef=0.0, lam=0.95):
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
        self.sess = sess
        self.train_every = train_every
        self.use_raw = True
        self.memory = []
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.total_t = 0
        self.policy = PPOPolicy(
            action_num=action_num,
            state_shape=state_shape,
            learning_rate=learning_rate,
            gamma=gamma,
            critic_layers=critic_layers if critic_layers else [100],
            actor_layers=actor_layers if actor_layers else [100],
            cliprange=cliprange,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            lam=lam
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
        if len(self.memory) == self.replay_memory_size:
            self.memory.pop(0)
        self.memory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

    def train(self):
        ''' Train the model'''
        # TODO implement the sampling for the memory

        samples = random.sample(self.memory, self.batch_size)
        import pdb; pdb.set_trace()
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample()
        self.policy.update(self.sess, state_batch, action_batch, reward_batch, next_state_batch)

    # step vs. eval_step as per README
        # env.run(is_training=False): Run a complete game and return trajectories and payoffs.
        # The function can be used after the `set_agents` is called. If `is_training` is `True`,
        # it will use `step` function in the agent to play the game. If `is_training` is `False`,
        # `eval_step` will be called instead.

    def step(self, state: dict):
        ''' Predict the action for generating training data
        Args:
            state (dict): current state
        Returns:
            action (int): an action id
        '''
        # Sample according to probs (based on dqn_agent impl)
        state_obs = np.expand_dims(state['obs'], 0)
        A = self.policy.predict(self.sess, state_obs)[0]
        import pdb; pdb.set_trace()
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def eval_step(self, state: dict):
        ''' Predict the action given the current state for evaluation.
        Args:
            state (dict): current state
        Returns:
            action (int): an action id
            probs (list): a list of probabilies
        '''
        # Sample according to probs (based on dqn_agent impl)
        A = self.policy.predict(state['obs']) # todo in dqn this is not self.predict but rather self.estimator.predict
        probs = remove_illegal(A, state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

class PPOPolicy(object):
    ''' PPO Policy context and data '''
    def __init__(self, action_num, state_shape, learning_rate, gamma=0.9, critic_layers=[100], actor_layers=[100],
                 # TODO params, same as above
                 cliprange=0.2, vf_coef=0.5, ent_coef=0.0, lam=0.95):
        self.gamma = gamma  # This is the discount factor
        # This is our critic network, which estimates value of any given state
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, state_shape[0]), name="X")
        self.X_next = tf.placeholder(dtype=tf.float32, shape=(None, state_shape[0]), name="next_X")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=(None), name='reward')
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None), name='action')
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self._build_value_net(layers=critic_layers)
        # This is our actor network, which is defining our policy
        self._build_actor_net(action_num=action_num, layers=actor_layers)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='ppo_adam')

# this part down is based on OpenAI
# already have actions (int or float?), lr, cliprate, rewards
        # self.A = A = tf.placeholder(dtype=tf.int32, shape=(None, state_shape[0]), name="actions") # actions
        # self.ADV = tf.placeholder(tf.float32, [None]) # advantages
        # self.R = R = tf.placeholder(tf.float32, [None]) # returns
        # Keep track of old actor
        self.OLDNEGLOGPAC = tf.placeholder(tf.float32, [None]) # old neg log probs
        # Keep track of old critic
        self.OLDVPRED = tf.placeholder(tf.float32, [None]) # old value preds
        # self.LR = LR = tf.placeholder(tf.float32, []) # learning rate
        # Cliprange
        self.CLIPRANGE = cliprange #= tf.placeholder(tf.float32, [])

        # sf params
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        # Training
        self.train_op = self.optimizer.minimize(self._loss, global_step=tf.contrib.framework.get_global_step())
        # self.train_op = self.optimizer.minimize(self._loss_pg(), global_step=tf.contrib.framework.get_global_step())

    def update(self, sess, state_batch, action_batch, rewards_batch, next_state_batch):
        oldvpred, oldneglogpac = sess.run(
            [self._neg_log_probs_for_taken_actions, self.value_pred],
            {self.X: state_batch, self.X_next: next_state_batch, self.is_train: False}
        )

        _, loss = sess.run(
            [self.train_op, self._loss],
            # [self.train_op, self._loss_pg],
            {
                self.X: state_batch,
                self.actions: action_batch,
                self.rewards: rewards_batch,
                self.X_next: next_state_batch,
                self.is_train: True,
                self.OLDVPRED: oldvpred,
                self.OLDNEGLOGPAC: oldneglogpac
            }
        )
        return loss

    def predict(self, sess, state):
        action_prob = sess.run(
            self.action_probabilities,
            {
                self.X: state,
                self.is_train: False,
            }
        )
        return action_prob

    def _build_value_net(self, layers):
        #TODO need to pass in is_train at all layers
        '''Build the value network'''
        """reference https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f, dqn_agent"""
        # Batch normalization
        train_X = tf.layers.batch_normalization(self.X, training=self.is_train)
        train_X_next = tf.layers.batch_normalization(self.X_next, training=self.is_train)

        # Fully connected layers for X and X_next
        fc = tf.contrib.layers.flatten(train_X)
        fc_next = tf.contrib.layers.flatten(train_X_next)
        for layer_width in layers:
            fc = tf.contrib.layers.fully_connected(fc, layer_width, activation_fn=tf.tanh)
            fc_next = tf.contrib.layers.fully_connected(fc_next, layer_width, activation_fn=tf.tanh)
        self.value_pred = tf.contrib.layers.fully_connected(fc, 1, activation_fn=None)
        self.next_value_pred = tf.contrib.layers.fully_connected(fc_next, 1, activation_fn=None)

        # Calculate advantage and loss
        self.advantage = self._advantage(self.rewards, self.next_value_pred, self.value_pred)
        #TODO Need to validate this function works as we expect
        self._value_loss = tf.reduce_mean(tf.pow(self.advantage, 2))

    def _neg_log_probs_for_taken_actions(self):
        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.action_probabilities)[0]
        action_count = tf.shape(self.action_probabilities)[1]
        gather_indices = tf.range(batch_size) * action_count + self.actions
        action_predictions = tf.gather(tf.reshape(self.action_probabilities, [-1]), gather_indices)
        log_action_probs = tf.negative(tf.math.log(action_predictions))
        return log_action_probs

    def _build_actor_net(self, action_num, layers):
        '''Build the actor network'''
        fc = tf.layers.batch_normalization(self.X, training=self.is_train)
        for layer_width in layers:
            fc = tf.contrib.layers.fully_connected(fc, layer_width, activation_fn=tf.tanh)
        self.action_probabilities = tf.contrib.layers.fully_connected(fc, action_num, activation_fn=None)
        # TODO what is the actor loss? this is prediction error on action taken I think
        # self._actor_loss = tf.losses.log_loss(self.actions, self.action_probabilities)

    def _loss_pg(self):
        """
        L^{PG}(\theta) = \hat{E}[log \pi_{\theta}(a_t | s_t) \hat{A_t}]
        """
        return tf.reduce_mean(self._value_loss)


    def _entropy(self):
        # open AI: return tf.add_n([p.entropy() for p in self.categoricals])
        raise NotImplementedError

    def _loss(self):
        '''Compute the loss'''
        # This should compute the loss for the overall system using the PPO Loss statement
        # which is a function of actor loss, value loss, the clip, the variance penalty, and the entropy bonus
        # which is described in the PPO Paper here: https://arxiv.org/pdf/1707.06347.pdf in equation 9

        # TODO for OpenAI (PPO paper authors') implementation, consult https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py
        # Calculate ratio (pi current policy / pi old policy) - use neg log probabilities
        neglogpac = self._neg_log_probs_for_taken_actions()
        ratio = tf.exp(self.OLDNEGLOGPAC - neglogpac)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        # TODO
        # entropy = tf.reduce_mean(train_model.pd.entropy())
        entropy = 0 #self.entropy()

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = self.value_pred
        vpredclipped = self.OLDVPRED + tf.clip_by_value(vpred - self.OLDVPRED, - self.CLIPRANGE, self.CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - self.rewards)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.rewards)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Defining Loss = - J is equivalent to max J
        pg_losses = -self.advantage * ratio
        pg_losses2 = -self.advantage * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE, 1.0 + self.CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        return loss

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
        return returns + (1.0 - self.gamma * next_value_preds) - value_preds

