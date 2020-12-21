''' Training a PPO Agent on Texas No-Limit Holdem
'''

import csv
import time
import tensorflow as tf
import os

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from agents.ppo_agent import PPOAgent

# Make environment
env = rlcard.make('no-limit-holdem', config={'seed': 0})
eval_env = rlcard.make('no-limit-holdem', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 100
evaluate_num = 1000
episode_num = 100000

# The intial memory size
memory_init_size = 1000
max_buffer_size = 10000

# Train the agent every X steps
train_every = 10

# The paths for saving the logs and learning curves
log_dir = f'./experiments/nolimit_holdem_ppo_result_adv_{evaluate_every}/'

# Set a global seed
set_global_seed(0)

with tf.Session() as sess:

    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = PPOAgent(sess,
                     action_num=env.action_num,
                     train_every=train_every,
                     state_shape=env.state_shape,
                     replay_memory_init_size=memory_init_size,
                     replay_memory_size=max_buffer_size,
                     actor_layers=[64, 64],
                     critic_layers=[64, 64],
                     )
    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Include this line to verify graph not being updated in each iteration. This helps identify memory leaks.
    # Leave uncommented since tf.train.Saver() below is a graph operation.
    # sess.graph.finalize()

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    start_time = time.time()
    for episode in range(episode_num):
        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            if episode > 0:
                current_time = time.time()
                episodes_per_sec = episode / (current_time - start_time)
                remaining_mins = (episode_num - episode) / episodes_per_sec / 60
                print(f"Current Rate: {episodes_per_sec:.2f}, Estimated Time Remaining: {remaining_mins:.2f} mins")
            reward = tournament(eval_env, evaluate_num)[0]
            logger.log_performance(env.timestep, reward)
            with open(os.path.join(log_dir, "perf.csv"), "a+") as fd:
                fieldnames = ['timestep', 'reward']
                writer = csv.DictWriter(fd, fieldnames=fieldnames)
                if episode == 0:
                    writer.writeheader()
                writer.writerow({'timestep': env.timestep, 'reward': reward})
    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('PPO')
    
    # Save model
    save_dir = '../models/nolimit_holdem_ppo_64x64_oldadv_lrsmall'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
    
