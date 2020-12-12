import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('blackjack')
env.set_agents([RandomAgent(action_num=env.action_num)])

trajectories, payoffs = env.run()
print("Done!")