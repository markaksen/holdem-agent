import pandas as pd
#import os
from matplotlib import pyplot as plt

max_time = 400e3

results = pd.read_csv('../experiments/all_performance.csv')

results_NFSP = results[['timestep1', 'NFSP']].dropna()
results_NFSP = results_NFSP[results_NFSP['timestep1']< max_time]

results_DQN = results[['timestep2', 'DQN']].dropna()
results_DQN = results_DQN[results_DQN['timestep2']<max_time]

results_PPO = results[['timestep3', 'PPO - 1stepAdv']].dropna().iloc[::10,:]
results_PPO = results_PPO[results_PPO['timestep3']<max_time]

# Interpolate results and then apply moving average
times = np.linspace(0,400000, 20)
vals_DQN = np.interp(times, results_DQN['timestep2'], results_DQN['DQN'])
vals_NFSP = np.interp(times, results_NFSP['timestep1'], results_NFSP['NFSP'])
vals_PPO = np.interp(times, results_PPO['timestep3'], results_PPO['PPO - 1stepAdv'])


line1,  = plt.plot(times/1000, vals_DQN)
line2,  = plt.plot(times/1000, vals_NFSP)
line3, = plt.plot(times/1000, vals_PPO)
plt.xlabel('Timestep(1e3)')
plt.ylabel('reward')
plt.title('Learning Curves')
plt.legend((line1, line2, line3), ('DQN', 'NFSP', 'PPO'))
#plt.show()
#plt.show()
plt.savefig('learning_curves.png')