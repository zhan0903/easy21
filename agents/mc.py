from environment import Generate_epsiode
import numpy as np

def mc_learn(num_episodes):
    #initialize
    value = np.zeros((10,21,2))
    counter = np.zeros((10,21,2))
    totalreward = 0

    for i in range(1,num_episodes+1):
        #1 generate a epsiode using policy pi
        epsiode,totalreward = Generate_epsiode(value,counter)
        #2 update the value function based on the generated epsiode
        #Incremental Implementation
        for dealercard, playersum,action in epsiode:
            #a means step
            idx = dealercard-1, playersum-1, action
            #counter[idx] += 1
            a = 1.0 / counter[idx]
            g = totalreward
            value[idx] = value[idx] + a*(g - value[idx])

    return value
