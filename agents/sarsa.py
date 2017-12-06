import pickle,random
from environment import Epsilon_greedy_policy,Step,mse,State
import numpy as np

GAMMA = 1.0
#num_episodes = 20000
#num_episodes = 20000

def Sarsa_lamda_Control(lmbd,opt_value,num_episodes):
    #initialize
    value = np.zeros((10,21,2))
    counter = np.zeros((10,21,2))
    totalreward = 0
    error_history = []

    for episode in range(1,num_episodes+1):
        # initialize env
        state1 = State()
        state1.dealercard = random.randint(1,10)
        state1.playersum = random.randint(1,10)

        E = np.zeros((10,21,2))
        while state1 != "terminal":
            action1 = Epsilon_greedy_policy(value,counter,state1)
            state2,reward= Step(state1,action1)
            idx1 = (state1.dealercard-1,state1.playersum-1,action1)
            Q1 = value[idx1]

            if state2 == "terminal":
                Q2 = 0.0
            else:
                action2 = Policy(value,counter,state2)
                idx2 = (state2.dealercard-1, state2.playersum-1, action2)
                Q2 = value[idx2]

            counter[idx1] += 1
            E[idx1] += 1

            alpha = 1.0 / counter[idx1]
            delta = reward + GAMMA * Q2 - Q1

            value += alpha * delta * E
            E *= GAMMA*lmbd

            state1 = state2

        error_history.append((episode, mse(value, opt_value)))

    return value,error_history
