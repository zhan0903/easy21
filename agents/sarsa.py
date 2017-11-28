import pickle,random
from lib import Policy,Step,mse,State
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

    for episode in range(1,num_episodes):
        # initialize env
        state1 = State()
        state1.dealercard = random.randint(1,10)
        state1.playersum = random.randint(1,10)
        # eligibility traces
        E = np.zeros((10,21,2))
        while state1 != "terminal":
        #    print("state in TD0:",state1.dealercard-1,state1.playersum-1)
            action1 = Policy(value,counter,state1)
        #    print("state in TD1:",state1.dealercard-1,state1.playersum-1)
            state2,reward= Step(state1,action1)

            #print("state2 in TD2:",state2.dealercard-1,state2.playersum-1)
            idx1 = (state1.dealercard-1,state1.playersum-1,action1)
        #    print("idx1:",idx1)
            Q1 = value[idx1]

            if state2 == "terminal":
                Q2 = 0.0
            else:
                #print("state2 in TD2:",state2.dealercard-1,state2.playersum-1)
                action2 = Policy(value,counter,state2)
                idx2 = (state2.dealercard-1, state2.playersum-1, action2)
                #print("idx2:",idx2)
                Q2 = value[idx2]

            counter[idx1] += 1
            E[idx1] += 1

            alpha = 1.0 / counter[idx1]
            delta = reward + GAMMA * Q2 - Q1

            value += alpha * delta * E
            #print("E,GAMMA,lmbd",E,GAMMA,lmbd)
            E *= GAMMA*lmbd
            #print("E,GAMMA,lmbd,delta,alpha",E,GAMMA,lmbd,delta,alpha)

            state1 = state2

        #print("in TD.py:,epsiode,value:",episode,value,opt_value)
        error_history.append((episode, mse(value, opt_value)))
        #print("in TD.py:,epsiode,mse:",episode,mse(value, opt_value))

    return value,error_history
