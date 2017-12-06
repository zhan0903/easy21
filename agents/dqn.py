# implement the deep Q-learning networks algorithm

import pickle,random
from environment import (
Step,mse,State,ACTIONS,
DEALER_RANGE,PLAYER_RANGE
)
import numpy as np


GAMMA = 1
LAMBDA = 0
EPSILON = 0.05
ALPHA = 0.01

HIT,STICK = ACTIONS

CUBOID_INTERVALS = {
  "dealer": ((1, 4), (4, 7), (7, 10)),
  "player": ((1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)),
  "action": ((HIT,), (STICK,))
}

FEATS_SHAPE = tuple(
  len(CUBOID_INTERVALS[key]) for key in ("dealer", "player", "action")
)

def phi(state, action=None):
    if state == "terminal": return 0

    dealer, player = state.dealercard,state.playersum

    state_features = np.array([
    (di[0] <= dealer <= di[1]) and (pi[0] <= player <= pi[1])
    for di in CUBOID_INTERVALS['dealer']
    for pi in CUBOID_INTERVALS['player']
    ]).astype(int).reshape(FEATS_SHAPE[:2])

    if action is None: return state_features

    features = np.zeros(FEATS_SHAPE)
    for i, ai in enumerate(CUBOID_INTERVALS['action']):
        if action in ai:
          features[:, :, i] = state_features

    print("in phi,features:",features)
    return features.astype(int)


def expand_Q(w):
    Q = np.zeros((10,21,2))

    for dealer in DEALER_RANGE:
        for player in PLAYER_RANGE:
          for action in ACTIONS:
            #state = (dealer, player)
            state = State()
            state.dealercard = dealer
            state.playersum = player
            feats = phi(state, action)
            Q[dealer-1, player-1][action] = np.sum(feats * w)

    return Q

def policy(state,w):
    if state == "terminal":
        return 0.0, None

    if np.random.rand() < (1 - EPSILON):
        Qhat, action = max(
        # same as dotproduct in our case
        ((np.sum(phi(state, a) * w), a) for a in ACTIONS),
        key=lambda x: x[0])
    else:
        action = np.random.choice(ACTIONS)
        Qhat = np.sum(phi(state, action) * w)

    return Qhat, action

def store_e(e,D):
#store experiences
    if D.length >= N:
        index = random.randint(0,N-1)
        D[index] = e
    else:
        D.append(e)

    return D

def policy(Q_value,state,greedy=False):
    if !greedy:
        epsilon = 100.0 / (100.0 + np.sum(counter[state.dealercard-1, state.playersum-1,:],axis=0))
        #the possibility to choose random
        if (random.random() < epsilon):
            action = random.randint(0,1)
        else:#greedy choose
            action = np.argmax(value[state.dealercard-1, state.playersum-1,:])
        return action

    action = np.argmax(value[state.dealercard-1, state.playersum-1,:])
    return value[state.dealercard-1, state.playersum-1,action]




def dqn_learn(opt_value,num_episodes):
    #initialize
    Q = np.zeros((10,21,2))
    counter = np.zeros((10,21,2))
    totalreward = 0
    error_history = []
    w  = (np.random.rand(*FEATS_SHAPE) - 0.5) * 0.001
    D = []

    for episode in range(1,num_episodes+1):
        # initialize env
        state1 = State()
        state1.dealercard = random.randint(1,10)
        state1.playersum = random.randint(1,10)
        #state1 = (state1.dealercard,state1.playersum)
        E = np.zeros_like(w)

        while state1 != "terminal":
            action1 = Epsilon_greedy_policy(Q,counter,state1）
            #Qhat1, action1 = policy(state1,w)
            state2, reward = Step(state1,action1)
            #Qhat2, action2 = policy(state2,w)
            e = (state1,action1,reward,state2)
            D = store_e(e,D)

            index = random.randint(0,N-1)
            e = D[index]
            if e[3] is "terminal":
                y = e[2]
            else:
                y = e[2]+GAMMA*policy(Q,e[3],True)





            grad_w_Qhat1 = feats1

            delta = reward + GAMMA * Qhat2 - Qhat1
            E = GAMMA * lmbd * E + grad_w_Qhat1
            dw = ALPHA * delta * E

            w += dw
            state1 = state2

            Q = expand_Q(w)
            #print("in lfa while")
            error_history.append((episode, mse(Q,opt_value)))

    return Q,error_history
