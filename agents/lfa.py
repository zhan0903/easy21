import pickle,random
from environment import (
Policy,Step,mse,State,ACTIONS,
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

def lfa_learn(lmbd,opt_value,num_episodes):
    #initialize
    Q = np.zeros((10,21,2))
    counter = np.zeros((10,21,2))
    totalreward = 0
    error_history = []
    w  = (np.random.rand(*FEATS_SHAPE) - 0.5) * 0.001

    for episode in range(1,num_episodes+1):
        # initialize env
        state1 = State()
        state1.dealercard = random.randint(1,10)
        state1.playersum = random.randint(1,10)
        #state1 = (state1.dealercard,state1.playersum)
        E = np.zeros_like(w)

        while state1 != "terminal":
            Qhat1, action1 = policy(state1,w)
            state2, reward = Step(state1,action1)
            Qhat2, action2 = policy(state2,w)

            feats1 = phi(state1, action1)
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
