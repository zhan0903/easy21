import matplotlib
import numpy as np
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import cm, rc
import pickle

DEALER_RANGE = range(1, 11)
PLAYER_RANGE = range(1, 22)
ACTIONS = (HIT, STICK) = (0, 1)

class State:
    dealercard = random.randint(1,10)
    playersum = random.randint(1,10)

def mse(A, B):
    return np.sum((A - B) ** 2) / np.size(A)

Q_DUMP_BASE_NAME = "Q_dump"
def dump_Q(Q, num_episode):
    filename = ("./{}_episodes_{}.pkl"
              "".format(Q_DUMP_BASE_NAME,num_episode))

    print("dumping Q: ", filename)

    with open(filename, "wb") as f:
        pickle.dump(Q, f)


def create_surf_plot(X, Y, Z, fig_idx=1):
    fig = plt.figure(fig_idx)
    ax = fig.add_subplot(111, projection="3d")
    #print("X,Y,Z:",X,Y,Z)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #surf = ax.plot_wireframe(X, Y, Z)

    return surf

def plot_V(Q, save=None, fig_idx=0):
    #print("Q value:",Q)
    V = np.max(Q, axis=2)
    X, Y = np.mgrid[DEALER_RANGE,PLAYER_RANGE]

    surf = create_surf_plot(X, Y, V)

    #print("end cread_surf_plot")

    plt.title("V*")
    plt.ylabel('player sum', size=18)
    plt.xlabel('dealer', size=18)

    if save is not None:
        plt.savefig(save, format='pdf', transparent=True)
    else:
        plt.show()

    plt.clf()


def plot_learning_curve(learning_curves, save=None, agent_args={}, fig_idx=2):

    fig = plt.figure(fig_idx)

    plt.title("Mean-squared error vs. 'true' Q values against episode number")
    plt.ylabel(r'$\frac{1}{|S||A|}\sum_{s,a}{(Q(s,a) - Q^{*}(s,a))^2}$', size=18)
    plt.xlabel(r'$episode$', size=18)

    colors = iter(cm.rainbow(np.linspace(0, 1, len(learning_curves))))
    for lmbd, D in learning_curves.items():
        X, Y = zip(*D)
        plt.plot(X, Y, label="lambda={:.1f}".format(lmbd),
                 linewidth=1.0, color=next(colors))

    plt.legend()

    if save is not None:
        plt.savefig(save, format='pdf', transparent=True)
    else:
        plt.show()

    plt.clf()



#draw one card black or red
def Drawcard(current):
    #add black color 2/3 pros;subtract red color 1/3
    if random.randint(1,3) < 3:
        current += random.randint(1,10)
    else:
        current -= random.randint(1,10)
    return current

# action {0: stick, 1: hit}
# return reward and next state
def Step(state,action):
    state_next = State()
    if action == 1:
        state_next.playersum = Drawcard(state.playersum)
        state_next.dealercard = state.dealercard
        if state_next.playersum < 1 or state_next.playersum > 21:
            return "terminal",-1.0
        else:
            return state_next,0
    elif action == 0:#dealer's turn
        dealercardsum = state.dealercard
        while(dealercardsum < 17):
            dealercardsum = Drawcard(dealercardsum)
            if dealercardsum < 1 or dealercardsum > 21:
                return "terminal", 1.0
        if dealercardsum > state.playersum:
            return "terminal", -1.0
        elif dealercardsum < state.playersum:
            return "terminal", 1.0
        else:
            return "terminal", 0.0

# Base on epsilon-policy,number of action and Q value table, return the action.
def Policy(value,counter,state):
    #get the time-varying epsilon, N_0 = 100.0
    #print("state in Policy:",state.dealercard-1,state.playersum-1)
    epsilon = 100.0 / (100.0 + np.sum(counter[state.dealercard-1, state.playersum-1,:],axis=0))
    #the possibility to choose random
    if (random.random() < epsilon):
        action = random.randint(0,1)
    else:#greedy choose
        action = np.argmax(value[state.dealercard-1, state.playersum-1,:])

    return action

# An episode is an array of (state, action, reward) tuples
def Generate_epsiode(value,counter):
    epsiode = []
    # initialize the fisrt State
    state = State()
    #make sure the start state is different every time
    state.dealercard = random.randint(1,10)
    state.playersum = random.randint(1,10)
    totalreward = 0

    while state != "terminal":
        #print("state in Generate_epsiode:",state.dealercard-1,state.playersum-1)
        action = Policy(value,counter,state)
        counter[state.dealercard-1, state.playersum-1,action] += 1
        epsiode.append((state.dealercard, state.playersum,action))
        state,reward= Step(state,action)
        totalreward += reward

    return epsiode,totalreward

# Based on article 5 on-policy  MC Control algorithm-Incremental Implementation
