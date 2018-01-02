from agents.sarsa import *
from agents.lfa import *
from agents.mc import *
from agents.dqn3 import *
from agents.pg import *
from environment import State,plot_learning_curve,plot_V
import random,time
import argparse

parser = argparse.ArgumentParser(
  description="Simple Reinforcement Learning")
parser.add_argument("-a", "--agent",
                    choices=['mc', 'sarsa', 'lfa', 'pg','dqn'],
                    help=("Agent Type: "
                          "mc (monte carlo), "
                          "sarsa, "
                          "lfa (linear function approximation)"))

def get_agent_args(args):
  agent_type = args.agent
  agent_args = {
    "agent_type": agent_type,
    "num_episodes": args.num_episodes
  }

  if agent_type == "mc":
    return agent_args
  elif agent_type == "sarsa" or agent_type == "lfa":
    agent_args.update({
      key: getattr(args, key) for key in ["gamma"]
      if key in args
    })
    agent_args["save_error_history"] = getattr(
      args, "plot_learning_curve", False )

  return agent_args


def Sarsa():
    lmbd = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #lmbd = [0.1]
    learning_curves = {}
    state1 = State()
    num_episodes = 20000

    with open("./Q_dump_episodes_1000000.pkl", "rb") as f:
        opt_value = pickle.load(f)
    for item in lmbd:
        #print("in main, item:",item)
        state1.dealercard = random.randint(1,10)
        state1.playersum = random.randint(1,10)
        #print("state in main:",state1.dealercard,state1.playersum)
        Q_value,error_history = Sarsa_lamda_Control(item,opt_value,num_episodes)
        learning_curves[item] = error_history
        #print("learning_curves:",learning_curves)


    plot_file = ("./outcome/Sarsa_error_{}_episodes_time_{}.pdf".format(20000,time.time()))
    plot_learning_curve(learning_curves, save=plot_file)

def Lfa():
    lmbd = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    learning_curves = {}
    state1 = State()
    num_episodes = 2000

    with open("./Q_dump_episodes_1000000.pkl", "rb") as f:
        opt_value = pickle.load(f)

    for item in lmbd:
        state1.dealercard = random.randint(1,10)
        state1.playersum = random.randint(1,10)
        Q_value,error_history = lfa_learn(item,opt_value,num_episodes)
        learning_curves[item] = error_history

    plot_file = ("./outcome/lfa_error_{}_episodes_time_{}.pdf".format(20000,time.time()))
    plot_learning_curve(learning_curves, save=plot_file)

def Mc():
    #initialize
    # value[action][dealercard][playersum] is the value function, table lookup should work with so little states
    #value = np.zeros((2,11,22))
    num_episodes = 1000000
    value = np.zeros((10,21,2))
    counter = np.zeros((10,21,2))
    totalreward = 0


    Q_value = mc_learn(num_episodes)

    #4 plot the final value_function
    # plot monte-carlo value func
    plot_file = ("./outcome/V_MC_{}_episodes_time_{}.pdf".format(num_episodes,time.time()))

    #plot_file = ("./outcome/V_MC_{}_episodes3.pdf".format(num_episode))

    plot_V(Q_value,save=plot_file)
    #dump_Q(Q_value,num_episode)

def dqn():
    num_episodes = 100000
    Q_value = dqn_control(num_episodes)

    plot_file = ("./outcome/dqn/V_dqn_{}_episodes_time_{}.pdf".format(num_episodes,time.time()))

    #plot_file = ("./outcome/V_MC_{}_episodes3.pdf".format(num_episode))

    plot_V(Q_value,save=plot_file)
    #dump_Q(Q_value,num_episode)

def pg():
    num_episodes = 10000000
    Q_value = pg_control(num_episodes)

    plot_file = ("./outcome/pg/V_pg_{}_episodes_time_{}.pdf".format(num_episodes,time.time()))

    #plot_file = ("./outcome/V_MC_{}_episodes3.pdf".format(num_episode))

    #plot_V(Q_value,save=plot_file)



def main(agrs):
    #agent_args = get_agent_args(args)
    agent_type = args.agent

    if agent_type == "sarsa":
        Sarsa()
    elif agent_type == "lfa":
        Lfa()
    elif agent_type == "mc":
        Mc()
    elif agent_type == "dqn":
        dqn()
    elif agent_type == "pg":
        pg()
    else:
        print("please choose the agent type. use -a [type]")

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
