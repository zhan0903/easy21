import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import (ACTIONS,
DEALER_RANGE,PLAYER_RANGE)

# Hyper Parameters for DQN
GAMMA = 1 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 128 # size of minibatch


class State:
    dealercard = random.randint(1,10)
    playersum = random.randint(1,10)

class ENV():
    def __init__(self):
        self.reset()

    #init the first state
    def reset(self):
        state = State()
        state.dealercard = random.randint(1,10)
        state.playersum = random.randint(1,10)
        return state

    def Drawcard(self,current):
        #add black color 2/3 pros;subtract red color 1/3
        if random.randint(1,3) < 3:
            current += random.randint(1,10)
        else:
            current -= random.randint(1,10)
        return current
    # action {0: stick, 1: hit}
    # return reward and next state
    def step(self,state,action):
        state_next = State()
        if action == 1:
            state_next.playersum = self.Drawcard(state.playersum)
            state_next.dealercard = state.dealercard
            if state_next.playersum < 1 or state_next.playersum > 21:
                return "terminal",-1.0
            else:
                return state_next,0
        elif action == 0:#dealer's turn
            dealercardsum = state.dealercard
            while(dealercardsum < 17):
                dealercardsum = self.Drawcard(dealercardsum)
                if dealercardsum < 1 or dealercardsum > 21:
                    return "terminal", 1.0
            if dealercardsum > state.playersum:
                return "terminal", -1.0
            elif dealercardsum < state.playersum:
                return "terminal", 1.0
            else:
                return "terminal", 0.0

# DQN Agent
class DQN():
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        #self.state_dim = env.observation_space.shape[0]
        self.state_dim = 2
        #self.action_dim = env.action_space.n
        self.action_dim = 2

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        #axis=1表示行相加，Q_value和action_input做内积
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),1)
        #返回所有项的平均值
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    # action {0: stick, 1: hit}
    def perceive(self,state,action,reward,next_state):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state))
        if len(self.replay_buffer) > REPLAY_SIZE:
          self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
          self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [(data[0].dealercard,data[0].playersum) for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = []

        for data in minibatch:
            if data[3] == "terminal":
                #next_state_batch.append(data[3])
                next_state_batch.append((0,0))
            else:
                next_state_batch.append((data[3].dealercard,data[3].playersum))

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        # print("Q_value_batch",Q_value_batch)

        for i in range(0,BATCH_SIZE):
            #print("Q_value_batch[i]",Q_value_batch[i])
            #state = minibatch[i][3]
            #print("state in train_Q_network:",state)
            done = next_state_batch[i]
            #print("done in train_Q_network",done)
            if done == (0,0):
                #print("come here in train_Q_network",done)
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))


        # print("next_state_batch",next_state_batch)
        # print("y_batch",y_batch)
        # print("action_batch",action_batch)
        # print("state_batch",state_batch)

        #state_batch = [(9, 4), (7, 1), (3, 11), (6, 11), (4, 4)]
        #state_batch_test = np.array(state_batch)
        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })

    def egreedy_action(self,state):

        #feed_dict = {self.state_input:[state.dealercard,state.playersum]}
        #print("feed_dict,self.Q_value in egreedy_action:",feed_dict,self.Q_value)

        Q_value = self.Q_value.eval(feed_dict = {
          self.state_input:[[state.dealercard,state.playersum]]
          })[0]

        if random.random() <= self.epsilon:
            action = random.randint(0,self.action_dim - 1)
        else:
            action = np.argmax(Q_value)

        if self.epsilon >= FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

        return action

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
          self.state_input:[[state.dealercard,state.playersum]]
          })[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def expand_Q(self):
        Q = np.zeros((10,21,2))

        for dealer in DEALER_RANGE:
            for player in PLAYER_RANGE:
                for action in ACTIONS:
                    Q[dealer-1, player-1][action] = self.Q_value.eval(feed_dict = {
                      self.state_input:[(dealer,player)]})[0][action]

        return Q


# Hyper Parameters
ENV_NAME = 'CartPole-v0'
#EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def dqn_control(num_episodes):
  # initialize env and dqn agent
    env = ENV()
    agent = DQN(env)
    EPISODE = num_episodes

    for episode in range(EPISODE):
    # initialize task
        state = env.reset()
    # Train
        while state != "terminal":
          action = agent.egreedy_action(state) # e-greedy action for train
          #print("~~~++++state,action:",state,action)
          next_state,reward= env.step(state,action)
          # Define reward for agent
          #reward_agent = -1 if done else 0.1
          agent.perceive(state,action,reward,next_state)
          state = next_state
          # if state is "terminal":
          #   break
        # Test every 100 episodes

    return agent.expand_Q()

    state = env.reset()
    print("after training-----,agent.Q_value:",agent.Q_value.eval(feed_dict = {
      agent.state_input:[(state.dealercard,state.playersum)]})[0])
    print("after training-----,agent.Q_value:",agent.Q_value.eval(feed_dict = {
      agent.state_input:[(state.dealercard,state.playersum)]})[0][0])
    print("after training-----,agent.Q_value:",agent.Q_value.eval(feed_dict = {
      agent.state_input:[(state.dealercard,state.playersum)]})[0][1])
        # if episode % 100 == 0:
        #   total_reward = 0
        #   for i in range(TEST):
        #     state = env.reset()
        #     while state != "terminal":
        #       #env.render()
        #       action = agent.action(state) # direct action for test
        #       state,reward = env.step(state,action)
        #       total_reward += reward
        #
        #   ave_reward = total_reward/TEST
        #   print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        #   if ave_reward >= 200:
        #     break
