import numpy as np
import random
from collections import deque
from environment import (ACTIONS,
DEALER_RANGE,PLAYER_RANGE)
# import sklearn
# import sklearn.datasets
# import sklearn.linear_model


GAMMA = 0.8 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
STATE_DIM = 2
ACTION_DIM = 2


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

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


class DQN():
    def __init__(self, env):
        self.epsilon = 0.05
        self.replay_buffer = deque()
        self.epsilon = INITIAL_EPSILON
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.initialize_parameters()
        # self.create_Q_network()
        # self.create_training_method()

    def layer_sizes(self,n_x=2,n_h=20,n_y=2):
        return n_x,n_h,n_y

    def initialize_parameters(self):
        np.random.seed(2)
        n_x,n_h,n_y = self.layer_sizes()
        W1 = np.random.randn(n_h,n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*0.01
        b2 = np.zeros((n_y,1))

        print("initialize_parameters,W1,b1,W2,b2,shape:",W1.shape,b1.shape,W2.shape,b2.shape)

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        self.parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

        #return parameters

    def forward_propagation(self,X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initia
        lization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Implement Forward Propagation to calculate A2 (probabilities)
        #print("X shape in forward_propagation:",X.shape)
        #print("in forward_propagation,W1 X shape:",W1.shape,X.shape)

        X = np.array(X)

        Z1 = np.dot(W1,X)+b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1)+b2
        A2 = sigmoid(Z2)
        #print("in forward_propagation,Z1,A1,Z2,A2,shape",Z1.shape,A1.shape,Z2.shape,A2.shape)

        #print("A2.shape",A2.shape)
        #print("X.shape[1]",X.shape[1])


        assert(A2.shape == (2, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

    #GRADED FUNCTION
    def compute_cost(self,A2, Y):
        """
        Computes the cross-entropy cost given in equation
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of ex
        amples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2

        Returns:
        cost -- cross-entropy cost given equation
        """
        #print("in compute_cost,Y:",Y)
        #print("in compute_cost,A2.T:",A2.T)
        #print("in compute_cost,action_batch:",action_batch)

        #Q_value = np.sum(np.multiply(action_batch,A2.T),axis=1)
        #print("Q_value,Q_value.shape",Q_value,Q_value.shape)

        #print("Y-Q_value",(Y-Q_value))

        cost = (1/BATCH_SIZE)*np.sum(np.square(Y-A2))


        #print("cost in compute_cost:",cost)

        assert(isinstance(cost, float))


        return cost

    def backward_propagation(self,parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.
        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to differe
        nt parameters
        """
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]

        # Backward propagation: calculate dW1, db1, dW2, db2.
        #print("in backward_propagation, X.shape,Y.shape",X.shape,Y.shape)
        #A2_value = np.multiply(action_batch,A2.T)
        #print("in backward_propagation, A2_value",A2_value)
        dZ2= A2-Y
        dW2 = (1/BATCH_SIZE)*np.dot(dZ2,A1.T)
        #print("in backward_propagation, dZ2 dW2 shape",dZ2.shape,dW2.shape)
        db2 = (1/BATCH_SIZE)*np.sum(dZ2,axis=1,keepdims=True)
        #print("in backward_propagation, db2.shaoe",db2.shape)
        dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
        #print("in backward_propagation, dZ1.shape",dZ1.shape)
        dW1 = (1/BATCH_SIZE)*np.dot(dZ1,X.T)
        db1 = (1/BATCH_SIZE)*np.sum(dZ1,axis=1,keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self,parameters, grads, learning_rate = 0.1):
        """
        Updates parameters using the gradient descent update rule given above
        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        #learning_rate =

        # Update rule for each parameter
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2

        self.parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters


    #later change all to column vectors.
    def nn_model(self, episodes):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (2, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predi
        ct.
        """
        np.random.seed(3)

        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        #state_batch = np.zeros(shape=)
        state_batch = np.array([(data[0].dealercard,data[0].playersum) for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = [data[2] for data in minibatch]

        next_state_batch = np.zeros(shape=(BATCH_SIZE,2))
        for index,data in enumerate(minibatch):
            if data[3] == "terminal":
                next_state_batch[index] = [0,0]
            else:
                next_state_batch[index] = [data[3].dealercard,data[3].playersum]

        y_batch = np.zeros(shape=(BATCH_SIZE,2))
        Q_next_value_batch = self.Q_value(next_state_batch)
        parameters = self.parameters
        X = state_batch

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = self.forward_propagation(X.T, parameters)

        # print("in nn_model,reward_batch",reward_batch)
        # print("in nn_model,action_batch",action_batch)

        #print("A2 in nn_model",A2)
        for i in range(0,BATCH_SIZE):
            done = next_state_batch[i]
            action_index = np.argmax(action_batch[i])
            #print("in nn_model,y_batch[i],y_batch[i][action_index],reward_batch[i]",y_batch[i],y_batch[i][action_index],reward_batch[i])
            if np.array_equal(done,[0,0]):
                y_batch[i][action_index] = reward_batch[i]
            else :
                y_batch[i][action_index] = reward_batch[i] + GAMMA * np.max(Q_next_value_batch.T[i])

            y_batch[i][1-action_index] = A2.T[i][1-action_index]

        Y = y_batch
        #print("in nn_model,shape of A2,X,Y,",A2.shape,X.shape,Y.shape)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = self.compute_cost(A2, Y.T)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = self.backward_propagation(parameters, cache, X.T, Y.T)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        self.update_parameters(parameters,grads)

        if  episodes % 1000 == 0:
            print ("Cost after iteration %i: %f" %(episodes, cost))


    def Q_value(self,state):
        """
        Using the learned parameters, predicts a class for each example in X
        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        ### START CODE HERE ### (≈ 2 lines of code)
        if isinstance(state, State):
            #print("in Q_value 0, type of state:",type(state))
            state_tmp = np.array([[state.dealercard,state.playersum]]).T
            A2, cache = self.forward_propagation(state_tmp, self.parameters)
        else:
            #print("in Q_value 1, type of state:",type(state))
            A2, cache = self.forward_propagation(state.T, self.parameters)
        #print(A2)
        ### END CODE HERE ###
        return A2


    def perceive(self,state,action,reward,next_state,episode):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state))
        if len(self.replay_buffer) > REPLAY_SIZE:
          self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
          #self.train_Q_network()
          self.nn_model(episode)

    def egreedy_action(self,state):
        #1 calculate the Q-value:Q_value
        Q_value = self.Q_value(state)
        #2 based on the Q_value, choose the action using egreedy policy
        if random.random() <= self.epsilon:
            action = random.randint(0,self.action_dim - 1)
        else:
            action = np.argmax(Q_value)

        if self.epsilon >= FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

        return action

    def expand_Q(self):
        Q = np.zeros((10,21,2))

        for dealer in DEALER_RANGE:
            for player in PLAYER_RANGE:
                for action in ACTIONS:

                    state = State()
                    state.dealercard = dealer
                    state.playersum = player

                    action_value = self.Q_value(state)
                    Q[dealer-1, player-1][action] = action_value[action]

        return Q


def dqn_control(num_episodes):
    env = ENV()
    agent = DQN(env)

    for episode in range(num_episodes):
    # initialize task
        state = env.reset()
    # Train
        while state != "terminal":
          action = agent.egreedy_action(state) # e-greedy action for train
          #print("~~~++++state,action:",state,action)
          next_state,reward = env.step(state,action)
          # Define reward for agent
          #reward_agent = -1 if done else 0.1
          agent.perceive(state,action,reward,next_state,episode)
          state = next_state
          # if state is "terminal":
          #   break
        # Test every 100 episodes
    return agent.expand_Q()
