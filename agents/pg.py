'''
Trains an agent with (stochastic) Policy Gradients on easy21 game
'''
import numpy as np
import pickle
import random,math,sys
from environment import (ACTIONS,
DEALER_RANGE,PLAYER_RANGE)

DEBUG = False
# DEBUG = True
DEBUG_level0 = False #help function level
DEBUG_level1 = True#True #main function level

# hyperparameters
H = 10 # 200 number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
D = 2
#model,grad_buffer,rmsprop_cache = {},{},{}


model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization

#print("come here, model['W1']",model['W1'])
model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
#print("grad_buffer,rmsprop_cache",grad_buffer,rmsprop_cache)

class State:
    dealercard = random.randint(1,10)
    playersum = random.randint(1,10)


class easy21():
    def __init__(self):
        self.reset()

    #init the first state
    def reset(self):
        state = State()
        state.dealercard = random.randint(1,10)
        state.playersum = random.randint(1,10)
        return state

    def drawcard(self,current):
        #add black color 2/3 pros;subtract red color 1/3
        if random.randint(1,3) < 3:
            current += random.randint(1,10)
        else:
            current -= random.randint(1,10)
        return current
    # action {0: stick, 1: hit}
    # return next_state, reward, done
    def step(self,state,action):
        state_next = State()
        if action == 1:
            state_next.playersum = self.drawcard(state.playersum)
            state_next.dealercard = state.dealercard
            if state_next.playersum < 1 or state_next.playersum > 21:
                return state,-1.0,True#"terminal",-1.0
            else:
                return state_next,0,False
        else:# action == 0:#dealer's turn
            dealercardsum = state.dealercard
            while(dealercardsum < 17):
                dealercardsum = self.drawcard(dealercardsum)
                if dealercardsum < 1 or dealercardsum > 21:
                    return state,1.0,True#"terminal", 1.0
            if dealercardsum > state.playersum:
                return state,-1.0,True#"terminal", -1.0
            elif dealercardsum < state.playersum:
                return state,1.0,True#"terminal", 1.0
            else:
                return state,0.0,True#"terminal", 0.0

def sigmoid(x): 
	return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


def discount_rewards(r):
	''' take 1D float array of rewards and compute discounted reward '''
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


def policy_forward(x):
	if(DEBUG and DEBUG_level0):print("in policy_forward,model['W1']:",model['W1'])
	if(DEBUG and DEBUG_level0):print("in policy_forward,x:", x)
	h = np.dot(model['W1'], x)
	#if(DEBUG and DEBUG_level0):print("in policy_forward,h:",h)
	h[h<0] = 0 # ReLU nonlinearity
	logp = np.dot(model['W2'], h)
	p = sigmoid(logp)
	return p, h # return probability of taking action 2, and hidden state


def policy_backward(eph, epx, epdlogp):
	''' backward pass. (eph is array of intermediate hidden states) '''

	dW2 = np.dot(eph.T, epdlogp).ravel()
	for item in dW2:
		if math.isnan(item):
			#print("dW2,eph.T,epdlogp",dW2,eph.T,epdlogp)
			sys.exit(0)

	dh = np.outer(epdlogp, model['W2'])
	dh[eph <= 0] = 0 # backpro prelu
	dW1 = np.dot(dh.T, epx)
	return {'W1':dW1, 'W2':dW2}

def prepro(observation):
	'''preprocess the obervation, here do nothing'''
	pass

def expand_Q():
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

def pg_control(num_episodes):
	env = easy21()
	xs,hs,dlogps,drs = [],[],[],[]
	running_reward = None
	reward_sum = 0
	episode_number = 0
	
	state = env.reset()
	#x = [state.dealercard,state.playersum]

	while episode_number <= num_episodes:
		# preprocess the observation, set input to network to be difference image
		#cur_x = prepro(observation)
		# forward the policy network and sample an action from the returned probability
		x = [state.dealercard,state.playersum]
		if(DEBUG and DEBUG_level1):print("x",x)

		aprob, h = policy_forward(x)
		action = 1 if np.random.uniform() < aprob else 0 # roll the dice! # action {0: stick, 1: hit}

		# record various intermediates (needed later for backprop)
		xs.append(x) # observation
		hs.append(h) # hidden state
		y = 1 if action == 1 else 0 # a "fake label"
		dlogps.append(y - aprob)#grad that encourages the action that was taken to be taken,(see http://cs231n.github.io/neural-networks-2/#losses if confused)

		# step the environment and get new measurements
		state, reward, done = env.step(state,action)
		reward_sum += reward

		if(DEBUG and DEBUG_level1):print("reward,",reward,state.dealercard,state.playersum)

		drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
		#print("x,action,drs,episode_number",x,action,drs,episode_number)

		if done: # an episode finished
			episode_number += 1
			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			#if(DEBUG and DEBUG_level1):print("in done drs,episode_number:",drs,episode_number)
			epx = np.vstack(xs)
			eph = np.vstack(hs)
			epdlogp = np.vstack(dlogps)
			epr = np.vstack(drs)
			xs,hs,dlogps,drs = [],[],[],[] # reset array memory

			 # compute the discounted reward backwards through time
			discounted_epr = discount_rewards(epr)
			#if(DEBUG and DEBUG_level1):print("discounted_epr:",discounted_epr)
			#print("x,action,discounted_epr,episode_number",x,action,discounted_epr,episode_number)

			# standardize the rewards to be unit normal (helps control the gradient estimator variance)
			if np.std(discounted_epr) != 0:
				discounted_epr -= np.mean(discounted_epr)
				discounted_epr /= np.std(discounted_epr)

			#print("discounted_epr",discounted_epr)
			epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
			
			grad = policy_backward(eph, epx, epdlogp)
			if(DEBUG and DEBUG_level1):print("grad:",grad)
			
			for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

			# perform rmsprop parameter update every batch_size episodes

			if(DEBUG and DEBUG_level1):print("model before:",model)

			if episode_number % batch_size == 0:
				for k,v in model.items():
					#print("k,v",k,v)
					g = grad_buffer[k] # gradient
					#if(DEBUG and DEBUG_level1):print("g:",g)
					rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
					#if(DEBUG and DEBUG_level1):print("rmsprop_cache[k]:",rmsprop_cache[k])
					
					model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
					grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

			if(DEBUG and DEBUG_level1):print("model after:",model)
			# boring book-keeping
			running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
			#print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
			if episode_number % 1000 == 0: 
				pickle.dump(model, open('save.p', 'wb'))
				#print ("Cost after iteration %i: %f" %(episode_number, cost))
			reward_sum = 0
			state = env.reset() # reset env
			#x = [state.dealercard,state.playersum]

			#prev_x = None



		# if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
		# 	print ('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
		if episode_number % 1000 == 0: 
				#pickle.dump(model, open('save.p', 'wb'))
			print ("Cost after iteration %i: %f" %(episode_number, (y - aprob)))



















