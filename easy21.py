#rewrite code is necessary if you want to achieve a better performance. for test, to finish it is more important.
from lib import *
import time

num_episode = 1000000
# Based on article 5 on-policy  MC Control algorithm-Incremental Implementation
def Monte_Carlo_Control():
    #initialize
    # value[action][dealercard][playersum] is the value function, table lookup should work with so little states
    #value = np.zeros((2,11,22))
    value = np.zeros((10,21,2))
    counter = np.zeros((10,21,2))
    totalreward = 0

    #main logic
    for i in range(1,num_episode):
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


Q_value = Monte_Carlo_Control()
#4 plot the final value_function
# plot monte-carlo value func
plot_file = ("./outcome/V_MC_{}_episodes_time_{}.pdf".format(num_episode,time.time()))

#plot_file = ("./outcome/V_MC_{}_episodes3.pdf".format(num_episode))

plot_V(Q_value,save=plot_file)
dump_Q(Q_value,num_episode)
