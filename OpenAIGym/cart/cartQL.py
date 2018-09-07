# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:31:34 2017

@author: ahsan
"""

import gym
import pandas as pd
import numpy  as np
import random


env = gym.make('CartPole-v0')
#experiment_filename = './cartpole-experiment-1-b'
#env.monitor.start(experiment_filename, force=True)

nB  = 10 # number of bins
nS  = nB ** env.observation_space.shape[0] 
nA  = env.action_space.n


random.seed(0)
Q = np.zeros((nS,nA))
#Q = np.random.uniform(-1,1,(nS, nA))




#eps array
#eps = np.ones((nS)) * 0.5

alpha = 0.2
gamma = 1.0
epsilon = 1
decay  = 0.99
        
F = np.zeros((nS))


cart_pos_l,pole_ang_l,cart_vel_l,angle_rate_l  =  env.observation_space.low  
cart_pos_h,pole_ang_h,cart_vel_h,angle_rate_h  =  env.observation_space.high 
""" 
#obseravatoon space high and low conatains tuple of  4 :
#cart position, pole angle,cart velocity,angle rate
"""

cart_pos_bins = pd.cut([cart_pos_l, cart_pos_h], bins=nB, retbins=True)[1][1:-1]
#cart_pos_bins = pd.cut([-2.4,2.4], bins=nB, retbins=True)[1][1:-1]
#pole_ang_bins = pd.cut([pole_ang_l, pole_ang_h], bins=nB, retbins=True)[1][1:-1]
pole_ang_bins = pd.cut([-1.75,1.75], bins=nB, retbins=True)[1][1:-1]
#cart_vel_bins = pd.cut([cart_vel_l, cart_vel_h], bins=nB, retbins=True)[1][1:-1]
cart_vel_bins = pd.cut([-2.0, 2.0], bins=nB, retbins=True)[1][1:-1]
angle_rate_bins = pd.cut([angle_rate_l, angle_rate_h], bins=nB, retbins=True)[1][1:-1]
#angle_rate_bins = pd.cut([-3.5,3.5], bins=nB, retbins=True)[1][1:-1]

#print(cart_pos_bins)
#print(pole_ang_bins)



def state_num(cp,pa,cv,ar): # short forms of cart position, pole angle,cart velocity,angle rate
    cpp,paa,cvv,arr = np.digitize(cp,cart_pos_bins),np.digitize(pa,pole_ang_bins),np.digitize(cv,cart_vel_bins),np.digitize(ar,angle_rate_bins)  
    return cpp*((nB*nB)*nB) + paa*(nB*nB) + cvv*nB + arr
    #return build_state([cpp,paa,cvv,arr])


def take_action(state):
        choose_random_action = (1 - epsilon) <= np.random.uniform(0, 1)

        if choose_random_action:
            action = random.randint(0, nA - 1)
        else:
            action = Q[state].argsort()[-1]
    
        return action


max_time_steps = 200
n_episode = 10000
maxx = 0
last = 0
rew  = 1
count = 0
tot = 0
longest_run = 0

#experiment_filename = '/tmp/cartpole-experiment-1'
#env.monitor.start(experiment_filename, force=True)
#env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

for i_episode in range(n_episode):

    observation = env.reset() #reset environment to beginning 
    cp,pa,cv,ar = observation
    state = state_num(cp,pa,cv,ar)
    #epsilon = 1.00 / (i_episode+1)
    #run for several time-steps
    #print "###start :" 
    #print(state)
    #F[state] += 1 
    for t in range(max_time_steps):
        #display experiment
        #env.render() 
        pre_state = state
        #sample a random action
        action = take_action(state)
        
        #observe next step and get reward 
        observation, reward, done, info = env.step(action)
        cp,pa,cv,ar = observation
        state = state_num(cp,pa,cv,ar)
        F[state] += 1        
        #print(state)
        last = t + 1
        if done:
            R = -200
            Q[pre_state,action] += ( alpha * (R + gamma*np.max(Q[state,:]) - Q[pre_state,action]))            
            #Q[pre_state,action] =  ( alpha * (R + gamma*np.max(Q[state,:]) + (1-alpha)*Q[pre_state,action]) )    
                    
            last = t + 1
            #print(reward)
            #print(i_episode)
            #print(cnt)
            #print(state)
            #print(reward)
            print "Episode {0} : ".format(i_episode+1)
            print "Simulation finished after {0} timesteps".format(t+1)
            #print "Max Value - {0}".format(maxx)
            break
        Q[pre_state,action] +=  ( alpha * (reward + gamma*np.max(Q[state,:]) - Q[pre_state,action]) )    
        #Q[pre_state,action] =  ( alpha * (reward + gamma*np.max(Q[state,:]) + (1-alpha)*Q[pre_state,action]) )    
        
    if last >= 195: 
        count += 1
        tot += 1
    else : 
        count = 0

    if longest_run < count:
        longest_run = count
    
    
    if maxx < last : 
        maxx = last
    epsilon *= decay 


#env.monitor.close()
#print(maxx)
#print(Q)

print(tot)
print(maxx)
print(longest_run)

#gym.upload('./cartpole-experiment-2', api_key='sk_6zBBt0fbRTaBSpbiaMY5Q')