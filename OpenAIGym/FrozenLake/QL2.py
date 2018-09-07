# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 22:49:10 2017

@author: ahsan
"""


import numpy as np
import gym

    
    
    
#start environment. 
env = gym.make('FrozenLake-v0')
nA, nS = env.nA, env.nS
   

Q = np.zeros((nS,nA))

alpha = 0.1
gamma = 0.9
epsilon = 1.00
#decay = 0.99

#start learning
max_time_steps = 100000
n_episode = 1000

#env.monitor.start('./frozenlake-experiment', force=True)
cnt = 0

for i_episode in range(n_episode):

    state = env.reset() #reset environment to beginning 
    fl = 0
    epsilon = 1.00 / (i_episode+1)
    #run for several time-steps
    t = 0
    while True:  
        #display experiment
        #env.render() 
        t = t+1        
        fl = 1
        pre_state = state
        #sample a random action
        if np.random.rand() > epsilon :
            action = np.argmax(Q[state,:])
        else :
            action = env.action_space.sample() 
        
        #observe next step and get reward 
        state, reward, done, info = env.step(action)
        
        if done:
            R = 0
            if reward > 0:
                R = 1000
                cnt += 1                 
            else :
                R = -1000
                
                
            Q[pre_state,action] += alpha * (R + gamma*np.max(Q[state,:]) - Q[pre_state,action])            
            
            env.render() 
            #print(i_episode)
            #print(cnt)
            #print(reward)
            print "Simulation finished after {0} timesteps".format(t)
            break
        
        Q[pre_state,action] += alpha * (reward + gamma*np.max(Q[state,:]) - Q[pre_state,action])    
    
    #print "the {0} th time ".format(i_episode)
    #if reward > 0:
    #    cnt =+ 1
    #    break
    env.render()
    print(i_episode, cnt, reward ,fl)
    #epsilon *= decay 
        
#print(Q)

#env.monitor.close()

#again run the agent to see how it perform

t = 0
state = env.reset()
while  True : 
    #display experiment
    env.render() 
    
    #sample a random action 
    action = np.argmax(Q[state,:])
    t = t+1  
        
    #observe next step and get reward 
    state, reward, done, info = env.step(action)
        
    if done:
    
        if reward == 0:
            print "NO"                 
        else :
            print "YES"
            
        env.render() 
        print "Simulation finished after {0} timesteps".format(t)
        break
    
 
print(Q)    #display Q matrix 
print(cnt)  #number of times it was successful during training