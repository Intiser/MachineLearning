# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 22:49:10 2017

@author: ahsan
"""


import numpy as np
import gym

    
    
    
#start environment. 
env = gym.make('CartPole-v0')
nA, nS = 2,4
   

Q = np.zeros((nS,nA))

alpha = 0.1
gamma = 0.9
epsilon = 1.00

#start learning
max_time_steps = 100
n_episode = 10

#env.monitor.start('./frozenlake-experiment', force=True)
cnt = 0

for i_episode in range(n_episode):

    state = env.reset() #reset environment to beginning 
    fl = 0
    epsilon = 1.00 / (i_episode+1)
    #run for several time-steps
    for t in xrange(max_time_steps): 
        #display experiment
        #env.render() 
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
    
        
print(Q)
print(cnt)
#env.monitor.close()

"""
for t in xrange(max_time_steps): 
    #display experiment
    env.render() 
    pre_state = state
    #sample a random action 
    action = np.argmax(Q[state,:])
            
        
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
"""        
