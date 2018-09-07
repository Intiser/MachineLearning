# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 22:49:10 2017

@author: ahsan
"""


import numpy as np
import gym
import matplotlib.pyplot as plt
    
    
    
#start environment. 
env = gym.make('FrozenLake-v0')
nA, nS = env.nA, env.nS

st = [0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6]   

#Q = np.zeros((nS,nA))
Q = np.random.random((nS,nA))

alpha = 0.1
gamma = 1.0
epsilon = 1.00
epsilon_decay = 0.99

#start learning
max_time_steps = 100000
n_episode = 1000

#env.monitor.start('./frozenlake-experiment', force=True)
cnt = 0
last = 0
plotsX = np.zeros((1000))
plotsY = np.zeros((1000))

for i_episode in range(n_episode):

    state = env.reset() #reset environment to beginning 
    fl = 0
    #epsilon = 1.00 / (i_episode+1)
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
            last = state
            print (info)    
                
            Q[pre_state,action] += alpha * (R + gamma*np.max(Q[state,:]) - Q[pre_state,action])            
            
            env.render() 
            #print(i_episode)
            #print(cnt)
            #print(reward)
            print "Simulation finished after {0} timesteps".format(t)
            break
        
        Q[pre_state,action] += alpha * (reward + gamma*np.max(Q[state,:]) - Q[pre_state,action])    
    
    epsilon = epsilon * epsilon_decay 
    #print "the {0} th time ".format(i_episode)
    #if reward > 0:
    #    cnt =+ 1
    #    break
    plotsX[i_episode] = i_episode + 1 
    plotsY[i_episode] = st[last]   
    env.render()
    print(i_episode, cnt, reward ,fl)
    
        
print(Q)
print(cnt)
#env.monitor.close()

plt.plot(plotsX,plotsY)
plt.show()

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