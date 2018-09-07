# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 01:12:23 2017

@author: ahsan
"""

import numpy as np
import gym

def policy_evaluation(env,pi,gamma=1.00,k = 100):
    nA = env.nA
    nS = env.nS
    P  = env.P
    U = np.zeros(nS)
    u = np.zeros(nS)
    print(gamma)
    for it in range(k):
        for s in range(nS):
            a  = pi[s]
            nwState = P[s][a]
            sm = 0
            u = U.copy()
            for trans,nextS,rwd,done in nwState:
                sm +=   ((gamma * u[nextS]) + rwd )*trans
            U[s] = sm  

    return U

def policy_iteration(env,policy,gamma = 1.00,k=100):
    nA = env.nA
    nS = env.nS
    P  = env.P
    U  = np.zeros(nS)
    pi = np.zeros(nS)
    opt = np.argmax(policy,axis = 1)
    #print(opt)
    while True:
        pi = opt.copy()
        U = policy_evaluation(env,pi,gamma,k)
        for s in range(nS):
            for a in range(nA):
                nState = P[s][a]
                sm     = 0
                for trans,nextS,rwd,done in nState:
                        sm +=   ((gamma * U[nextS]) + rwd )*trans
                if sm > U[s]:
                    opt[s] = a
        
        if np.array_equal(pi,opt):
            break
    return pi
    
    
    
    
#start environment. 
env = gym.make('FrozenLake-v0')
nA, nS = env.nA, env.nS
   
#reward and transition matrices
T = np.zeros([nS, nA, nS])
R = np.zeros([nS, nA, nS])
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans,next_s,rew,done in transitions:
            T[s,a,next_s] += p_trans
            R[s,a,next_s] = rew
        T[s,a,:]/=np.sum(T[s,a,:])
        
#calculate optimal policy
policy = (1.0/nA)*np.ones([nS,nA])
opt = policy_iteration(env,policy,gamma =0.9999,k= 100)

print(opt)

#test optimal policy
max_time_steps = 100000
n_episode = 1000

#env.monitor.start('./frozenlake-experiment', force=True)

for i_episode in range(n_episode):

    observation = env.reset() #reset environment to beginning 

    #run for several time-steps
    for t in xrange(max_time_steps): 
        #display experiment
        env.render() 

        #sample a random action 
        action = opt[observation]

        #observe next step and get reward 
        observation, reward, done, info = env.step(action)

        if done:
            env.render() 
            print "Simulation finished after {0} timesteps".format(t)
            break
    
    if reward > 0:
        print "the {0} th time ".format(i_episode)
        break
    
        
            
#env.monitor.close()