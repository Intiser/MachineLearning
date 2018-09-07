# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 22:49:10 2017

@author: ahsan
"""


import numpy as np
import gym

def value_iteration(env,diff=0.001,gamma=1.00):
    nA, nS = env.nA, env.nS
    U = np.zeros((nS))
    U1 = np.zeros((nS))
    T = np.zeros([nS, nA, nS])
    R = np.zeros([nS, nA, nS])
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans,next_s,rew,done in transitions:
                T[s,a,next_s] += p_trans
                R[s,a,next_s] = rew
            T[s,a,:]/=np.sum(T[s,a,:])
    mx = 0
    while True :
        U = U1.copy()
        delta = 0
        
        for s in range(nS):
            val = 0
            for a in range(nA):
               sums = 0
               for s1 in range(nS):
                   sums += T[s][a][s1]*(U[s1] + R[s][a][s1]) 
                   if R[s][a][s1] > mx :
                       mx = R[s][a][s1]
                       
               val =  np.max(val,sums)
            U1[s] =   val*gamma
            ab = np.abs(U1[s]-U[s])
            delta = np.max(ab, delta)
            print(delta,mx)
        if(delta < diff):
            break
        
    return U                

    
#start environment. 
env = gym.make('FrozenLake8x8-v0')
nA, nS = env.nA, env.nS
   

Q = np.zeros((nS,nA))

alpha = 0.01
gamma = 0.9
epsilon = 1.00
decay = 0.99

U = value_iteration(env,gamma)
print(U)
