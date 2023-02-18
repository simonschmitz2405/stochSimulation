import numpy as np
from matplotlib import pyplot as plt

S = 12 #defining number of strata
M = 1000 #number timestep
N = 2  #number brownian sample paths

# Generate S times Brownian motion and save endpoints as Omega
t = []
for i in range(M):
    t.append(i/M)

Omega = []
for u in range(S):   
    Wt = []
    Wt.append(0)
    for k in range(M-1):
        deltaWk = np.random.normal(0,t[k+1]-t[k])
        Wt.append(Wt[-1]+deltaWk)
    Omega.append(Wt[-1])
    
# For  each stratum omega we know want to construct with brownian bridge sampling two Brownian sample paths


