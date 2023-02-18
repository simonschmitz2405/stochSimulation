import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from mpl_toolkits.mplot3d import Axes3D

# Time continuos Markov Chain

# We try to implement Reaction like random walk
# we have prob for each option
n = 100
prob1 = 0.1
prob2 = 0.05
prob3 = 0.3
prob4 = 0.1
prob5 = 1 - prob1 - prob2 - prob3 -prob4

N = np.zeros((n,3))
# Define start conditions
N[0,:] = np.array([2,3,1])
for i in range(n-1):
    zufall = np.random.uniform(0,1)
    if(zufall>= 0 and zufall < prob1):
        # s1 - 1
        N[i+1,:] = N[i,:] + np.array([-1,0,0])
    if(zufall>= prob1 and zufall < prob2+prob1):
        # s1 - 2
        N[i+1,:] = N[i,:] + np.array([-2,0,0])
    if(zufall>= prob1 + prob2 and zufall < prob3 + prob1 + prob2):
        # s1+2,s2-1
        N[i+1,:] = N[i,:] + np.array([2,-1,0])
    if(zufall>= prob1+prob2+prob3 and zufall < prob1+prob2+prob3+prob4):
        # s2-1,s3+1
        N[i+1,:] = N[i,:] + np.array([0,-1,1])
    if(zufall>= prob1+prob2+prob3+prob4 and zufall <= 1):
        # gleich
        N[i+1,:] = N[i,:]
        
# 3.2
# Define time series
T = 0.1
h = 0.001
c = np.array([1,5,15,3/4])
N_2 = np.zeros((n,3))
N_2[0,:] = np.array([400,800,0])
a = np.zeros(4)
S = np.zeros(n)
J = np.zeros(n)
J[0] = 0
for i in range(n-1):
    # Defining the propensity function a(N)
    a[0] = c[0]*N_2[i,0]
    a[1] = c[1]*(N_2[i,0]*(N_2[i,0]-1))/2
    a[2] = c[2]*N_2[i,1]
    a[3] = c[3]*N_2[i,1]
    # Implement the Reaction simulation
    lam = np.sum(a) # Step 3
    S[i] = np.random.exponential(lam) # Step 4
    J[i+1] = J[i] + S[i] 
    # Now we have to define the four probabilities for the four cases
    probI1 = a[0]/np.sum(a)
    probI2 = a[1]/np.sum(a)
    probI3 = a[2]/np.sum(a)
    probI4 = a[3]/np.sum(a)
    # step 6: 
    for i in range(n-1):
        zufall = np.random.uniform(0,1)
        if(zufall>= 0 and zufall < probI1):
            # s1 - 1
            N_2[i+1,:] = N_2[i,:] + np.array([-1,0,0])
        if(zufall>= probI1 and zufall < probI2+probI1):
            # s1 - 2
            N_2[i+1,:] = N_2[i,:] + np.array([-2,0,0])
        if(zufall>= probI1 + probI2 and zufall < probI3 + probI1 + probI2):
            # s1+2,s2-1
            N_2[i+1,:] = N_2[i,:] + np.array([2,-1,0])
        if(zufall>= probI1+probI2+probI3 and zufall < probI1+probI2+probI3+probI4):
            # s2-1,s3+1
            N_2[i+1,:] = N_2[i,:] + np.array([0,-1,1])
        if(zufall>= probI1+probI2+probI3+probI4 and zufall <= 1):
            # gleich
            N_2[i+1,:] = N_2[i,:]
            
plt.step(J,N[:,0],label="N1")
plt.step(J,N[:,1],label="N2")
plt.step(J,N[:,2],label="N3")
plt.legend()
plt.title("Number of molecules in Time t in [0,{}]".format(T))
plt.show()
    

    
