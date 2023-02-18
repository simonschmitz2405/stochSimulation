import numpy as np
from matplotlib import pyplot as plt

# Exercise 2
# 2.1
# Generate random walk with following transition probabilities

# Define fixed alpha 
alpha = 0.25
deltaT = np.array([0.1,0.01,0.001,0.0001])

# We have a discrete space and discrete time
# Define time steps
n = int(1/deltaT[1])
X = np.zeros(n)

for i in range(n-1):   
    zufall = np.random.uniform(0,1)
    if(zufall >= 0 and zufall <= alpha):
        X[i+1] = X[i] + 1
    elif(zufall > 2*alpha and zufall <= 1):
        X[i+1] = X[i]
    else:
        X[i+1] = X[i] - 1
        
plt.plot(np.arange(0,n),X)
plt.title("Random Walk")
plt.show()

# 2.2
# Rescaled process Y = sqrt(deltaT/2a)*X
Y = np.zeros(n)

# Create rescaled Process
def rescaledProcess(deltaT,X,alpha):
    for i in range(0,len(X)):
        Y[i] = np.sqrt(deltaT/(2*alpha))*X[i]
    return Y

# Generation Wiener process 
def WienerProcess(n,deltaT):
    W = np.zeros(n)
    t = np.zeros(n)
    for i in range(n):
        t[i] = i*deltaT
    for k in range(n-1):
        deltaW = np.random.normal(0,t[k+1]-t[k])
        W[k+1] = W[k] + deltaW
    return W

# Wiener Process evaluated at discrete times (t_i = i*deltaT)
def WienerProcessDiscrete(W,deltaT):
    change = np.random.normal(0,deltaT,size=len(W))
    return W + change


fig, axs = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        b = i+j
        axs[i,j].plot(np.arange(0,n)*deltaT[i+j],WienerProcess(n,deltaT[i+j]),label="Wiener process")
        axs[i,j].plot(np.arange(0,n)*deltaT[i+j],rescaledProcess(deltaT[i+j],X,alpha), label = "rescaled process")
        axs[i,j].plot(np.arange(0,n)*deltaT[i+j],WienerProcessDiscrete(WienerProcess(n,deltaT[i+j]),deltaT[i+j]), label = "Wiener Process Discrete")
        axs[i,j].legend()
        axs[i,j].set_title('DeltaT= %i'%i)
fig.suptitle("Show that WienerProcess and Rescaled Process are igual")
plt.show()

# 4.3 Optional
# a)
