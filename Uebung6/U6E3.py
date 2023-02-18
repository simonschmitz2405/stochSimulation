import numpy as np

# Conside discrete time random walk with transition prob

# 3.1
K = 4
alpha = 1/3
T = 10

# Computation of Monte Carlo estimate of P(stopping time < T)

N = 10000 #Number of Simulations
X = 0
Z = [] # Zufallsvariable which gives stoptime
for _ in range(N):
    X = 0
    stoptime = 0
    while X < 4 and stoptime <= 10:
        stoptime = stoptime + 1
        prob = np.random.uniform(0,1)
        if(prob >= 0 and prob<=alpha):
            X = X + 1
        elif(prob > alpha and prob <= 1):
            X = X - 1
    Z.append(stoptime)

# Transform RV of stoptime to RV Z
for i in range(len(Z)):
    if Z[i] <= 10:
        Z[i] = 1
    else:
        Z[i] = 0

mean_MC = 1/N*Z.count(1)
# We calculate the prob by counting how often the RW stopped for the 11 try.
print("Die Wahrscheinlichkeit vor der Zeit T aufzuhören ist: ", mean_MC)

# 3.2
# Using antithetic variate variance reduction technique
# X - Bin(p)

K = 4
a = 1/3
T = 10
N = 100 # N must be even

# psi(x1,..,xn) = I(sum xn >= s)

def psi(X):
    if X > K:
        return 1
    else:
        return 0

# Define random Variable X
def X(N):
    X = 0
    for i in range(N):
        Z = np.random.uniform(0,1)
        if (Z>= 0 and Z <= a):
            X = X+1
        else:
            X = X-1
    return X


        
    
# Expected value of X
mean_X = 1/3*1+2/3*(-1)

mean_AV = 2/N*np.sum((psi(X(N))+psi(N*mean_X-X(N)))/2)

print("Die Wahrscheinlichkeit mit AV vor der Zeit T aufzuhören ist: ", mean_AV)

