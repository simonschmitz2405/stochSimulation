import numpy as np

# Exercise 1
# 1.2

X = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]] # starting points for N_S = 10
N = 4
T = 10
alpha = 0.8 # alpha > 1-alpha : alpha > 0.5
N_S = 10

def psi(X):
    if X == N:
        return 1
    else:
        return 0
    
def w(X):
    product = 1
    for j in range(len(X)):
        product = 0.5/q(X[j-1],X[j])
    return product

def q(X,Y):
    if Y == X+1:
        return alpha
    else: 
        return 1-alpha
    
    
# MC with probability q and starting p0
for i in range(N_S):
    index = 0
    while(X[i][-1] < N and index < T ):
        U = np.random.uniform(0,1)
        index = index + 1
        if(U>=0 and U<alpha):
            X[i].append(X[i][-1]+1)
        else:
            X[i].append(X[i][-1]-1)
    

# Calculate sum in MC estimator
sum = 0
for i in range(N_S):
    sum = sum + psi(X[i][-1])*w(X[i]) # todo product with w(X)
    
# MC estimator
mean_IS = 1/N_S*sum

print(mean_IS)
    