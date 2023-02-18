import numpy as np
import matplotlib.pyplot as plt  

# Exercise 1.1 
n = 2 # size of vector X
X = np.random.uniform(-1,1,size=n) # random vector uniformly distributed over the n-dimensional square
# define random variable Z
def Z_random(X):
    if (np.linalg.norm(X)<1):
        Z = 1
    else:
        Z = 0
    return Z

N = np.array([10,100,1000,10000])

def mean(X):
    return 1/len(X)*np.sum(X)

def sigma(X,mean_estimator):
    return 1/(len(X)-1)*np.sum(X-mean_estimator)

# For each N we calculate the random vector X and then the random variable 

def MC_Z(N):
    Z = np.zeros(N)
    for i in range(N):
        X = np.random.uniform(-1,1,size=n)
        Z[i] = Z_random(X)
    return Z


Z_10 = MC_Z(N[0])
Z_100 = MC_Z(N[1])
Z_1000 = MC_Z(N[2])
Z_10000 = MC_Z(N[3])

mean_estimator_10 = mean(Z_10)
mean_estimator_100 = mean(Z_100)
mean_estimator_1000 = mean(Z_1000)
mean_estimator_10000 = mean(Z_10000)

sigma_estimator_10 = sigma(Z_10,mean_estimator_10)
sigma_estimator_100 = sigma(Z_100,mean_estimator_100)
sigma_estimator_1000 = sigma(Z_1000,mean_estimator_1000)
sigma_estimator_10000 = sigma(Z_10000,mean_estimator_10000)

I = np.pi # since 2-dim square r = 1 and B(0,1)
alpha = 0.05
C = 1.96 # 0.975 Quantil
CDI_10 = np.array([mean_estimator_10-(sigma_estimator_10/np.sqrt(N[0])*C),mean_estimator_10+(sigma_estimator_10/np.sqrt(N[0])*C)])
CDI_100 = np.array([mean_estimator_100-(sigma_estimator_100/np.sqrt(N[1])*C),mean_estimator_100+(sigma_estimator_100/np.sqrt(N[1])*C)])
CDI_1000 = np.array([mean_estimator_1000-(sigma_estimator_1000/np.sqrt(N[2])*C),mean_estimator_1000+(sigma_estimator_1000/np.sqrt(N[2])*C)])
CDI_10000 = np.array([mean_estimator_10000-(sigma_estimator_10000/np.sqrt(N[3])*C),mean_estimator_10000+(sigma_estimator_10000/np.sqrt(N[3])*C)])

relativeerror_10 = np.abs(mean_estimator_10-I)/I
relativeerror_100 = np.abs(mean_estimator_100-I)/I
relativeerror_1000 = np.abs(mean_estimator_1000-I)/I
relativeerror_10000 = np.abs(mean_estimator_10000-I)/I

plt.plot(N,np.array([relativeerror_10,relativeerror_100,relativeerror_1000,relativeerror_10000]))
plt.yscale('log')
plt.show()

# TODO Convergence rate

# 1.2
p = np.pi/4
Z = np.random.binomial(p)
eps = 10e-2
alpha = 10e-4
# non asymptocitic error bounds
# Chebycheffs inequality 
