from sobol_new import *
import numpy as np
from matplotlib import pyplot as plt

# Exercise 1

d = 2
#d = 20

N = 10000


def osci(x):
    w1 = 1/2
    cj = 9/d
    return np.cos(2*np.pi*w1 + np.sum(cj*x))

def prod_peak(x):
    cj = 7.25/d
    wj = 1/2
    np.prod(cj**(-2)+(x-wj)**2)**(-1)
    
def gaussian(x):
    cj = 7.03/d
    wj = 1/2
    return np.exp(-np.sum(cj**2*(x-wj)**2))

def continous(x):
    cj = 2.04/d
    wj = 1/2
    return np.exp(-np.sum(cj*np.abs(x-wj)))

def discontinuous(x):
    cj = 4.3/d
    w1 = np.pi/4
    w2 = np.pi/5
    if x[0]>w1 or x[1]>w2:
        return 0
    else:
        return np.exp(np.sum(cj*x))
    
def simplex(x):
    if np.sum(x)<=1:
        return 1
    else:
        return 0
    

# 1. Crude Monte Carlo

# Estimate error using CLT |mu - mu_cmc| <= C_1-alpha/2*sqrt(Var(Z))/sqrt(N)

X = np.random.uniform(size=(N,d))
I_CMC = 1/N*np.sum(osci(X))
exact = np.real(np.exp(2*np.pi*0.5j)*np.prod(1/(9j/d)*(np.exp(9j/d-1))))
print(exact)

function = []
for i in range(N):
    X = np.random.uniform(size=(i,d))
    function.append(1/N*np.sum(osci(X))-exact)
    
x = np.linspace(1,N)
plt.plot(x,function)
plt.show()
    




# 2. Latin Hypercube Sampling (LHS)
U = np.random.uniform(size=(N,d))
X =[]
for i in range(N):
    X.append(np.random.permutation(U[i,:])-1+U[i,:]/N)

#I_LHS = 1/N*np.sum(osci(X))
#print(I_LHS)
print(exact)


# 3. Quasi Monte Carlo (QMC)

R = generate_points(N,d,0)
print(R)
I_QMC = 1/N*np.sum(osci(R))
print(I_QMC)

# Estimate the error of QMC
k = 100
U = np.random.uniform(size=(k,d))
mean_QMC = []
for j in range(k):
    mean_QMC.append(1/N*np.sum(osci(R+U[j,:])))

meanhat_QMC = 1/(N*k)*np.sum(np.sum(osci(R+U[j,:])))

print(meanhat_QMC)