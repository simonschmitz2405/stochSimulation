# Exercise 2
# Pricing a Barrier option with maturity T > 0 based on stock price S

import numpy as np
import matplotlib.pyplot as plt  

T = 2
m = 1000
r = 0.5
sigma = 0.3
K = 10
S0 = 5
B = 3 # different barrier values

deltaT = T/m
ti = []
ti.append(0)
i = 1
# Discrete observation times
while ti[-1] < T:
    ti.append(i*deltaT)
    i = i+1
    
mean_true_St = np.exp(r*T)*S0

N = 100

# TODO indicator for Barrier value

def psi(ST):
    if ((ST[:,-1]-K).all()>= 0 and ST[:,-1].all()>= B):
        return ST[:,-1]-K
    else:
        return 0
X = np.zeros((N,len(ti)))

for i in range(N):
    for j in range(len(ti)):
        X[i,j] = np.random.lognormal(ti[i])

#X = np.random.normal((r-(sigma**2)/2)*T,(sigma**2)*T,size = N)
mean_av = 2/N*np.sum((psi(X)+psi((2*r-sigma**2)*T-X))/2)

print(mean_av)
print(X)


    
plt.plot(ti,X[0,:])
plt.show()
