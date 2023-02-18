import numpy as np
from matplotlib import pyplot as plt  
# Exercise 4
# Poisson Process with rate lambda
# 4.2
# joint density function of Jump times

# We do it similar to the generating uniform order statistics

n1 = 0
n2 = 80
n = 10
# Define the Poisson Process 
N_t = np.linspace(0,n2,n)
T = 1000

# Define start and endpoint (conditional -> Poisson bridge)
N_t[0] = n1
N_t[-1] = n2

# We generate uniform order statistics
U_n = np.random.uniform(0,1)
X_n = np.zeros(n)
X_n[n-1] = U_n**(1/n)
for j in range(n-2,0,-1):
    X_n[j] = X_n[j+1]*np.random.uniform(0,1)**(1/j)
X_n[0] = X_n[1]*np.random.uniform(0,1)

# Transform uniform sample (0,1) to (0,T) -> F(X)=X/T
X_n = X_n * T

# Jump times have same distribution like ordered sample of size n form the uniform distribution on [0,T]
# We already have the Jump times through uniform sample (0,T)
plt.step(X_n,N_t, label="Poisson Process")
plt.ylabel("N_t")
plt.xlabel("Jump Times")
plt.title("Poisson Process")
plt.legend()
plt.show()