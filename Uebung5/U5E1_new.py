import numpy as np
import matplotlib.pyplot as plt  
import scipy.stats as sc 
from scipy.special import gamma

# 1.1
n = 2

# Define Random variable Z
def Z(n):
    X = np.random.uniform(-1,1,size=n)
    norm = np.linalg.norm(X)
    if (norm<1):
        Z = 1
    else:
        Z = 0
    return Z


# Monte Carlo Simulation dependend on Number of Simulation
def MC(N):
    Z_vector = []
    for _ in range(N):
        random = Z(n)
        Z_vector.append(random)
    I_bar = 1/N*np.sum(Z_vector)
    return I_bar


# Vector of number of Simulations
N = np.array([10,100,1000,10000])

# Define the I_bar for each N (Simulationnumber)
I_bar = []
for i in N:
    I_bar.append(MC(i))
    
# Exact value of I
rho = 2*2
B = np.pi
I = 1/rho*B
    
# Define the relative Error
rel_error = np.zeros(len(I_bar))
for i in range(len(I_bar)):
    rel_error[i] = np.abs(I_bar[i]-I)/I

# We plot the relative error np.abs(I_bar-I)/I versus N in logarithmic scale
plt.plot(N,rel_error, marker='o', label= 'n=2')
plt.yscale('logit')
plt.ylabel('relative Error')
plt.xlabel('Number of Simulations')
plt.title('Convergence Rate for n = 2')
plt.legend()
#plt.show()
    
# Verifying the convergence rate
# TODO O(N**(-0.5)) factor in logscale

# 1.2
#N = 10
p = np.pi/4
#Z = np.random.binomial(p)
eps = 10**-2
alpha = 10**-4
# Disadvantage: The two non-asymptotic error bounds (Cebycheff, Berry Esseen) are not really sharp
# Advantage: They work for small sample size and for distributions far away from N(0,1)

# Chebycheff inequality
# P(np.abs(error)<= sigma/np.sqrt(N*alpha))>= 1-alpha
# we need sigma/np.sqrt(N*alpha) <= eps
# N>= (sigma/eps)**2/alpha

# Berry Esseen Theorem
# 


# Disadvantage: We need big sample or distributin similar to N(0,1)
# Advantage: Easy to calculate with asymptotic 
# Leap of faith
# We find error bound for 
C = sc.norm.ppf(1-alpha/2)
var = p*(1-p) # since Z is distributed like Bernoulli(p)
bound = C*np.sqrt(var/N)





# 1.3
n = 6

# Define the I_bar for each N (Simulationnumber)
I_bar = []
for i in N:
    I_bar.append(MC(i))
    
# Exact value of I
I = (1/(2**n))*(np.pi**(n/2)/gamma(n/2+1))
    
# Define the relative Error
rel_error = np.zeros(len(I_bar))
for i in range(len(I_bar)):
    rel_error[i] = np.abs(I_bar[i]-I)/I

# We plot the relative error np.abs(I_bar-I)/I versus N in logarithmic scale
plt.plot(N,rel_error, marker='o',label = 'n=6',color = 'b')
plt.yscale('log')
plt.ylabel('relative Error')
plt.xlabel('Number of Simulations')
plt.legend()
plt.show()