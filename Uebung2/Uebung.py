from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time as tm
from scipy.stats import burr12

# Exercise 1
'''''
Implement the inverse-transform method (Continuous)
1. Generate U - Uni(0,1)
2. Set X = F^-1(U)
'''''
def theo_cdf(x):
    return 1-2/3*np.exp(-x/2)

# def generated_cdf(y):
#     if(y==0):
#         return -1
#     if(y>0 and y<1/3):
#         return 0
#     if(y>= 1/3 and y <= 1-2*np.exp**(-1)/3):
#         return -2*np.log(3/2-3/2*y)
#     if(y> 1-2*np.exp**(-1)/3 and y <= 1):
#         return 2
    

# def inverse_transform(n):
#     U = uniform.rvs(size=n)
#     #X = generated_cdf(U)
#     actual1 = expon.rvs(size=n,scale=2)
#     #actual = theo_cdf(np.linspace(0,2,n))
#     plt.figure(figsize=(12,9))
#     plt.hist(X, bins=50, alpha = 0.5, label="Generated rv.")
#     #plt.hist(actual,bins=50,alpha=0.5,label="Actual rv.")
#     plt.title("Generated vs Actual")
#     plt.legend()
#     plt.show()
#     return X

#plt.plot(generated_cdf(np.linspace(-1,3,100)))
#plt.show()

#inverse_transform(5000)
    
# TO DO Check Distribution, check intervall

# Exercise 2
'''''
Acceptance Rejection Method
1. Generate Y from proposal g
2. Generate U - Uni(0,1) independently of Y
3. If U <= f(y)/C*g(y) -> Set X=Y, otherwiese step 1

'''''
# 2.1 
def f(x):
    return (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)*np.exp(-x**2/2)

def g(x):
    return np.exp((-x**2)/2)/np.sqrt(2*np.pi)
k = 0.169
'''''
Since the function f has a maximum value less than 5 and looks like normal distribution
Therefore there must exist an C>0 so that f<=C*g
C = sup(f/g)
'''''

# 2.2
# Generate n = 10^4 random variable according PDF f using AR Method
C = 5*np.sqrt(2*np.pi)
# Y = norm.rvs(size=10**6)
# U = uniform.rvs(size=n)
# X = np.ones(n)
# index = 0
# for i in Y:
#     if U[index]<= f(i)/C*g(i):
#         X[index]=i
#         index = index + 1
#         if(index == n):
#             break 
index = 0
n=0
X = np.ones(10**2)
while(n<10**2):
    Y = norm.rvs(size=1)
    U = uniform.rvs(size=1)
    if (U<=f(Y)/C*g(Y)):
        X[index] = Y
        index = index + 1
        n = n+1


plt.hist(X,alpha=0.5, label="Generated normal")
plt.title("Random Variable")
plt.legend()
plt.show()

# 2.3
# Derive estimate of the normalization constant k

# acceptance probability
accep_prob = 1/C*k
print(accep_prob)
# k = 1/C*Prob(U<=f/C*g)

# Exercise 3
'''''
Box Muller Method 
1. Generate U - Uni(0,1) and set p = sqrt(-2*ln(U)) -> p^2 - Exp(1/2)
2. Generate V - Uni(0,1) and set thau = 2*pi*V -> thau - Uni(0,2*pi)
3. Set X = p*cos(thau) and Y = p*sin(thau)
'''''

# 3.1 X,Y - N(0,1) independent -> p and thau independent

'''''
p^2 = x^2 + y^2 - chi_2^2 and this = Exp(1/2)
by radical symmetry of the bivariant N(0,I_2) the distribution of (X,Y) is uniform on [0,2pi]
TODO: Independency
'''''

# 3.2 Opposite direction of 3.1

# 3.3 

def aux_pdf(x):
    return np.exp(-np.abs(x)/2)


# 3.4 
# Implementation of the AR and Box Muller method

def Box_Muller(n):
    U = uniform.rvs(size=n,loc=0,scale=1)
    p = np.sqrt(-2*np.log(U))
    V = uniform.rvs(size=n,loc=0,scale=2*np.pi)
    thau = 2*np.pi*V
    X = p*np.cos(thau)
    Y = p*np.sin(thau)
    return X,Y
anfang = tm.time()
X,Y = Box_Muller(500)
ende = tm.time()

print("Das ist X:",X)
print("Das ist Y:",Y)
print(ende-anfang)

plt.hist(X,alpha=0.5,label="X")
plt.hist(Y,alpha=0.5,label="Y")
plt.legend()
plt.show()

# Exercise 4
# 4.1 Implement a kernel density estimator
n=100
delta = n**(-1/5)
def K(x):
    return (np.exp(-x**2/2))/np.sqrt(2*np.pi)

X = burr12.rvs(1,2,4,size=n)

def K_delta(x,delta):
    return 1/delta*K(x/delta)

def f(x):
    return 1/n*np.sum(K_delta(x-X,delta))
    
# 4.2 
# n = 100
# x = np.linspace(-100,100,n)
# F = x
# for i in len(X):
#    F[i] = f(X[i]) 
# plt.plot(x,F)
# plt.show()

# Exercise 5 
'''''
Composition Method
1. Generate r.v. Y with P(Y=i)=p_i
2. Generate sample X - Fy (e.g. by inversion

Alias Method
1. Generate Y = ceiling(n*U)
2. Generate X from Gy (Bernoulli)
'''''
# 5.3
def pdf_cauchy(x,x_o,gamma):
    return 1/(np.pi*gamma*(1+((x-x_0)/gamma)**2))



    






