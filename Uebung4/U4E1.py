# Exercise 1
# Fractional Brownian Motion centred Gaussian process

import numpy as np
from scipy import linalg

# 1.1
# Define the Hurst index
H = 0.2

# Gaussian process uniquely determined:
# mean function and
# symmetric non-negative definte covariance function
# non negative definite: for all n the matrix is non negative definte

'''
E(W^tilde_t) = E(W_t+h - W_t) = E(W_t+h) - E(W_t) = 0
C(s,t) = Cov(W^tilde_t,W^tilde_s) = E(W^tilde_t*W^tilde_s) =
E((W^tilde_s-W^tilde_0)*(W^tilde_t-W^tilde_s))+E(W^tilde^2_s) =
E(W^tilde^2_s) = s for t<=s
'''

# 1.2 
# We have to show that the Gaussian process/random field is weakly stationary
# C_x(t,s) only depends on s-t
# True since covariance function given

# Brownian Motion with n timesteps
n = 4

# Define the two fixed distance parameter h
h_1 = 1/n
h_2 = 0.6

#Define the omega Matrix
E_H1 = np.zeros((n,n))

# Function to define cov Matrix
def covar_X(s,t,H):
    return 0.5*(np.abs(t)**(2*H) + np.abs(s)**(2*H)-np.abs(t-s)**(2*H))

# Sigma vector
# sigma_h1 = np.zeros(n)
# sigma_h2 = np.zeros(n)
# for i in range(n):
#     for j in range(n):
#         E_H1[i][j] = covar_X(0,(j-i)*h_1,H)


# We define the Sigma values
sigma = np.zeros(n+1)
for i in range(n+1):
    sigma[i] = 1/2*(np.abs(h_1-i*h_1)**2*H+np.abs(i*i*h_1)**2*H-np.abs(-i*h_1)**2*H)

# Toeplitz Matrix
toep = linalg.circulant(sigma)

# Circular embedding
sigma_cir = np.append(sigma,sigma[2::-1])
toep_cir = linalg.toeplitz(sigma_cir)



