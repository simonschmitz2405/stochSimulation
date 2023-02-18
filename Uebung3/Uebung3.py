import numpy as np
import matplotlib.pyplot as plt

# Exercise 1
# Consider multivariant Gaussian Random variable

# 1.1
N = 10**6
mu = np.array([2,1])
cov = np.array([[1,2],[2,5]])
A = np.linalg.cholesky(cov)
Y = np.random.randn(N,2)
X = np.random.randn(N,2)
for i in range(N):
    X[i] = mu.T + np.dot(A,Y[i].T)


# plotting bivariate histogram
    
plt.hist2d(X[:,0], X[:,1], bins=(50,50))
plt.title("1.1")
plt.show()
    

plt.hist(X,alpha=0.5, range=(-20,20), bins=100, label="Independent Random Vectors N(mu,cov)")
plt.title("1D Quality of margin distribution")
plt.show()

# 1.2
# Problem: The Covariance Matrix is not positive definite
# We cannot use the multivariante Gaussian
# Since ad-bc = 0 the matrix has no inverse

# We use the Spectral decomposition

V = np.array([[0.447214,-0.894427],[0.894427,0.447214]])
D_sqrt = np.array([[np.sqrt(5),0],[0,0]])
A_2 = np.dot(V,D_sqrt)
Y_2 = np.random.randn(N,2)
X_2 = np.random.randn(N,2)
for i in range(N):
    X[i] = mu.T + np.dot(A_2,Y[i].T)
    
plt.hist2d(X_2[:,0], X_2[:,1], bins=(50,50))
plt.title("1.2")
plt.show()


# Exercise 2
# 2.1
# We consider a Gaussian Process
def mean_X(t):
    return np.sin(2*np.pi*t)
    
def covar_X(s,t,p):
    return np.exp(-np.abs(t-s)/p)
    
# Generate Gaussian process in set of n points 
n = 100
p = 10
t = np.linspace(0,1,n)
mu_vector = mean_X(t)
cov_Matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        cov_Matrix[i][j] = covar_X(i,j,p)
        

A = np.linalg.cholesky(cov_Matrix)
Y = np.random.randn(n)
X = mu_vector + np.dot(A,Y)

plt.plot(t,X)
plt.show()

# 2.2

p = 1/200
n = 51
t = np.zeros(51)
for i in range(n):
    t[i] = (i-1)/(n-1)
x = np.linspace(0,1,n)

mu_vector = mean_X(t)
cov_Matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        cov_Matrix[i][j] = covar_X(i,j,p)
        

A = np.linalg.cholesky(cov_Matrix)
Y = np.random.randn(n)
Z = mu_vector + np.dot(A,Y)

m = n-1
t_new = np.zeros(m)
for i in range(m):
    t_new[i] = (2*i-1)/(2*(n-1))
    
t_gesamt = np.append(t,t_new)

# TODO Conditioned multivariante Gaussian
mu_vector = mean_X(t_gesamt)

cov_Matrix_gesamt = np.zeros((n+m,n+m))
for i in range(n+m):
    for j in range(n+m):
        cov_Matrix_gesamt[i,j] = covar_X(i,j,p)
        
X_condi = np.random.randn(n+m)
Y = X_condi[:-m]
cov_YZ = cov_Matrix_gesamt[:-m,n-1:]
cov_ZZ = cov_Matrix_gesamt[n-1:,n-1:]
invers_cov_ZZ = np.linalg.inv(cov_ZZ)
Y_con = Y + np.dot(cov_YZ,invers_cov_ZZ)*(Z-X_condi[n-1:])

plt.plot(t_gesamt,Y_con)
plt.show()



# Exercise 3
# Brownian bridge as a Wiener process
# TODO Show that mean function and covariance function

# 3.2
# iterative algo that generates X conditioned
b = 20
X_brownian = np.zeros(n)
for i in range(n):
    X_brownian[i] = Z[i] - t[i]*(Z[n-1]-b)
    
plt.plot(x,X_brownian)
plt.show()



    