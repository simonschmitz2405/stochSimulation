import numpy as np
import matplotlib.pyplot as plt  
import scipy.linalg as linalg
import time
from matplotlib import rc  

plt.style.use('ggplot')

N = int(1e6)
mu = np.array([2,1])
Sigma = np.array([[1,2],[2,5]])
p = 2

# Generate samples based on Cholesky Factorization

mu = np.ones((p,N))
mu[0,:] = 2.0*np.ones(N)
mu[1,:] = 1.0*np.ones(N)

tstart = time.time()
A = np.linalg.cholesky(Sigma)
xi = np.zeros([2,N])
xi[0,:] = np.random.standard_normal(N)
xi[1,:] = np.random.standard_normal(N)
X = mu + A@xi
telapsed = time.time()-tstart
print('time needed ' +str(telapsed)+ ' '+ 'seconds')

# Compute 2D histogram

plt.figure(1)
plt.hist2d(X[0,:],X[1,:], bins = 70, cmap = 'Spectral')
plt.colorbar()
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title('2d Histogram Cholesky')


# Samples based on eigenvalue decomposition

Sigma = np.array([[1,2],[2,4]])
tstart = time.time()
U,s,Vh = linalg.svd(Sigma)
A = U*np.sqrt(s)
xi = np.zeros([2,N])
xi[0,:] = np.random.standard_normal(N)
xi[1,:] = np.random.standard_normal(N)
X = mu + A@xi
telapsed = time.time()-tstart
print('time needed '+str(telapsed) + ' '+'seconds')

# compute 2D histogram

plt.figure(2)
plt.hist2d(X[0,:],X[1,:], bins = 70, cmap = 'Spectral')
plt.colorbar()
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title('2D histogram, SVD')

plt.show()
