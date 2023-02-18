import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 

# define function

def mu(t):
    return np.sin(2*np.pi*t)

def Cov(t,s,rho = 1/20):
    return np.exp(-np.abs(t-s)/rho)


# define parameters
N = 50
dt = 1/N
t = np.linspace(0,1,N)
muvec = mu(t)
sigma = np.zeros((N,N))
rho = 1/10
for i in range(N):
    for j in range(N):
        sigma[i,j] = Cov(t[i],t[j],rho)
        
print(np.linalg.det(sigma)) # check if its singular
plt.contourf(sigma,50,cmap='Spectral')
plt.title(r'Covariance, $\rho=' +str(rho)+'$')
plt.show()

U,D,V = np.linalg.svd(sigma)
A = U@np.diag(np.sqrt(D))
proc = muvec + A@np.random.standard_normal(N)
plt.plot(t,proc)
plt.title(r'$\mu(t)=\sin(2\pi t)$')
plt.show()

# add new data by adding on midpoint
tall = np.linspace(0,1,N*2)
Nall = len(tall)
M = Nall-N
zobs = proc
idz = np.arange(0,Nall,2)
idy = np.arange(1,Nall,2)
idP = np.concatenate((idy,idz))
P = np.eye(Nall)
P = P[idP,:]
mu_all = mu(tall)
sigma_all = np.zeros((Nall,Nall))

for i in range(Nall):
    for j in range(Nall):
        sigma_all[i,j] = Cov(tall[i],tall[j],rho)
        
sigmaX = P@sigma_all@np.transpose(P)
muX = P@mu_all
muY = muX[0:M]
muZ = muvec

sigmaYY = sigmaX[:M,:M]
sigmaYZ = sigmaX[:M,M:]
sigmaZZ = sigmaX[M:,M:]

muYgZ = muY + sigmaYZ@(np.linalg.solve(sigmaZZ,zobs-muZ))
sigmaYgZ = sigmaYY - sigmaYZ@(np.linalg.solve(sigmaZZ,sigmaYZ.T))

U,D,V = np.linalg.svd(sigmaYgZ)
A = U@np.diag(np.sqrt(D))
Y = muYgZ+A@np.random.standard_normal(M)
X = np.concatenate((Y,zobs))
proc_all = np.linalg.solve(P,X)

tg = np.linspace(0,1,10*N)

plt.plot(t,proc, '-o')
plt.plot(tall,proc_all, '-*')
plt.plot(tg,mu(tg),'--',color='k')
plt.show()