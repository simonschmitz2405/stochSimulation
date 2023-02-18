import numpy as np
import matplotlib.pyplot as plt  
from matplotlib import rc 


def brownian_bridge(t,a,b):
    n = np.size(t)-2
    X = np.zeros(n+2)
    X[0] = a
    X[-1] = b
    tfinal = t[-1]
    Z = np.random.standard_normal(n,)
    for k in range(1,n+1):
        mu = X[k-1] + (b-X[k-1])*(t[k]-t[k-1])/(tfinal-t[k-1])
        sig2 = (tfinal-t[k])*(t[k]-t[k-1])/(tfinal-t[k-1])
        X[k] = mu+np.sqrt(sig2)*Z[k-1]
    return X
a = np.pi
b = np.sqrt(5)
n = 1000
t = np.linspace(0,1,n)
X = brownian_bridge(t,a,b)
plt.plot(t,X,label='Path')
plt.plot([0,1],[a,b],marker='X',linestyle='None', label='Initial and final points')

plt.xlabel("t")
plt.ylabel("t")
plt.legend('Brownian Brigde')
plt.show()