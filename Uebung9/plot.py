from matplotlib import pyplot as plt
import numpy as np
  

def f(N):
    return np.log10(N)**7/N

def g(N):
    return 1/np.sqrt(N)

x = np.linspace(1,1000000)
plt.loglog(x,f(x),scaley="log")
plt.loglog(x,g(x),scaley="log")
plt.show()