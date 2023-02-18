import numpy as np
from matplotlib import pyplot as plt  


# Defining Markov Chain Xn

def Xnstep(p):
    U = np.random.uniform()
    if U <= p:
        return 1
    else:
        return -1
    
def Xn(p,N):
    X = []
    X.append(0) # starting point
    for i in range(N-1):
        X.append(max(X[-1]+Xnstep(p),0))
    return X

def Zn(p,N):
    Z = []
    Z.append(0) # starting point
    for i in range(N-1):
        Z.append(abs(Z[-1]+Xnstep(p)))
    return Z
    
N = 101

#print(Zn(1/8,N))

X = []
Z = []
Number_Simulation = 100
for i in range(Number_Simulation):
    X.append(Xn(1/8,N)[-1])
    Z.append(Zn(1/8,N)[-1])
    
x = np.sort(X)
z = np.sort(Z)

print(x)




# We want to plot pi hat and pi row
k = N*Number_Simulation
 
#plt.hist(x,label='xn')
#plt.hist(z,label='zn')

plt.subplot(2,1,1)
plt.plot(x,np.arange(len(x))/float(len(x)),label='Xn')
plt.plot(z,np.arange(len(z))/float(len(z)),label='Zn')
plt.legend()
plt.title('100')

plt.subplot(2,1,2)

X = []
Z = []
Number_Simulation = 101
for i in range(Number_Simulation):
    X.append(Xn(1/8,N)[-1])
    Z.append(Zn(1/8,N)[-1])
    
x = np.sort(X)
z = np.sort(Z)

plt.plot(x,np.arange(len(x))/float(len(x)),label='Xn')
plt.plot(z,np.arange(len(z))/float(len(z)),label='Zn')
plt.legend()
plt.title('101')

plt.show()

