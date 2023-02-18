import numpy as np
import matplotlib.pyplot as plt 

def f(y):
    return np.exp(-(1/4*y**4-1/2*y**2+1/4))

def check_natural_number(x):
    if type(x) == int and x >= 0:
        return True
    else:
        raise ValueError("Input is not natural number")

def psi(x,p):
    try:
        check_natural_number(p)
    except ValueError as ve:
        print(ve)
    return x**p

Nsample = 1000
sig = 0.15
x = np.zeros(1)
X = []
a = 0

def MHstep(x0,sig):
    global a
    xp = np.random.normal(x0,scale=sig**2)
    accprob = f(xp)/f(x0) # acceptance prob
    u = np.random.uniform(size=1)
    if u <= accprob:
        x1 = xp # new point is candidate
        a += 1
    else:
        x1 = x0 # new point is the same as old one
    return x1

for i in range(Nsample):
    x = MHstep(x,sig)
    X.append(x)

X_MH = np.vstack(X)
print(X_MH)

t = np.linspace(-4,4,100) 
plt.plot(t,f(t))
plt.show()
plt.hist(X_MH,bins=100)
plt.show()
