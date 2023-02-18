import numpy as np

# Exercise 2
# Random boundary value problem

d = 3
# i
def a1(x):
    mu = 1
    sigma = 4
    Y = np.random.uniform(low=-1,high=1,size=d)
    n = np.linspace(1,d,d)
    return mu + sigma/np.pi**2*np.sum(np.cos(np.pi*n*x)/n**2*Y)

def a2(x):
    Y = np.random.normal(0,1,size=d)
    n = np.linspace(1,d,d)
    return np.exp(x+np.sqrt(2)*np.sum((np.sin((n-0.5)*np.pi*x))/(n-0.5)*np.pi)*Y)


print(a2([1,1,1]))