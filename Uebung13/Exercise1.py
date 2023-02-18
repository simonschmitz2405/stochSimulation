import numpy as np
import matplotlib.pyplot as plt  
from scipy import stats
import statsmodels.graphics.tsaplots as sm 
import seaborn as sbn

def f(x):
    prob = np.exp(-(-x[0])**2-(x[1]-x[0]**2)**2)+np.exp(-(x[0]+1)**2-(x[1]+3+x[1]**2)**2)
    return prob

def p(x,mu,sigma):
    return stats.multivariate_normal.pdf(x,mean=mu,cov=sigma)

# Proposal density for Independent Sampler MH
def q1(x,y):
    return 0.5*p(y,[1,1],[[0.5,0],[0,0.5]])+0.5*p(y,[-1,-3],[[0.5,0],[0,0.5]])

# Proposal density for Random Walk MH
def q2(x,y):
    sigma = 0.15
    return p(y,x,[[sigma**2,0],[0,sigma**2]])

w = 0.5

def K(w,x,y):
    return w*q1(x,y)+(1-w)*q2(x,y)

Nsample = 1000
sig = 0.15
x = np.zeros(2)
X = []
a = 0

def MHstep(x0,sig):
    global a
    # we generate density of K using the coin flip
    u = np.random.uniform(0,1)
    if u > w:
        xp = np.random.multivariate_normal(mean = x0, cov = [[sig**2,0],[0,sig**2]])
        wahr = 1
    else:
        xp = 0.5*np.random.multivariate_normal(mean=[1,1],cov=[[0.5,0],[0,0.5]])+ 0.5*np.random.multivariate_normal(mean=[-1,-3],cov=[[0.5,0],[0,0.5]])
        wahr = 0
    accprob = (f(xp)*K(wahr,xp,x0))/(f(x0)*K(wahr,x0,xp)) # acceptance prob
    u = np.random.uniform(size=1)
    if u <= accprob:
        x1 = xp # new point is candidate
        a += 1
    else:
        x1 = x0 # new point is the same as old one
    return x1

for i in range(Nsample):
    x = MHstep(x,sig)
    X.append(np.abs(x)**2)
    
X_total = np.vstack(X)


    
print("Die Acceptance Rate beträgt: ",a/Nsample)
#print(X)

# x = np.linspace(-5,5,100)
# X,Y = np.meshgrid(x,x)
# xx = np.vstack((X.flatten(),Y.flatten())).T

#plt.figure()
#plt.contour(X,Y,pdf)
#plt.contour(X,Y,Z)
#plt.contour(X,Y,np.log(pdf))
#plt.show()

sbn.kdeplot(X_total)
sm.plot_acf(X_total[:,0],label='Autocorrelation X1')
sm.plot_acf(X_total[:,1],label='Autocorrelation X2')
plt.legend() 
plt.show()

        