import numpy as np

def Xn(alpha,N):
    X = []
    X.append(0) # starting probability fixed on 0
    for i in range(N):
        U = np.random.uniform()
        if U <= 1-2*alpha:
            X.append(X[-1])
        elif U >= (1-2*alpha+alpha*(1+X[-1]/(2*N**2))):
            X.append(X[-1]-1)
        else:
            X.append(X[-1]+1)
    return X

print(Xn(0.3,100))

# a
n = 100
p = np.array([1,2,4])
alpha = 0.3
# monte carlo estimator
NumberSimulation = 100
X = []
for i in range(NumberSimulation):
    X.append(Xn(alpha,n)[-1])
    
E = (1/NumberSimulation*np.sum(X**p[0]))**(1/p[0])

def M(t):
    if t>1:
        return 'Zahl zwischen -1 und 1'
    elif t<-1:
        return 'Zahl zwischen -1 und 1'
    else: 
        return 1/NumberSimulation*np.sum(np.exp(t*X))

# b
N = 10
P = np.zeros((10,10))
for i in range(N):
    for j in range(N):
        if i == j:
            P[i,j]= 1-2*alpha
        if i == j+1:
            P[i,j] = alpha*(1-i/2*N**2)
        if i == j-1:
            P[i,j] = alpha*(1+i/2*N**2)
            
w,v = np.linalg.eig(P)
print(w)
print(v)
    