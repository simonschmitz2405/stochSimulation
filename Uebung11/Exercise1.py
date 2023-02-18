import numpy as np
from scipy import constants
from scipy.stats import randint
import matplotlib.pyplot as plt

T = 100
beta = 1/3
#beta = 1/(constants.Boltzmann*T)
S = np.array([[1,1],[1,1]])
Nsample = 1000
m = 50
J = 1
B = 10

# Define random initial state
S = np.zeros((m+2,m+2),dtype='int')
for i in range(1,m+1):
    for j in range(1,m+1):
        u = randint.rvs(0,2)
        if u == 1:
            S[i,j] = 1
        else:
            S[i,j] = -1
        S[0,j]=0
        S[j,0]=0
        S[m+1,j]=0
        S[j,m+1]=0
            


# def MetroHast(number_steps, number_atoms, beta, J, B, initial_state):
#     '''
#     Implementation of Metropolis-Hastings algorithm for Ising model
#     '''
#     S =1 # Matrix of current state
#     summe = 0
#     for i in range():
#         1
#     return 1
#     #return list of energy, mean_magnetic_moment, final_configuration


def MagneticMoment(S):
    summe = 0
    for i in range(1,m-1):
        for j in range(1,m-1):
            summe += S[i,j]/m**2
    return summe

def MagneticMoment(S):
    return np.sum(S)/m**2
            


def H(S):
    summe = 0
    for i in range(1,len(S[:,0])-1):
        for j in range(1,len(S[:,0])-1):
            summe += (1/2*J*S[i,j]*(S[i-1,j]+S[i+1,j]+S[i,j-1]+S[i,j+1])+B*S[i,j])
    return summe

def targetdist(S):
    return np.exp(-H(S)*beta)

def MHstep(S, i):
    xp = randint.rvs(1,len(S[:,0]-1),size=2) # generate candidate (index)
    Sp = S.copy()
    Sp[xp] = S[xp]*(-1) # flip candidate
    accprob = targetdist(Sp)/targetdist(S) # acceptance prob
    u = np.random.uniform(size=1)
    if u <= accprob:
        S1 = Sp # new point is candidate
    else:
        S1 = S # new point is the same as old one
        print(i)
    return S1

X = []
magenticmom = []
for i in range(Nsample):
    S = MHstep(S,i)
    #print(H(S))
    #print(MagneticMoment(S))
    magenticmom.append(MagneticMoment(S))
    X.append(S)
#X=np.array(X, dtype=int)    
print(X[-1])

averageMagneticMoment = 1/Nsample*np.sum(MagneticMoment(X))
print(averageMagneticMoment)
    
plt.plot(magenticmom)
plt.show()