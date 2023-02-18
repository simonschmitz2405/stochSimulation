import numpy as np
import scipy.stats as sc

# Algorithm 2 with one-at-a-time Sample Variance based SMC

def TMC_lognormal(eps,alpha, N_0):
    # Step 1: Pilot run with N_bar replicas
    Z = np.random.lognormal(0,1,size=N_0)
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    k = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(k))> eps):
        k = k+1
        Z = np.append(Z,np.random.lognormal(0,1))
        sme_alt = sme
        sme = k/(k+1)*sme+1/(k+1)*Z[-1]
        sve = (k-1)/k*sve+1/(k+1)*((Z[-1]-sme_alt)**2)
    return sme, k

def TMC_pareto(eps,alpha, N_0,xm,gamma):
    # Step 1: Pilot run with N_bar replicas
    Z = (np.random.pareto(gamma,N_0)+1)*xm
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    k = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(k))> eps):
        k = k+1
        Z = np.append(Z,(np.random.pareto(gamma)+1)*xm)
        sme_alt = sme
        sme = k/(k+1)*sme+1/(k+1)*Z[-1]
        sve = (k-1)/k*sve+1/(k+1)*((Z[-1]-sme_alt)**2)
    return sme, k

def TMC_u(eps,alpha, N_0):
    # Step 1: Pilot run with N_bar replicas
    Z = np.random.uniform(-1,1,size=N_0)
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    k = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(k))> eps):
        k = k+1
        Z = np.append(Z,np.random.uniform(-1,1,size=N_0))
        sme_alt = sme
        sme = k/(k+1)*sme+1/(k+1)*Z[-1]
        sve = (k-1)/k*sve+1/(k+1)*((Z[-1]-sme_alt)**2)
    return sme, k

alpha = 10**-1.5
eps = 1/10
N_0 = 10
K = 20*alpha**-1
xm = 1
gamma = 3.1

# Define sample sizes
N_pareto = np.ones(int(K))
N_lognormal = np.ones(int(K))
N_Uni = np.ones(int(K))

# Define computed sample means
X_pareto = np.ones(int(K))
X_lognormal = np.ones(int(K))
X_Uni = np.ones(int(K))

# Do the simulation K times and save sample sizes and computed sample means
for i in range(int(K)):
    X_pareto[i], N_pareto[i] = TMC_pareto(eps,alpha,N_0,xm,gamma)
    X_lognormal[i], N_lognormal[i] = TMC_lognormal(eps,alpha, N_0)
    X_Uni[i], N_Uni[i] = TMC_u(eps, alpha, N_0)


# Estimation of probability of failure p_bar

def p_failure(K,eps,expect, X):
    sum = 0
    for i in range(int(K)):
        if(np.abs(X[i]-expect)>eps):
            sum = sum + 1
        else:
            sum = sum + 0
            
    return sum/K

p_pareto = p_failure(K,eps,(gamma*xm)/(gamma-1),X_pareto)
p_lognormal = p_failure(K,eps,np.exp(0.5),X_lognormal)
p_Uni = p_failure(K,eps,0,X_Uni)

print("The estimator of the probabilty of failure of the pareto is: "+ str(p_pareto))
print("The estimator of the probabilty of failure of the lognorml is: "+ str(p_lognormal))
print("The estimator of the probabilty of failure of the Uniform is: "+ str(p_Uni))



