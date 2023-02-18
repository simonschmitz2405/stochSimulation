import numpy as np
import scipy.stats as sc  
import matplotlib.pyplot as plt  

def SMC_pareto(eps,alpha, N_0, xm, y):
    # Step 1: Pilot run with N_bar replicas
    # f_x(x) = y*xm^y*x^(-y+1)
    # F_x(x) = 1-(xm/x)^y
    # inverse transformation
    # U = F_x(x) -> 1-U = (xm/x)^y
    # (1-U)^(1/y) = xm/x -> x = xm/(1-)
    U = np.random.uniform(0,1,size=N_0)
    Z = xm/((1-U)**(1/y))
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    N = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(N))> eps):
        N = 2*N
        Z = np.random.uniform(0,1,size=N)
        X = xm/(1-Z)**(1/y)        
        sme = 1/N*np.sum(Z) #sample mean estimator
        sve = 1/(N-1)*np.sum((Z-sme)**2) #sample variance estimator
    return sme, N

def SMC_pareto_new(eps,alpha, N_0,xm,gamma):
    # Step 1: Pilot run with N_bar replicas
    Z = (np.random.pareto(gamma,N_0)+1)*xm
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    N = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(N))> eps):
        N = 2*N
        Z = (np.random.pareto(gamma,N)+1)*xm
        sme = 1/N*np.sum(Z) #sample mean estimator
        sve = 1/(N-1)*np.sum((Z-sme)**2) #sample variance estimator
    return sme, N

def SMC_lognormal(eps,alpha, N_0):
    # Step 1: Pilot run with N_bar replicas
    Z = np.random.lognormal(0,1,size=N_0)
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    N = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(N))> eps):
        N = 2*N
        Z = np.random.lognormal(0,1,size=N)
        sme = 1/N*np.sum(Z) #sample mean estimator
        sve = 1/(N-1)*np.sum((Z-sme)**2) #sample variance estimator
    return sme, N

def SMC_U(eps,alpha, N_0):
    # Step 1: Pilot run with N_bar replicas
    Z = np.random.uniform(-1,1,size=N_0)
    sme_bar = 1/N_0*np.sum(Z) #sample mean estimator
    sve_bar = 1/(N_0-1)*np.sum((Z-sme_bar)**2) #sample variance estimator
    # Step 2
    N = N_0
    sme = sme_bar
    sve = sve_bar
    # Step 3
    C = sc.norm.ppf(1-alpha/2)
    while((sve*C/np.sqrt(N))> eps):
        N = 2*N
        Z = np.random.uniform(-1,1,size=N)
        sme = 1/N*np.sum(Z) #sample mean estimator
        sve = 1/(N-1)*np.sum((Z-sme)**2) #sample variance estimator
    return sme, N


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
    X_pareto[i], N_pareto[i] = SMC_pareto_new(eps,alpha,N_0,xm,gamma)
    X_lognormal[i], N_lognormal[i] = SMC_lognormal(eps,alpha, N_0)
    X_Uni[i], N_Uni[i] = SMC_U(eps, alpha, N_0)


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

