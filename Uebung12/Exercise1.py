import numpy as np
from matplotlib import pyplot as plt
import arviz as az


gamma = 1
x_null = [1,4,9,25] 
Nsample = 10000
sig = 2

def bimodal(x,gamma,x0):
    return np.exp(-gamma*(x**2-x0)**2)

x = np.linspace(-4,4,100)
plt.plot(x,bimodal(x,1,1),label='gamma=1,x0=1')
plt.plot(x,bimodal(x,10,1),label = 'gamma=10,x0=1')
plt.plot(x,bimodal(x,1,10),label='gamma=1,x0=10')
plt.plot(x,bimodal(x,10,10),label='gamma=10,x0=10')

plt.legend()
plt.show()


def MHstep(x0,sig,gamma,x_null):
    xp = np.random.normal(loc=x0,scale=sig) # generate candidate
    accprob = bimodal(xp,gamma,x_null)/bimodal(x0,gamma,x_null) # acceptance prob
    u = np.random.uniform(size=1)
    if u <= accprob:
        x1 = xp # new point is candidate
        a = 1 # note acceptance
    else:
        x1 = x0 # new point is the same as old one
        a = 0 # note rejections
    return x1, a



def MHSample(Nsample,x0):
    X = []
    x = 0
    for i in range(Nsample):
        x,a = MHstep(x,sig,gamma,x_null=x0)
        X.append(x)
    return X
    
fig, ax = plt.subplots(2,2)
ax[0,0].hist(MHSample(Nsample,x_null[0]),label='x01')
ax[0,0].set_title('x01')
ax[0,1].hist(MHSample(Nsample,x_null[1]),label='x04')
ax[0,1].set_title('x04')
ax[1,0].hist(MHSample(Nsample,x_null[2]),label='x09')
ax[1,0].set_title('x09')
ax[1,1].hist(MHSample(Nsample,x_null[3]),label='x025')
ax[1,1].set_title('x025')


# plt.hist(MHSample(Nsample,x_null[0]),label='x01')
# plt.show()
# plt.hist(MHSample(Nsample,x_null[1]),label='x04')
# plt.show()
# plt.hist(MHSample(Nsample,x_null[2]),label='x09')

# plt.hist(MHSample(Nsample,x_null[3]),label='x025')
plt.show()

az.style.use('arviz-doc')
az.plot_trace(np.array(MHSample(Nsample,x_null[0])))
plt.show()
