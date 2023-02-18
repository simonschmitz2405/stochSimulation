from cgi import test
from random import uniform
from re import L
import scipy.stats 
from scipy.stats import uniform
import matplotlib.pyplot as plt  
import numpy as np
import scipy.stats as ss


# Exercise 1
n = 10000

# Aufgabe 1.1
x = np.linspace(0,1,n)
# theoretical CDF
theo_y = ss.uniform.cdf(x)
plt.plot(x,theo_y)
# empirical CDF
u = uniform.rvs(size=n)
sort_u = np.sort(u)
plt.plot(x,sort_u)
plt.show()

# Produce Q Q Plot of data
#fig = sm.qqplot(u,color='r',linewidth=1)
plt.show()

# Aufgabe 1.2
# Implement Kolmogorov Smirnov test
alpha = .1
# We have to calculate the supremum (D_n)
max = 0
index = 0
for i in sort_u:
    d = abs(theo_y[index]-i)
    index = index + 1
    if(d>=max):
        max = d

K_alpha = 1.63
teststatistic = np.sqrt(n)*max
if(teststatistic>K_alpha):
    print("We reject Nullhypothesis on level alpha")
else:
    print("We cannot reject nullhypothesis")

print(teststatistic)

# Aufgabe 1.3 
# Implement X^2 goodness of fit test
I = np.linspace(0,1,n)
k = 5
u_split = np.array_split(I,k)






