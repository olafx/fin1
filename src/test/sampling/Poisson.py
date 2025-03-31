'''
1st method: Knuth
2nd method: Atkinson, 'The Computer Generation of Poisson Random Variables', 1979
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy import special

lam = 100
N = int(1e5)

np.random.seed(2036-8-12)

x = np.random.poisson(lam, N)
y_ls = np.arange(np.min(x), np.max(x)+1)
y = poisson.pmf(y_ls, lam)

# Knuth
def sample_raw_1():
  ret = -1
  prod_u = 1
  while True:
    u = np.random.uniform()
    prod_u *= u
    ret += 1
    if prod_u <= np.exp(-lam): break
  return ret

# Atkinson
def sample_raw_2():
# There is some minimum lambda for which this works, at least 5 or so, but
# Knuth's algorithm is faster for 5, this is suggested to be used for lambda>30.
  c = .767-3.36/lam
  bet = np.pi/(3*lam)**.5
  alp = bet*lam
  k = np.log(c/bet)-lam
  while True:
    u = np.random.uniform()
# Replaced (1-u)/u by u/(1-u) here, probably Atkinson assumes u in (0,1], we use
# u in [0,1).
    x = (alp-np.log(u/(1-u)))/bet
    n = math.floor(x+.5)
    if n < 0: continue
    v = np.random.uniform()
    y = alp-bet*x
    lhs = y+np.log(v/(1+np.exp(y))**2)
# This loggamma is a standard function in cmath, good to know since infrequently
# used. Used for integers n here, could be a lookup table instead. For this it
# is good to know n scales with lambda, just a bit bigger than lambda is good
# enough, and otherwise just cap it, or calculate the largest possible n
# exactly. If lambda is too large, the Poisson distribution becomes the normal
# distribution, so just use normal samples (with careful rounding).
    rhs = k+n*np.log(lam)-special.loggamma(n+1)
    if (lhs <= rhs): return n

def sample():
# Could also take a case here to sample the normal distribution and round to
# nearest if lambda is truly very large, if performance and latency are really
# that important, but this algorithm does not really get more expensive for
# lambda. It is hard to say at what point the normal approximation becomes
# valid, depends on the usecase, but I can visually see the skew in lambda=100.
  return sample_raw_1() if lam <= 30 else sample_raw_2()

z = np.array([sample() for _ in range(N)])

# Manually assigning the bins avoids empty bins due to the discrete nature of
# the distribution. And offset them by .5 in this way so that they are centered,
# to avoid shifting due to bins [left,right) only collecting left, which would
# shift the histogram unphysically.
x_counts, x_bins = np.histogram(x, bins=np.arange(np.min(x)-.5, np.max(x)+1.5))
z_counts, z_bins = np.histogram(z, bins=np.arange(np.min(z)-.5, np.max(z)+1.5))
plt.figure(1)
plt.hist(x_bins[:-1], x_bins, weights=x_counts, density=True, color='black')
plt.plot(y_ls, y, c='red')
plt.figure(2)
plt.hist(z_bins[:-1], z_bins, weights=z_counts, density=True, color='black')
plt.plot(y_ls, y, c='red')
plt.show()
