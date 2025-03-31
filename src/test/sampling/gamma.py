'''
Marsaglia, Tsang, 'A Simple Method for Generating Gamma Variables', 2001
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Should try alpha<1, but keep in mind the distribution funtion diverges at the
# origin if alpha=0, and quickly becomes extreme for small alpha.
alp = 1.18
th = 1/10.57
N = int(1e5)
N_bins = 64

np.random.seed(2036-8-12)

x = np.random.gamma(alp, th, N)
y_ls = np.linspace(np.min(x), np.max(x), N_bins*10)
y = gamma.pdf(y_ls, alp, loc=0, scale=th)

def sample_raw(alpha):
  d = alpha-1/3
  c = 1/(9*d)**.5
  while True:
    v = 0
    while v <= 0:
      x = np.random.normal()
      v = 1+c*x
    v = v**3
    u = np.random.uniform()
    if u < 1-.0331*x**4:
      return d*v
    if np.log(u) < .5*x**2+d*(1-v+np.log(v)):
      return d*v

def sample():
  if alp >= 1:
    return th*sample_raw(alp)
  u = np.random.uniform()
  return th*sample_raw(1+alp)*u**(1/alp)

z = np.array([sample() for _ in range(N)])

x_counts, x_bins = np.histogram(x, bins=N_bins)
z_counts, z_bins = np.histogram(z, bins=N_bins)
plt.figure(1)
plt.title('numpy')
plt.hist(x_bins[:-1], x_bins, weights=x_counts, density=True, color='black')
plt.plot(y_ls, y, c='red')
plt.figure(2)
plt.title('mine')
plt.hist(z_bins[:-1], z_bins, weights=z_counts, density=True, color='black')
plt.plot(y_ls, y, c='red')
plt.show()
