from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as normal
import torch
import ml.utils, ml.BSM_E

filename_base = 'BSM_E'
filters = [('M', '>=', 1e-3), ('rel err', '<', 2e-2)]
re_MC_set = 1e-2
n_bins = 128

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)

table, col_names = ml.utils.get_data_pq(filename_base, filters)

print(f'n samples {(n := len(table))}')
S0_o_K = table['S0/K'   ].to_numpy()
r      = table['r'      ].to_numpy()
q      = table['q'      ].to_numpy()
sig    = table['sig'    ].to_numpy()
T      = table['T'      ].to_numpy()
M_2    = table['M'      ].to_numpy()
re_MC  = table['rel err'].to_numpy()

Phi = normal.cdf
d1 = (np.log(S0_o_K)+(r-q+sig**2/2)*T)/(sig*T**.5)
d2 = d1-sig*T**.5
M_1 = S0_o_K*np.exp(-q*T)*Phi( d1)-np.exp(-r*T)*Phi( d2)
re_21 = np.where(M_1 == 0, 1, np.abs(M_2-M_1)/M_1+1e-9)

data, target = ml.utils.get_data_torch(filename_base, ml.BSM_E.col_target, ml.BSM_E.col_blacklist, filters)
model, preprocessor = ml.utils.load(filename_base, ml.BSM_E.Net)
preprocessor.transform(data, target)
model.eval()
with torch.no_grad():
  M_3 = model(data)
  preprocessor.detransform(target=M_3)
  M_3 = M_3.squeeze().cpu().detach().numpy()
re_31 = np.where(M_1 == 0, 1, np.abs(M_3-M_1)/M_1+1e-9)

def identify_outliers(re, re_threshold):
  i_outliers = np.where(re > re_threshold)[0]
  print(f'{len(i_outliers)}/{n}')
  for i in i_outliers:
    print(f'S0/K {S0_o_K[i]:.1e} r {r[i]:+.1e} q {q[i]:+.1e} sig {sig[i]:.1e} T {T[i]:.1e} M {M_1[i]:.1e} M MC {M_2[i]:.1e} M ML {M_3[i]:.1e}')

print(f'rel err MC-analytical outliers @ {2.5e-2:.2e}')
identify_outliers(re_21, 2.5e-2)
print(f'rel err ML-analytical outliers @ {2e-1:.2e}')
identify_outliers(re_31, 2e-1)

def hist_log(data):
  counts, edges = np.histogram(np.log(data), bins=np.log(10)*np.linspace(-10, 1, n_bins+1))
  centers = .5*(edges[1:]+edges[:-1])
  return counts/n, np.exp(edges), np.exp(centers)

re_MC_counts, re_MC_edges, re_MC_centers = hist_log(re_MC)
re_21_counts, re_21_edges, re_21_centers = hist_log(re_21)
re_31_counts, re_31_edges, re_31_centers = hist_log(re_31)

plt.figure(1)
plt.hist(re_MC_edges[:-1], re_MC_edges, weights=re_MC_counts, color='red')
plt.xlabel('rel err MC estimated')
plt.ylabel('density')
plt.xscale('log')
plt.yscale('log')
plt.savefig(Path('out')/f'anal_{filename_base}_1.png', bbox_inches='tight', dpi=400)
plt.figure(2)
plt.hist(re_21_edges[:-1], re_21_edges, weights=re_21_counts, color='green')
plt.xlabel('rel err MC-analytical')
plt.ylabel('density')
plt.xscale('log')
plt.yscale('log')
plt.savefig(Path('out')/f'anal_{filename_base}_2.png', bbox_inches='tight', dpi=400)
plt.figure(3)
plt.hist(re_31_edges[:-1], re_31_edges, weights=re_31_counts, color='blue')
plt.xlabel('rel err ML-analytical')
plt.ylabel('density')
plt.xscale('log')
plt.yscale('log')
plt.savefig(Path('out')/f'anal_{filename_base}_3.png', bbox_inches='tight', dpi=400)
plt.figure(4)
plt.hist(re_MC_edges[:-1], re_MC_edges, weights=re_MC_counts, color='red', alpha=.5, label='MC estimated')
plt.hist(re_21_edges[:-1], re_21_edges, weights=re_21_counts, color='green', alpha=.5, label='MC-analytical')
plt.hist(re_31_edges[:-1], re_31_edges, weights=re_31_counts, color='blue', alpha=.5, label='ML-analytical')
plt.xlabel('rel err')
plt.ylabel('density')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(Path('out')/f'anal_{filename_base}_4.png', bbox_inches='tight', dpi=400)
