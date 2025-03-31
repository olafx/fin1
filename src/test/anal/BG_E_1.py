from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['text.usetex'] = True

filename = 'test_BG_E_1'
path = Path('out')/f'{filename}.parquet'

md = pq.read_schema(path).metadata
md = {k.decode(): v.decode() for k,v in md.items()}
fp = pq.ParquetFile(path)
table = fp.read()
r     = table['r'    ][0].as_py()
q     = table['q'    ][0].as_py()
al_p  = table['al_p' ][0].as_py()
al_m  = table['al_m' ][0].as_py()
lam_p = table['lam_p'][0].as_py()
lam_m = table['lam_m'][0].as_py()
T     = table['T'    ][0].as_py()
S0    = table['S0'   ].to_numpy()
K     = table['K'    ][0].as_py()
V0    = table['V0'   ].to_numpy()

print(f'model {md['model']}')
print(f'r {r:.4e}')
print(f'q {q:.4e}')
print(f'al_p {al_p:.4e}')
print(f'al_m {al_m:.4e}')
print(f'lam_p {lam_p:.4e}')
print(f'lam_m {lam_m:.4e}')
print(f'T {T:.4e}')
print(f'option {md['option']}')
print(f'K {K:.4e}')

plt.plot(S0, V0, c='black')
plt.xlabel('$S_0$')
plt.ylabel('$V_0$')
plt.xlim(S0[0], S0[-1])
plt.tight_layout()
plt.savefig(Path('out')/f'{filename}.png', bbox_inches='tight', dpi=400)
