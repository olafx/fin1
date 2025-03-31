from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['text.usetex'] = True

names = ['S0', 'r', 'q', 'sig', 'T', 'K', 'B']
names_latex = [R'$S_0$', R'$r$', R'$q$', R'$\sigma$', R'$T$', R'$K$', R'$B$']

for i, (name, name_latex) in enumerate(zip(names, names_latex), 1):
  filename = f'test_BSM_EB_1_{i}'
  datapath = Path('out')/f'{filename}.parquet'
  md = pq.read_schema(datapath).metadata
  md = {k.decode(): v.decode() for k,v in md.items()}
  fp = pq.ParquetFile(datapath)
  table = fp.read()
  plt.figure(i)
  plt.plot(table[name], table['V0'], c='black')
  plt.xlabel(name_latex)
  plt.ylabel(R'$V_0$')
  col = table[name].to_numpy()
  plt.xlim(col[0], col[-1])
  plt.tight_layout()
  plt.savefig(Path('out')/f'{filename}.png', bbox_inches='tight', dpi=400)
