import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

np.random.seed(2036-8-12)

n = int(1e3) # number of elements
m = 2 # number of chunks when writing

a = np.random.uniform(0, 2/n, n).cumsum().astype(np.float32)
b = np.arange(n, dtype=np.uint64)
df = {'a':a, 'b':b}

# table = pa.Table.from_pandas(pd.DataFrame(df))
# table = pa.Table.from_arrays([pa.array(a) for a in df.values()], names=list(df.keys()))
table = pa.Table.from_pydict(df)
print(table)
pq.write_table(table, Path('out')/'1.parquet', row_group_size=n//m)
table = pq.read_table(Path('out')/'1.parquet', columns=['a'])
print(table)

# Now read a particular chunk.
fp = pq.ParquetFile(Path('out')/'1.parquet')
for i in range(fp.num_row_groups):
  table = fp.read_row_group(i)
  print(table)

# Chunk data writing from Python is unnecessary.
