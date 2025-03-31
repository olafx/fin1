import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


fp = pq.ParquetFile(Path('out')/'1.parquet')
for i in range(fp.num_row_groups):
  table = fp.read_row_group(i)
  print(table)
