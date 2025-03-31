from typing import List, Tuple, Dict, Optional, Union, Type
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

def get_data_paths(
  filename_base: str
) -> List[Path]:
  return sorted(Path('out').glob(f'{filename_base}_[1-9]*.parquet'))

def get_data_col_names(
  filename_base: str
) -> List[str]:
  return pq.read_schema(get_data_paths(filename_base)[0]).names

def get_data_pq(
  filename_base: str,
  filters: Optional[List[Tuple]] = None,
  col_blacklist: List[str] = []
):
  paths = sorted(Path('out').glob(f'{filename_base}_[1-9]*.parquet'))
  col_names = [name for name in get_data_col_names(filename_base) if name not in col_blacklist]
  return pa.concat_tables([pq.read_table(path, filters=filters, columns=col_names) for path in paths]), col_names

def get_data_torch(
  filename_base: str,
  col_target: str,
  col_blacklist: List[str] = [],
  filters: Optional[List[Tuple]] = None,
  dtype: torch.dtype = None,
) -> Tuple[Tensor, Tensor]:
  table, col_names = get_data_pq(filename_base, filters, col_blacklist)
  data_np = np.vstack([table[name].to_numpy() for name in col_names if name != col_target])
  target_np = table[col_target].to_numpy()
  data = torch.tensor(data_np, dtype=dtype).T
  target = torch.tensor(target_np, dtype=dtype)
  return data, target

class PqDataset(Dataset):

  def __init__(
    self,
    data: Tensor,
    target: Tensor
  ) -> None:
    self.data = data
    self.target = target.unsqueeze(1)

  def __len__(
    self
  ) -> int:
    return self.data.shape[0]

  def __getitem__(
    self,
    idx
  ) -> Tuple[Tensor, Tensor]:
    return self.data[idx], self.target[idx]

class Preprocessor():

  def __init__(
    self
  ) -> None:
    self.data_sds, self.data_means = None, None
    self.target_sds, self.target_means = None, None

  def fit(
    self,
    data: Tensor,
    target: Tensor
  ) -> None:
    self.data_sds, self.data_means = torch.std_mean(data, dim=0)
    self.target_sds, self.target_means = torch.std_mean(target.log(), dim=0)

  def transform(
    self,
    data: Tensor = None,
    target: Tensor = None
  ) -> None:
    if data is not None:
      data -= self.data_means
      data /= self.data_sds
    if target is not None:
      target.log_()
      target -= self.target_means
      target /= self.target_sds

  def fit_transform(
    self,
    data: Tensor = None,
    target: Tensor = None
  ) -> None:
    if data is not None:
      self.data_sds, self.data_means = torch.std_mean(data, dim=0)
      data -= self.data_means
      data /= self.data_sds
    if target is not None:
      target.log_()
      self.target_sds, self.target_means = torch.std_mean(target, dim=0)
      target -= self.target_means
      target /= self.target_sds

  def detransform(
    self,
    data: Tensor = None,
    target: Tensor = None
  ) -> None:
    if data is not None:
      data *= self.data_sds
      data += self.data_means
    if target is not None:
      target *= self.target_sds
      target += self.target_means
      target.exp_()

def get_loader(
  filename_base: str,
  col_target: str,
  batch_size_train: int,
  batch_size_test: int = None,
  col_blacklist: List[str] = [],
  filters: Optional[List[Tuple]] = None,
  dtype: torch.dtype = None,
  shuffle: bool = True,
  preprocess: bool = False,
  test_ratio: float = 0.,
  num_workers: int = 0
):
# TODO: I need to better understand how to do these types, Union is not right, it is 'or', between 4 things. There re no unions between 4 things, it's always 2.
# ) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
  ret = ()
  data, target = get_data_torch(filename_base, col_target, col_blacklist, filters, dtype)

  if test_ratio == 0:
    if preprocess:
      preprocessor = Preprocessor()
      preprocessor.fit_transform(data, target)
      ret += (preprocessor,)
    dataset = PqDataset(data, target)
    loader = DataLoader(dataset, batch_size=batch_size_train, shuffle=shuffle, num_workers=num_workers)
    ret = (loader,)+ret
  else:
    size_train = int(len(data)*(1-test_ratio))
    if shuffle:
      perm = torch.randperm(len(data))
      data = data[perm]
      target = target[perm]
    data_train, target_train = data[:size_train], target[:size_train]
    data_test, target_test = data[size_train:], target[size_train:]
    if preprocess:
      preprocessor = Preprocessor()
      preprocessor.fit_transform(data_train, target_train)
      preprocessor.transform(data_test, target_test)
      ret += (preprocessor,)
    dataset_train = PqDataset(data_train, target_train)
    dataset_test = PqDataset(data_test, target_test)
    loader_train = DataLoader(dataset_train, batch_size=batch_size_train, num_workers=num_workers)
    loader_test = DataLoader(dataset_test, batch_size=batch_size_test, num_workers=num_workers)
    ret = (loader_train, loader_test)+ret

  return (ret[0] if len(ret) == 1 else ret)

def model_n_params(
  model: torch.nn.Module
) -> Tuple[int, int]:
  n_params = sum(p.numel() for p in model.parameters())
  n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return n_params, n_params_trainable

def store(
  filename_base: str,
  model: nn.Module,
  preprocessor: Preprocessor
) -> None:
  torch.save(
  { 'model': model.state_dict(),
    'preprocessor':
    { 'data_sds': preprocessor.data_sds,
      'data_means': preprocessor.data_means,
      'target_sds': preprocessor.target_sds,
      'target_means': preprocessor.target_means
    }
  }, Path('out')/f'{filename_base}.pth')

def load(
  filename_base: str,
  type_model: Type[nn.Module]
) -> Tuple[nn.Module, Preprocessor]:
  model = type_model()
  preprocessor = Preprocessor()
  cp = torch.load(Path('out')/f'{filename_base}.pth')
  state_dict_model = cp['model']
  state_dict_preprocessor = cp['preprocessor']
  model.load_state_dict(state_dict_model)
  preprocessor.data_sds = state_dict_preprocessor['data_sds']
  preprocessor.data_means = state_dict_preprocessor['data_means']
  preprocessor.target_sds = state_dict_preprocessor['target_sds']
  preprocessor.target_means = state_dict_preprocessor['target_means']
  return model, preprocessor

if __name__ == '__main__':
  filename_base = 'BSM_E'
  print(f'col names {get_data_col_names(filename_base)}')
  filters = [('M', '>=', 1e-3), ('rel err', '<', 2e-2)]
  col_target = 'M'
  col_blacklist = ['rel err']
  batch_size = 32
  loader, preprocessor = get_data_torch(filename_base, col_target, batch_size, col_blacklist=col_blacklist, filters=filters, preprocess=True)
  print(f'n dataset {len(loader.dataset)}')
  print(f'n batches {len(loader)}')
