from ml.utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

filename_base = 'BSM_E'
filters = [('M', '>=', 1e-3), ('rel err', '<', 2e-2)]
col_target = 'M'
col_blacklist = ['rel err']

class Net(nn.Module):

  def __init__(
    self
  ) -> None:
    super(Net, self).__init__()

    self.l1 = nn.Linear(5, 10)
    self.l2 = nn.Linear(10, 10)
    self.l3 = nn.Linear(10, 10)
    self.l4 = nn.Linear(10, 5)
    self.l5 = nn.Linear(5, 1, bias=False)

    nn.init.zeros_(self.l1.bias)
    nn.init.zeros_(self.l2.bias)
    nn.init.zeros_(self.l3.bias)
    nn.init.zeros_(self.l4.bias)
    # nn.init.zeros_(self.l5.bias)

  def forward(
    self,
    x: torch.Tensor
  ) -> torch.Tensor:
    x = F.gelu(self.l1(x))
    x = F.gelu(self.l2(x))
    x = F.gelu(self.l3(x))
    x = F.gelu(self.l4(x))
    x = self.l5(x)
    # x = x.sinh()
    return x

def train(
  model: Net,
  device: torch.device,
  loader_train: DataLoader,
  optimizer: optim.Optimizer
) -> List[float]:
  losses = []
  model.train()
  for data, target in loader_train:
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    out = model(data)
    # loss = F.l1_loss(out, target)
    # loss = F.l1_loss((out-target).exp(), torch.ones_like(target))
    preprocessor.detransform(target=out)
    preprocessor.detransform(target=target)
    loss = F.mse_loss(out/target, torch.ones_like(target))
    losses += [loss.item()]
    loss.backward()
    optimizer.step()
  return losses

def test(
  model: Net,
  device: torch.device,
  loader_test: DataLoader,
  preprocessor: Preprocessor
) -> Tuple[List[float], Dict[str, List[float]]]:
  losses = []
  metrics = {'rel L1': []}
  model.eval()
  with torch.no_grad():
    for data, target in loader_test:
      data, target = data.to(device), target.to(device)
      out = model(data)
      losses += [F.l1_loss(out, target).item()]
      preprocessor.detransform(target=out)
      preprocessor.detransform(target=target)
      metrics['rel L1'] += [F.l1_loss(out/target, torch.ones_like(target)).item()]
  return losses, metrics

if __name__ == '__main__':

  test_ratio = .25
  batch_size_train = 32
  batch_size_test = 128
  n_epochs = 64
  device = torch.device('cpu')

  loader_train, loader_test, preprocessor = get_loader(filename_base, col_target, batch_size_train, batch_size_test, col_blacklist=col_blacklist, filters=filters, preprocess=True, test_ratio=test_ratio, num_workers=1)
  n_batches_train = len(loader_train)
  n_batches_test = len(loader_test)
  model = Net().to(device)
  n_params, n_params_trainable = model_n_params(model)
  print(f'n params {n_params} trainable {n_params_trainable}')
  optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.8, 0.99))
  scheduler = StepLR(optimizer, step_size=5, gamma=.65)
  epoch_losses_train, epoch_losses_test, epoch_metrics_test = [], [], []
  lr = []
  for i_epoch in range(n_epochs):
    lr += [scheduler.get_last_lr()[0]]
    losses_train = train(model, device, loader_train, optimizer)
    losses_test, metrics_test = test(model, device, loader_test, preprocessor)
    epoch_losses_train += [losses_train]
    epoch_losses_test += [losses_test]
    epoch_metrics_test += [metrics_test]
    scheduler.step()
  store(filename_base, model, preprocessor)
  for i_epoch in range(n_epochs):
    plt.figure(1)
    plt.plot(i_epoch+np.linspace(0, 1, n_batches_train, endpoint=False), epoch_losses_train[i_epoch], '.', c='black')
    plt.figure(2)
    plt.plot(i_epoch+np.linspace(0, 1, n_batches_test, endpoint=False), epoch_losses_test[i_epoch], '.', c='black')
    plt.figure(3)
    plt.plot(i_epoch+np.linspace(0, 1, n_batches_test, endpoint=False), epoch_metrics_test[i_epoch]['rel L1'], '.', c='black')
  plt.figure(1)
  plt.xlim(0, n_epochs)
  plt.yscale('log')
  plt.xlabel('epoch')
  plt.ylabel('rel MSE train')
  plt.tight_layout()
  plt.figure(2)
  plt.xlim(0, n_epochs)
  plt.yscale('log')
  plt.xlabel('epoch')
  plt.ylabel('L1 test')
  plt.tight_layout()
  plt.figure(3)
  plt.xlim(0, n_epochs)
  plt.yscale('log')
  plt.xlabel('epoch')
  plt.ylabel('rel L1 test')
  plt.tight_layout()
  plt.figure(4)
  plt.stairs(lr, color='black')
  plt.yscale('log')
  plt.xlabel('epoch')
  plt.ylabel('lr')
  plt.tight_layout()
  plt.show()
