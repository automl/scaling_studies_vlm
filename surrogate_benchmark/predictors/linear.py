import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from surrogate_benchmark.data import TorchSurrogateDataset
from surrogate_benchmark.metrics import compute_regression_metrics


def train_test_loaders():
    TRAIN_CSV_PATH = "../splits/train.csv"
    TEST_CSV_PATH = "../splits/test.csv"
    TARGET_COL = "val_loss"

    train_dataset = TorchSurrogateDataset(TRAIN_CSV_PATH, target_col=TARGET_COL)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

    test_dataset = TorchSurrogateDataset(TEST_CSV_PATH, target_col=TARGET_COL)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, num_workers=0)
    return train_loader, test_loader


def train(
        model: torch.nn.Module,
        device: str,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Adam,
        epoch: int,
        log_interval: int = 10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        optimizer.zero_grad()
        output = model(data)
        target = torch.squeeze(target)
        output = torch.squeeze(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


class Net(nn.Module):
    def __init__(self, layer_size: int) -> None:
        super(Net, self).__init__()

        nfeat = 28

        self.fc1 = nn.Linear(nfeat, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size)
        self.fc5 = nn.Linear(layer_size, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out

    @torch.no_grad()
    def validate(self, loader, device):
        self.eval()
        y_true, y_pred = [], []

        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()

            preds = self(x).squeeze(-1)
            y_true.append(y.squeeze(-1).cpu().numpy())
            y_pred.append(preds.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        return y_true, y_pred


if __name__ == "__main__":
    train_loader, test_loader = train_test_loaders()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_size = 128
    net = Net(layer_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(1, 120):
        train(net, device, train_loader, optimizer, epoch, log_interval=10)

    y_true, y_pred = net.validate(test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred)
    print(metrics)
