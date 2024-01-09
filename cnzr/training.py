import torch
from torch import optim
import torch.utils.data
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Union
from os import PathLike
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import logging
from cnzr.util import seed_everything
from cnzr.data import get_cancer_type_by_id

class FCNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int = 128, num_hidden: int = 3, dropout: float = 0.5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            ) for _ in range(num_hidden)],
            nn.Dropout1d(dropout),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, x):
        x = self.network(x)
        x = torch.softmax(x, dim=1)
        return x


def calculate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> (float, np.ndarray):
    model.eval()
    inds = np.arange(len(x))
    correct_predictions = 0
    correct_predictions_per_type = torch.zeros(y.shape[1])
    with torch.no_grad():
        for b_begin in range(0, len(x), batch_size):
            b_end = min(b_begin + batch_size, len(x))
            b_inds = inds[b_begin:b_end]
            prob = model(x[b_inds])
            pred = torch.argmax(prob, dim=1)
            correct_predictions += torch.sum(pred == torch.argmax(y[b_inds], dim=1)).item()
            for i in range(y.shape[1]):
                correct_predictions_per_type[i] += torch.sum(pred[y[b_inds][:, i] == 1] == torch.argmax(y[b_inds][y[b_inds][:, i] == 1], dim=1)).item()
    return correct_predictions / len(x), correct_predictions_per_type / torch.sum(y, dim=0)


def calculate_confusion_matrix(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    inds = np.arange(len(x))
    confusion_matrix = np.zeros((y.shape[1], y.shape[1]))
    with torch.no_grad():
        for b_begin in range(0, len(x), batch_size):
            b_end = min(b_begin + batch_size, len(x))
            b_inds = inds[b_begin:b_end]
            prob = model(x[b_inds])
            pred = torch.argmax(prob, dim=1)
            for i in range(y.shape[1]):
                for j in range(y.shape[1]):
                    confusion_matrix[i, j] += torch.sum((pred == j) & (y[b_inds][:, i] == 1)).item()
    return confusion_matrix


def train_model(x: np.ndarray, y: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray,
                model: nn.Module,
                sample_loss_weights: np.ndarray,
                epochs: int,
                batch_size: int,
                learning_rate: float,
                seed: int,
                log_dir: Union[str, PathLike]) -> nn.Module:
    log_dir = log_dir if isinstance(log_dir, Path) else Path(log_dir)
    writer = SummaryWriter(log_dir)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    sample_loss_weights = torch.tensor(sample_loss_weights, dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    global_step = 0
    for _ in tqdm(range(epochs), desc="Train epochs"):
        inds = np.arange(len(x))
        np.random.shuffle(inds)

        for b_begin in range(0, len(x), batch_size):
            b_end = min(b_begin + batch_size, len(x))
            b_inds = inds[b_begin:b_end]
            global_step += b_end-b_begin

            prob = model(x[b_inds])
            loss = loss_fn(prob, y[b_inds])
            loss = torch.sum(loss * sample_loss_weights[torch.argmax(y[b_inds], dim=1)]) / len(b_inds)

            with torch.no_grad():
                pred = torch.argmax(prob, dim=1)
                acc = torch.sum(pred == torch.argmax(y[b_inds], dim=1)) / len(b_inds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("charts/train_loss", loss.item(), global_step)
            writer.add_scalar("charts/train_acc", acc.item(), global_step)

    logging.info("Model has been Trained 💪")
    torch.save(model.state_dict(), log_dir / "model.pt")
    torch.save(optimizer.state_dict(), log_dir / "optimizer.pt")

    final_train_acc, _ = calculate_accuracy(model, x, y, batch_size)
    test_acc, test_type_acc = calculate_accuracy(model, x_test, y_test, batch_size)
    writer.add_scalar("charts/final_train_acc", final_train_acc, global_step)
    writer.add_scalar("charts/test_acc", test_acc, 0)
    for i, acc in enumerate(test_type_acc):
        writer.add_scalar(f"charts/test_acc_{get_cancer_type_by_id(i)}", acc, 0)

    test_confusion_matrix = calculate_confusion_matrix(model, x_test, y_test, batch_size)
    for i in range(test_confusion_matrix.shape[0]):
        for j in range(test_confusion_matrix.shape[1]):
            writer.add_scalar(f"confusion_matrix/{get_cancer_type_by_id(i)}/{get_cancer_type_by_id(j)}",
                              test_confusion_matrix[i, j], 0)

    writer.close()
    return model, test_acc