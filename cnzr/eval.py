from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Union
from os import PathLike
from omegaconf import OmegaConf
import seaborn as sns
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class TensorboardData:
    train_loss: pd.DataFrame
    train_acc: pd.DataFrame
    test_acc: pd.DataFrame
    final_train_acc: pd.DataFrame


def load_tb_file(tb_file: Path) -> TensorboardData:
    acc = EventAccumulator(str(tb_file))
    # load the data, because it is not automatically loaded when constructing the accumulator
    acc.Reload()

    train_loss = acc.Scalars("charts/train_loss")
    train_loss = pd.DataFrame.from_dict({
        "step": [r.step for r in train_loss],
        "loss": [r.value for r in train_loss]
    })

    train_acc = acc.Scalars("charts/train_acc")
    train_acc = pd.DataFrame.from_dict({
        "step": [l.step for l in train_acc],
        "acc": [l.value for l in train_acc]
    })

    test_acc = acc.Scalars("charts/test_acc")
    test_acc = pd.DataFrame.from_dict({
        "step": [l.step for l in test_acc],
        "acc": [l.value for l in test_acc]
    })

    final_train_acc = acc.Scalars("charts/final_train_acc")
    final_train_acc = pd.DataFrame.from_dict({
        "step": [l.step for l in final_train_acc],
        "acc": [l.value for l in final_train_acc]
    })

    return TensorboardData(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc,
                           final_train_acc=final_train_acc)


def is_log_dir(d: Path) -> bool:
    return d.is_dir and (d / "model.pt").is_file() and (d / "optimizer.pt").is_file()


def calculate_windowed_mean(window_size: int,
                            data: list[pd.DataFrame]):
    combined_data = pd.DataFrame({"step": [], "value": []})
    max_steps = max([d["step"].max() for d in data])
    group_bounds = range(0, max_steps, window_size)
    for d in data:
        groups = d.groupby(pd.cut(d["step"], group_bounds), observed=True)
        mean_group_data = groups.mean()
        mean_group_data["step"] = [int(s.mid) for s in mean_group_data.index]
        combined_data = pd.concat([combined_data, mean_group_data], ignore_index=True)
    return combined_data


class ExperimentResults:
    def __init__(self, log_dir: Union[str, PathLike]):
        self.root_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.log_dirs = [d for d in self.root_dir.iterdir() if is_log_dir(d)]
        assert len(self.log_dirs) > 0
        print(f"Found {len(self.log_dirs)} log directories.")
        for d in self.log_dirs:
            print(f" - {d}")

        self.model_files = [d / "model.pt" for d in self.log_dirs]
        assert all([f.is_file() for f in self.model_files])
        self.optimizer_files = [d / "optimizer.pt" for d in self.log_dirs]
        assert all([f.is_file() for f in self.optimizer_files])
        self.tb_files = [next(d.rglob("events.out.tfevents*")) for d in self.log_dirs]
        assert all([f.is_file() for f in self.tb_files])
        self.hydra_config_files = [d / ".hydra" / "config.yaml" for d in self.log_dirs]
        assert all([f.is_file() for f in self.hydra_config_files])
        self.hydra_configs = [OmegaConf.load(f) for f in self.hydra_config_files]
        self.tb_data = [load_tb_file(f) for f in tqdm(self.tb_files, desc="Loading tensorboard data")]

        self.sns_theme = "whitegrid"

    def _plot(self, data: pd.DataFrame, data_key_x: str, data_key_y: str,
              x_label: str, y_label: str,
              figsize: tuple[int, int],
              title: str,
              dst_file: Union[str, PathLike],
              x_lim: Optional[tuple[float, float]] = None, y_lim: Optional[tuple[float, float]] = None,
              sns_plot_fn: callable = sns.lineplot):
        sns.set_theme(style=self.sns_theme)
        _, ax = plt.subplots(figsize=figsize)
        sns_plot_fn(data=data, x=data_key_x, y=data_key_y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if x_lim is None:
            x_lim = (data[data_key_x].min(), data[data_key_x].max())
        if y_lim is None:
            y_lim = (data[data_key_y].min(), data[data_key_y].max())
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

        plt.savefig(dst_file)
        plt.close()

    def plot_train_loss(self, dst_file: Union[str, PathLike],
                        window_size: int = 50,
                        figsize: tuple[int, int] = (16, 9),
                        x_label: str = "Step",
                        y_label: str = "Train Loss",
                        title: str = "Train Loss",
                        x_lim: Optional[tuple[float, float]] = None,
                        y_lim: Optional[tuple[float, float]] = None):

        windowed_mean = calculate_windowed_mean(window_size, [d.train_loss for d in self.tb_data])
        self._plot(windowed_mean, "step", "loss",
                   x_label, y_label, figsize, title, dst_file, x_lim, y_lim)

    def plot_train_acc(self, dst_file: Union[str, PathLike],
                       window_size: int = 50,
                       figsize: tuple[int, int] = (16, 9),
                       x_label: str = "Step",
                       y_label: str = "Train Accuracy",
                       title: str = "Train Accuracy",
                       x_lim: Optional[tuple[float, float]] = None,
                       y_lim: Optional[tuple[float, float]] = None):

        windowed_mean = calculate_windowed_mean(window_size, [d.train_acc for d in self.tb_data])
        self._plot(windowed_mean, "step", "acc",
                   x_label, y_label, figsize, title, dst_file, x_lim, y_lim)

    def plot_test_acc(self, dst_file: Union[str, PathLike],
                      x_label: str = "Test Accuracy",
                      title: str = "Test Accuracy",
                      figsize: tuple[int, int] = (16, 9)):

        test_data = pd.concat([d.test_acc for d in self.tb_data], ignore_index=True)["acc"]
        final_train_data = pd.concat([d.final_train_acc for d in self.tb_data], ignore_index=True)["acc"]
        data = pd.DataFrame({"acc": [], "type": []})
        data = pd.concat([data, pd.DataFrame({"acc": test_data, "type": "test"})], ignore_index=True)
        data = pd.concat([data, pd.DataFrame({"acc": final_train_data, "type": "final_train"})], ignore_index=True)

        sns.set_theme(style=self.sns_theme)
        _, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(data=data, x="acc", hue="type", ax=ax)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        plt.savefig(dst_file)
        plt.close()

    def plot_all(self, dst_dir: Union[str, PathLike],
                 window_size: int = 50,
                 figsize: tuple[int, int] = (16, 9),
                 file_prefix: str = ""):

        dst_dir = Path(dst_dir) if isinstance(dst_dir, str) else dst_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        pb = tqdm(total=3, desc="Plotting")
        self.plot_train_loss(dst_dir / f"{file_prefix}_train_loss.png",
                             window_size=window_size,
                             figsize=figsize)
        pb.update()
        self.plot_train_acc(dst_dir / f"{file_prefix}_train_acc.png",
                            window_size=window_size,
                            figsize=figsize)
        pb.update()
        self.plot_test_acc(dst_dir / f"{file_prefix}_test_acc.png",
                           figsize=figsize)
        pb.update()
        pb.close()