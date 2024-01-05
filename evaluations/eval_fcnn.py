from pathlib import Path
from cnzr.eval import ExperimentResults

if __name__ == '__main__':
    experiment_log_dir = Path("../experiments/logs/2023-12-22/13-24-18")
    plot_dir = Path("./plots/fcnn")
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = ExperimentResults(experiment_log_dir)
    results.plot_all(plot_dir, file_prefix="fcnn")