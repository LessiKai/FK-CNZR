import hydra
from omegaconf import DictConfig
from cnzr.hydra import run_experiment

@hydra.main(config_path="./configs", config_name="fcnn_hpo", version_base=None)
def main(cfg: DictConfig):
    return run_experiment(cfg)


if __name__ == '__main__':
    main()