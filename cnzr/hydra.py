from omegaconf import DictConfig
from cnzr.data import load_data, one_hot_encode
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from cnzr.training import train_model, FCNN
from pathlib import Path
import hydra
import logging


def run_experiment(cfg: DictConfig):
    try:
        log_dir = Path(hydra.utils.HydraConfig.get().runtime.output_dir)
        seed = cfg.seed
        x, y = load_data("../data")

        cancer_type, cancer_count = np.unique(y, return_counts=True)
        sorted_inds = np.argsort(cancer_type)
        cancer_count = cancer_count[sorted_inds]
        cancer_type = cancer_type[sorted_inds]
        cancer_loss_weights = 1 / cancer_count
        cancer_loss_weights = cancer_loss_weights / np.sum(cancer_loss_weights)
        logging.info(f"Cancer types: {cancer_type}")
        logging.info(f"Cancer counts: {cancer_count}")
        if not cfg.use_loss_weights:
            cancer_loss_weights = np.ones_like(cancer_loss_weights)
        logging.info(f"Cancer loss weights: {cancer_loss_weights}")

        y = one_hot_encode(y, n_classes=3)

        np.random.seed(seed)
        inds = np.arange(x.shape[0])
        np.random.shuffle(inds)

        d_len = x.shape[0]
        split_points = [
            int((1 - (cfg.test_size + cfg.val_size)) * d_len),
            int((1 - cfg.test_size) * d_len)
        ]
        x_train, x_val, x_test = np.split(x[inds], split_points)
        y_train, y_val, y_test = np.split(y[inds], split_points)

        logging.info(f"Train: {x_train.shape}, {y_train.shape}")
        logging.info(f"Val: {x_val.shape}, {y_val.shape}")
        logging.info(f"Test: {x_test.shape}, {y_test.shape}")

    except Exception as e:
        logging.error(e)
        raise e

    if cfg.algo.name == "fcnn":
        try:
            model = FCNN(in_features=x.shape[1], out_features=y.shape[1], hidden_size=cfg.algo.hidden_size,
                         num_hidden=cfg.algo.num_hidden, dropout=cfg.algo.dropout)

            val_or_test_x, val_or_test_y = (x_val, y_val) if cfg.use_val else (x_test, y_test)
            model, acc = train_model(x_train, y_train, val_or_test_x, val_or_test_y, model,
                                     cancer_loss_weights,
                                     cfg.algo.epochs, cfg.algo.batch_size,
                                     cfg.algo.learning_rate, seed, log_dir)
            logging.info(f"Accuracy: {acc}")
        except Exception as e:
            logging.error(e)
            raise e
        return -acc
    else:
        raise ValueError(f"Algorithm {cfg.algo.name} not supported")
