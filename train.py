import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import mlflow

from prepare_data import preprocess_data
from pickle_utils import load_pickle
import type_utils as tu
from mlflow_utils import print_run_info


def load_data(cfg: DictConfig) -> Dict[str, tu.Dataset]:
    logger = logging.getLogger()

    datasets_root_path = Path(cfg.data.paths.preprocessed)
    logger.info("Trying to load preprocessed data from: %s", datasets_root_path)

    datasets_paths = tuple(datasets_root_path.glob("*_dataset.pkl"))
    if len(datasets_paths) == 0:
        logger.info("Preprocessed data not found. Running preprocessing...")
        preprocess_data(cfg.data)
    datasets_paths = tuple(datasets_root_path.glob("*_dataset.pkl"))
    assert len(datasets_paths) > 0, \
        f"Can't load datasets from {datasets_root_path}"

    logger.info("Found %d datasets", len(datasets_paths))

    name_to_dataset = dict()  # type: Dict[str, tu.Dataset]
    for dataset_path in datasets_paths:
        dataset_name = dataset_path.stem.split("_dataset")[0]
        logger.info("Loading %s...", dataset_name)

        dataset = load_pickle(dataset_path)
        assert isinstance(dataset, tu.Dataset), \
            f"Dataset {dataset_name} has wrong type: {type(dataset)}"

        logger.info("Dataset features shape: %s. Type: %s",
                    dataset.features.shape, type(dataset.features))
        name_to_dataset[dataset_name] = dataset

    return name_to_dataset



def eval_model_metrics(model: tu.ModelType, metrics: Dict[str, tu.MetricType],
                       dataset_name: str, dataset: tu.Dataset) -> tu.DatasetMetrics:
    predicted_values = model.predict(dataset.features)
    ds_metrics = {
        name: metric(dataset.values, predicted_values)
        for name, metric in metrics.items()
    }
    return tu.DatasetMetrics(dataset_name, predicted_values, ds_metrics)


def train_and_eval(cfg: DictConfig, train_ds: tu.Dataset,
                   evaluation_datasets: Dict[str, tu.Dataset]) \
        -> Tuple[tu.ModelType, Sequence[tu.DatasetMetrics]]:
    logger = logging.getLogger()

    metrics = {
        name: instantiate(metric)
        for name, metric in cfg.metrics.items()
    }
    logger.debug("Instantiated metrics: %s", metrics)

    logger.debug("Instantiating regression model %s", cfg.model)
    model = instantiate(cfg.model)
    logger.info("Training...")
    model.fit(train_ds.features, train_ds.values)

    logger.info("Evaluating training quality...")
    evaluated_metrics = [
        eval_model_metrics(model, metrics, "train", train_ds),
    ]
    for ds_name, dataset in evaluation_datasets.items():
        evaluated_metrics.append(
            eval_model_metrics(model, metrics, ds_name, dataset)
        )
    return model, evaluated_metrics


def train(cfg: DictConfig) -> Tuple[tu.ModelType, tu.DatasetMetrics]:
    logger = logging.getLogger()
    datasets = load_data(cfg)
    train_dataset = datasets.pop("train")
    model, metrics = train_and_eval(cfg, train_dataset, datasets)
    for dataset_metrics in metrics:
        logger.info("--- %s metrics ---", dataset_metrics.dataset_name)
        for metric_name, metric_value in dataset_metrics.metrics.items():
            logger.info("|  %-10s : %-10.6f |", metric_name, metric_value)
    return model, metrics


def train_with_tracking(cfg: DictConfig) -> Tuple[tu.ModelType, tu.DatasetMetrics]:
    logger = logging.getLogger()
    try:
        mlflow.set_tracking_uri(cfg.experiment_tracking.tracking_url)
        mlflow.set_experiment(cfg.experiment_tracking.experiment_name)
    except Exception as e:
        logger.exception(
            "Failed to establish connection to tracking server %s. Error: %s."
            " Ensure tracking server is available or launch local one. Example:\n"
            "mlflow ui --backend-store-uri sqlite:///experiment-tracking/mlflow.db "
            "--default-artifact-root experiment-tracking/artifacts",
            cfg.experiment_tracking.tracking_url, e
        )
        return -1
    if cfg.experiment_tracking.enable_autolog:
        mlflow.autolog()

    hydra_config = HydraConfig.get()
    with mlflow.start_run() as run:
        logger.info("Starting run id: %s", run.info.run_id)
        mlflow.log_params(cfg.model)
        # Log config files
        mlflow.log_artifacts(Path(hydra_config.runtime.output_dir,
                                  hydra_config.output_subdir))
        model, metrics = train(cfg)
        for dataset_metrics in metrics:
            for metric_name, metric_value in dataset_metrics.metrics.items():
                mlflow.log_metric(
                    f"{dataset_metrics.dataset_name}_{metric_name}", metric_value
                )
        print_run_info(mlflow.get_run(run.info.run_id))
        # Save training logs
        mlflow.log_artifact(HydraConfig.get().job_logging.handlers.file.filename)
    return model, metrics


@hydra.main(version_base=None, config_path="config",
            config_name="default_random_forest")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.experiment_tracking.enabled:
        _, _ = train_with_tracking(cfg)
    else:
        _, _ = train()


if __name__ == '__main__':
    main()
