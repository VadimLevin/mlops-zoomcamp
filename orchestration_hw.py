from pathlib import Path
from typing import Dict, Sequence, Tuple
import shutil

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import prefect

from prepare_data import preprocess_data
from pickle_utils import load_pickle, save_pickle
import type_utils as tu


@prefect.task
def load_data(cfg: DictConfig) -> Dict[str, tu.Dataset]:
    logger = prefect.get_run_logger()

    datasets_root_path = Path(cfg.data.paths.preprocessed)
    logger.info("Trying to load preprocessed data from: %s", datasets_root_path)

    datasets_paths = tuple(datasets_root_path.glob("*_dataset.pkl"))
    if cfg.data.preprocessing.always or len(datasets_paths) == 0:
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


@prefect.task
def evaluate(cfg: DictConfig, model: tu.ModelType,
             datasets: Dict[str, tu.Dataset]) -> Sequence[tu.DatasetMetrics]:
    logger = prefect.get_run_logger()

    logger.info("Evaluating training quality...")
    metrics = {
        name: instantiate(metric)
        for name, metric in cfg.metrics.items()
    }
    logger.debug("Instantiated metrics: %s", metrics)
    return tuple(
        eval_model_metrics(model, metrics, ds_name, dataset)
        for ds_name, dataset in datasets.items()
    )


@prefect.task
def run_train(cfg: DictConfig, train_ds: tu.Dataset) -> tu.ModelType:
    logger = prefect.get_run_logger()

    logger.debug("Instantiating regression model %s", cfg.model)
    model = instantiate(cfg.model)
    logger.info("Training...")
    model.fit(train_ds.features, train_ds.values)

    logger.info("Evaluating training quality...")

    return model

@prefect.flow(validate_parameters=False)
def train(cfg: DictConfig) -> None:
    logger = prefect.get_run_logger()
    datasets = load_data(cfg).result()
    model = run_train(cfg, datasets["train"]).result()
    metrics = evaluate(cfg, model, datasets).result()
    for dataset_metrics in metrics:
        logger.info("--- %s metrics ---", dataset_metrics.dataset_name)
        for metric_name, metric_value in dataset_metrics.metrics.items():
            logger.info("|  %-10s : %-10.6f |", metric_name, metric_value)
    hydra_config = HydraConfig.get()
    save_pickle(model, Path(hydra_config.runtime.output_dir) / "model.pkl")
    shutil.copyfile(
        Path(cfg.data.paths.preprocessed, "feature_vectorizer.pkl"),
        Path(hydra_config.runtime.output_dir, "feature_vectorizer.pkl")
    )




@hydra.main(version_base=None, config_path="config",
            config_name="default_linear_regression_categorical_only")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == '__main__':
    main()
