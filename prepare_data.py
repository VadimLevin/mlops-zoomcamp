import os
import io
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from omegaconf import DictConfig, OmegaConf
import hydra

from fetch_file import fetch
from pickle_utils import save_pickle
from preprocessing import preprocess
import type_utils as tu


def load(url: str, cache_data_dir: str,
         fetch_force: bool = False) -> pd.DataFrame:
    data = fetch(url, output_dir=cache_data_dir, force=fetch_force)
    df = pd.read_parquet(io.BytesIO(data))
    return df


def load_dataset_data(url: str, cache_path: os.PathLike,
                      files: Union[Sequence[str], str],
                      fetch_force: bool = False) -> pd.DataFrame:
    if isinstance(files, str):
        return load(f"{url}/{files}", cache_path, fetch_force)
    else:
        return pd.concat(load(f"{url}/{file}", cache_path, fetch_force) for file in files)


def preprocess_dataset(cfg: DictConfig, dataset_tag: str,
                       dv: Optional[DictVectorizer] = None) -> DictVectorizer:
    logger = logging.getLogger()

    data = cfg.datasets[dataset_tag]
    logger.info("Loading %s. Containing %d entries: %s", dataset_tag,
                len(data), data)
    dataset_df = load_dataset_data(cfg.base_url, cfg.paths.raw_data, data,
                                   cfg.fetch_force)
    logger.info("Dataset shape before preprocessing: %s", dataset_df.shape)
    fit_dict_vectorizer = dv is None
    if dv is None:
        dv = DictVectorizer()

    X, y = preprocess(cfg.preprocessing, dataset_df, dv, fit_dict_vectorizer)

    if fit_dict_vectorizer:
        save_pickle(dv, Path(cfg.paths.preprocessed, f"feature_vectorizer.pkl"))

    save_pickle(tu.Dataset(X, y),
                Path(cfg.paths.preprocessed, f"{dataset_tag}_dataset.pkl"))

    return dv


@hydra.main(version_base=None, config_path="config/data",
            config_name="nyc_taxi_green")
def preprocess_data(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    dv = preprocess_dataset(cfg, "train")
    for dataset_tag in filter(lambda tag: tag != "train", cfg.datasets):
        preprocess_dataset(cfg, dataset_tag, dv)


if __name__ == '__main__':
    preprocess_data()
