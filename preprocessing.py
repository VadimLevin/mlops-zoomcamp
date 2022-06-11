__all__ = ["preprocess"]

import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from omegaconf import DictConfig

CATEGORICAL_FEATURES = ("PU_DO", )
NUMERICAL_FEATURES = ("trip_distance", )


def _filter_data(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger()

    outliers_filter_mask = (df.duration >= cfg.valid_duration_in_minutes.min) & \
        (df.duration <= cfg.valid_duration_in_minutes.max)
    df = df[outliers_filter_mask]
    logger.info("Data frame records after duration filtering: %d", df.shape[0])
    if cfg.filter_nans:
        logger.info("Filtering NaNs...")
        null_mask = ~df[[cfg.pickup_location_key, cfg.drop_off_location_key]].isnull()
        df = df[null_mask]
        logger.info("Data frame records after NaNs filtering: %d", df.shape[0])
    return df


def _prepare_categorical_features(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    if "PU_DO" in cfg.categorical_features:
        df["PU_DO"] = df[cfg.pickup_location_key].astype(str) + "_" \
            + df[cfg.drop_off_location_key].astype(str)
    return df


def preprocess(cfg: DictConfig, df: pd.DataFrame, dv: DictVectorizer,
               fit_dv: bool = False) -> Tuple[dict, np.ndarray]:
    logger = logging.getLogger()

    logger.info("Data frame records: %d", df.shape[0])
    df['duration'] = df[cfg.drop_off_datetime_key] - df[cfg.pickup_datetime_key]
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = _filter_data(cfg, df)

    y = df.duration.to_numpy()

    df = _prepare_categorical_features(cfg, df)

    features_keys = cfg.categorical_features + cfg.numerical_features
    features_dict = df[features_keys].to_dict(orient='records')
    logger.info("Applying one-hot encoding for categorical features: %s ...",
                CATEGORICAL_FEATURES)
    if fit_dv:
        X = dv.fit_transform(features_dict)
    else:
        X = dv.transform(features_dict)

    logger.info("Preprocessing done")
    return X, y
