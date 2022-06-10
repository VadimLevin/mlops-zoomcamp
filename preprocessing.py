__all__ = ["preprocess"]

import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from omegaconf import DictConfig

LOCATIONS = ["PULocationID", "DOLocationID"]

CATEGORICAL_FEATURES = ("PU_DO", )
NUMERICAL_FEATURES = ("trip_distance", )


def _filter_data(df: pd.DataFrame, valid_duration_in_minutes,
                 filter_nans: bool) -> pd.DataFrame:
    logger = logging.getLogger()

    outliers_filter_mask = (df.duration >= valid_duration_in_minutes.min) & \
        (df.duration <= valid_duration_in_minutes.max)
    df = df[outliers_filter_mask]
    logger.info("Data frame records after duration filtering: %d", df.shape[0])
    if filter_nans:
        logger.info("Filtering NaNs...")
        null_mask = ~df[LOCATIONS].isnull()
        df = df[null_mask]
        logger.info("Data frame records after NaNs filtering: %d", df.shape[0])
    return df


def _prepare_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["PU_DO"] = df[LOCATIONS[0]].astype(str) + '_' + df[LOCATIONS[1]].astype(str)
    return df



def preprocess(df: pd.DataFrame, dv: DictVectorizer, cfg: DictConfig,
               fit_dv: bool = False) -> Tuple[dict, np.ndarray]:
    logger = logging.getLogger()

    logger.info("Data frame records: %d", df.shape[0])
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = _filter_data(df, cfg.valid_duration_in_minutes, cfg.filter_nans)

    y = df.duration.to_numpy()

    df = _prepare_categorical_features(df)

    features_dict = df[list(CATEGORICAL_FEATURES + NUMERICAL_FEATURES)].to_dict(orient='records')
    logger.info("Applying one-hot encoding for categorical features: %s ...",
                CATEGORICAL_FEATURES)
    if fit_dv:
        X = dv.fit_transform(features_dict)
    else:
        X = dv.transform(features_dict)

    logger.info("Preprocessing done")
    return X, y
