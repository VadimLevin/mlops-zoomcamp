import argparse
import os
import pickle
from pathlib import Path
from itertools import filterfalse

import mlflow
from mlflow.entities import Run as MlflowRun
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-default")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def print_run_info(run: MlflowRun):
    def print_as_bullet_list(kv_iterable):
        for key, value in kv_iterable:
            print("     -", key, ":", value)

    print("=== Run info ===")
    print("id:", run.info.run_id)
    print("artifacts:")
    for artifact in MlflowClient().list_artifacts(run.info.run_id, "model"):
        print("    -", artifact.path, "size:", artifact.file_size)
    print("parameters:")
    print_as_bullet_list(run.data.params.items())
    print("metrics:")
    print_as_bullet_list(run.data.metrics.items())
    print("run_tags:")
    print_as_bullet_list(
        filterfalse(lambda kv: kv[0].startswith("mlflow."),
                    run.data.tags.items())
    )


def run(data_path: Path):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    mlflow.autolog()
    with mlflow.start_run() as run:
        print("Active run id:", run.info.run_id)
        print("Training")
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        print("Validation")
        y_pred = rf.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("validation_rmse", rmse)
        print_run_info(mlflow.get_run(run.info.run_id))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        help="The location where the processed NYC taxi trip data was saved.")
    args = parser.parse_args()

    run(args.data_path)
