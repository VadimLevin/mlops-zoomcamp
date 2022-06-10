import logging
from itertools import filterfalse

from mlflow.entities import Run as MlflowRun
from mlflow.tracking import MlflowClient


def print_run_info(run: MlflowRun):
    def print_as_bullet_list(kv_iterable):
        for key, value in kv_iterable:
            logger.info("     - %-10s: %s", key, value)
    logger = logging.getLogger()

    logger.info("=== Run info ===")
    logger.info("id: %s", run.info.run_id)
    logger.info("artifacts:")
    for artifact in MlflowClient().list_artifacts(run.info.run_id):
        logger.info("    - %s, size: %d", artifact.path, artifact.file_size)
    logger.info("parameters:")
    print_as_bullet_list(run.data.params.items())
    logger.info("metrics:")
    print_as_bullet_list(run.data.metrics.items())
    logger.info("run_tags:")
    print_as_bullet_list(
        filterfalse(lambda kv: kv[0].startswith("mlflow."),
                    run.data.tags.items())
    )
