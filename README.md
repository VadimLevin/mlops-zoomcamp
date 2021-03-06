# mlops-zoomcamp
MLOps course https://github.com/DataTalksClub/mlops-zoomcamp


## Installing required packages

```shell
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Fetching data

```shell
python fetch_file.py "https://nyc-tlc.s3.amazonaws.com/trip+data/green_tripdata_2021-01.parquet" \
    --out datasets
python fetch_file.py "https://nyc-tlc.s3.amazonaws.com/trip+data/green_tripdata_2021-02.parquet" \
    --out datasets
python fetch_file.py "https://nyc-tlc.s3.amazonaws.com/trip+data/green_tripdata_2021-03.parquet" \
    --out datasets
```

## Launching experiment tracking server

```shell
mlflow ui --backend-store-uri sqlite:///experiment-tracking/mlflow.db --default-artifact-root experiment-tracking/artifacts
```

## Preprocessing

```shell
python preprocess_data.py --raw_data_path datasets --dest_path datasets/preprocessed
```

## Running training

```shell
python train.py --data_path datasets/preprocessed
```


To run with prefect and hydra had to hack prefect code to allow arbitrary parameters. In `site-packages/prefect/flows.py` move `self.parameters = parameter_schema(self.fn)` under the `if self.should_validate_parameters`.
