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
