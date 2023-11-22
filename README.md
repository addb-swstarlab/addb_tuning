# addb_tuning

ADDB tuning server for online tuning (ver 2023)

* Not finished version.. Under development..

Before start tuning, execute ```copybash.sh```. 
Please define ```${user}``` in the below bash file with your user name.
```
./scripts/copybash.sh
```

## Sample Collection
Collecting samples for historical data..
```--query``` indicates tpch query index to benchmark and  ```--size``` means the number of samples to collect..
```
python collect_samples.py --query 1 --size 10
```
The results are saved on ```/observation``` named ```history_q01_20231122-00.csv```.
