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

## Start
### Run main.py. Parser explanation as below,
```
type=int
--query ,  -q : Define a number of tpc-h query to test
--trails,  -t : Define a number of iteration to tune

type=str
--sqfile,  -s : Provide a required sql file path to be tuned

type=action(store_true)
--tuning      : If you want to skip collecting observations(historical data) and to do actual tuning, trigger this
```
### Collecting historical data by running bayesian optimization..
- If you want to run one of tpc-h query
- ```{num}``` is in ```{1,...,22}```
- ```{trial}``` is the number of iteration to collect history. It should be 2 or greater
```
python main.py -q {num} -t {trial}
```
