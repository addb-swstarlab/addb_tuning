#!/bin/bash

cd /home/user
source /home/user/.bashrc

cd addb-spark

addb_spark -start_conf '/home/user/addb-spark/test_spark.conf'

sleep 30
