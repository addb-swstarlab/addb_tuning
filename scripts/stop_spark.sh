#!/bin/bash
user=user_name

cd /home/${user}
source /home/${user}/.bashrc

cd addb-spark

addb_spark -stop
