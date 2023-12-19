#!/bin/bash
user=user_name
pw=password
IPLIST[0]=slave1_IP
IPLIST[1]=slave2_IP
IPLIST[2]=slave3_IP

timeout_duration=5000

cd /home/${user}
source /home/${user}/.bashrc

date

timeout $timeout_duration beeline -u jdbc:hive2://cluster01:10000/tpch100g -n addb -f /home/${user}/addb-spark/requested_query.sql

if [ $? -eq 124 ]; then
    echo "timeout.. Stop the process as the beeline is running too long..."
    for (( i=0; i<${#IPLIST[@]}; i++ ))
    do # Check the spark processes are alive
        process_count=$(sshpass -p ${pw} ssh ${user}@${IPLIST[$i]} "source /home/${user}/.bashrc && jps" | wc -l)
        if [ "$process_count" -lt 6 ]; then
            echo "cluster $((i+1)) is disconnected.. stop the process.."
            exit 1
        fi
    done

else
    echo "The beeline has terminated successfully.."
    for (( i=0; i<${#IPLIST[@]}; i++ ))
    do # Check the spark processes are alive
        process_count=$(jps | wc -l)
        if [ "$process_count" -lt 6 ]; then
            echo "cluster $((i+1)) is disconnected.. stop the process.."
            exit 1
        fi
    done
fi
