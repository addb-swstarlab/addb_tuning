user=user_name

for i in {1..22}
do  
    cp sample.sh run_tpch_q$i.sh
    echo beeline -u jdbc:hive2://cluster01:10000/tpch100g -n addb -f /home/${user}/addb-spark/tpch_query/q$i.sql >> run_tpch_q$i.sh
done
