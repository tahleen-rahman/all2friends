#!/bin/bash
for ((slice =0; slice <= $1; slice++))
do
echo "$slice"
python main_placenet.py ${slice}&
done
wait
exit 0
