#!/bin/bash
for ((slice =0; slice <= $1; slice++))
do
echo "$slice"
python im2proba.py ${slice} $2 &
done
wait
exit 0
