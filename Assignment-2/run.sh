#!/bin/sh
if [ $1 -eq 1 ]
then
    python3 Q1.py $2 $3 $4
elif [ $1 -eq 2 ]
then
    python3 Q2.py $2 $3 $4 $5
fi
