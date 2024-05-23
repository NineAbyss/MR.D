#!/bin/bash

models=("as")

datasets=("a_inject_03" "w_inject_03")


for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        python main.py --benchmarkmodel $model --benchmarkdataset $dataset --benchmark 1
    done
done