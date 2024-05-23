#!/bin/bash

# 定义benchmarkmodel的选项
models=("as")

# 定义benchmarkdataset的选项
datasets=("a_inject_03" "w_inject_03")

# 循环遍历所有模型和数据集组合
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        python main.py --benchmarkmodel $model --benchmarkdataset $dataset --benchmark 1
    done
done