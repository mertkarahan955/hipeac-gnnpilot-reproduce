#!/bin/bash
# 列举当前目录下的所有文件夹

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
    echo "[Usage]: run.sh {matrix_dir} {csvname} {full_csvname | optional}"
    exit 1
fi

matrix_dir=$1
csv_name=$2
full_name=$3

# dataset_list=("ogbn-arxiv.pt" "ogbl-collab.pt" "ogbn-mag.pt" "ogbl-ppa.pt" "ogbn-products.pt" "reddit.pt" "ogbn-proteins.pt")
dataset_list=("ogbn-arxiv.pt" "ogbn-mag.pt")

for data in "${dataset_list[@]}"; do    
    dir="$matrix_dir/$data"
    echo $dir
    python e2e_gat.py $dir $csv_name $full_name
done