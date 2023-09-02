#!/bin/sh

#SBATCH --job-name="pspnet"
#SBATCH --output=%j.out
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=1150G
#SBATCH --gpus=rtx_3090:1

PYTHON=/home/quanta/.conda/envs/semseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

dataset=$1
exp_name=$2
exp_dir=/cluster/scratch/guanji/Experiments/PSPNet/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train.sh tool/train.py tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
# copy files
$PYTHON -u ${exp_dir}/train.py --config=${config} 2>&1 | tee ${model_dir}/train-$now.log
# training
$PYTHON -u ${exp_dir}/train.py --config=${config} 2>&1 | tee ${model_dir}/train-$now.log
