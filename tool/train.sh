#!/bin/sh

#SBATCH --job-name="pspnet"
#SBATCH --output=%j.out
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=48G
#SBATCH --tmp=1150G
#SBATCH --gpus=rtx_2080_ti:1

module load gcc/8.2.0 python_gpu/3.8.5 cuda/10.1.243 cudnn/8.0.5 git-lfs/2.3.0 git/2.31.1 eth_proxy

nvidia-smi

PYTHON=/cluster/scratch/guanji/.python_venv/semseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

dataset=scannet
exp_name=pspnet50
exp_dir=/cluster/scratch/guanji/Experiments/PSPNet/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=/cluster/home/guanji/Projects/semseg/config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train.sh tool/train.py tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=/cluster/home/guanji/Projects/semseg
# copy files
# $PYTHON scannet_copy_data.py
# training
$PYTHON ${exp_dir}/train.py --config=${config}
