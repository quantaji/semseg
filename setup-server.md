```sh
module load gcc/8.2.0 python_gpu/3.8.5 cuda/10.1.243 cudnn/8.0.5 git-lfs/2.3.0 git/2.31.1 eth_proxy
python -m venv ${SCRATCH}/.python_venv/semseg
${SCRATCH}/.python_venv/semseg/bin/pip3 install  -r requirements.txt --cache-dir ${SCRATCH}/pip_cache
```
