# DistTraining
#On Gcloud deep learning vm -
#sudo apt update
#sudo apt-get install git
#which git
#pip3 install pytorch-ignite

#Reference : 
============
https://pytorch-ignite.ai/tutorials/intermediate/01-cifar10-distributed/

# Download CIFAR datasets to your local:
==========================================
# python -c "from torchvision.datasets import CIFAR10; CIFAR10('cifar10', download=True)"   

#git repo - https://github.com/hikrishn/DistTraining 
==================================================
Command history on my GPU
==========================
gcloud compute scp --project krish-gcp-iris --zone asia-east1-c --recurse /Users/krishna/Creative_Man/AI_ML/JobPrep/DistributedTrainingCheckout/DistTraining krish-pig-1-vm:~/
gcloud compute scp --project krish-gcp-iris --zone asia-east1-a --recurse /Users/krishna/Creative_Man/AI_ML/JobPrep/DistributedTrainingCheckout/DistTraining deeplearning-1-vm:~/
cd DistTraining/
pip3 install fire
pip3 install pytorch-ignite
# Reference for fixing the torch xla error : https://medium.com/the-owl/installing-pytorch-xla-in-google-colab-without-35d37d1d03c6
pip3 install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
pip3 install tensorboardX
# Run with torch.distributed.launch (Recommended) on a single node with 1 GPU
python -u -m torch.distributed.launch --nproc_per_node=1 --use_env disttrain.py run --backend="nccl"

#Run with internal spawining (torch.multiprocessing.spawn)
python -u disttrain.py run --backend="nccl" --nproc_per_node=2

python -u -m torch.distributed.launch --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=34.80.114.167 --master_port=2222 --use_env disttrain.py run --backend="nccl"
python -u -m torch.distributed.launch --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=34.80.114.167 --master_port=2222 --use_env disttrain.py run --backend="nccl"
docker pull 
docker run -it --gpus all --rm -v $(pwd):/mnt --network=host pytorchignite/base:latest