export CUDA_HOME=/usr/local/cuda-8.0
export NVIDIA_HOME=/usr/local/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$NVIDIA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin:/usr/local/nvidia/bin

