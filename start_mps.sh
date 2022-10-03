#!/bin/bash
LIST_GPUs=$(nvidia-smi -L)
nGPU=$(echo -n $LIST_GPUs | grep -Fo "UUID" | wc -l)
echo "Enable MPS server for $nGPU GPU"
for ((i=0; i<nGPU;i++))
do
        sudo nvidia-smi -i $i -c EXCLUSIVE_PROCESS
done
sudo nvidia-cuda-mps-control -d