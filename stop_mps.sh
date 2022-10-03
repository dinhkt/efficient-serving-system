#!/bin/bash
echo quit | sudo nvidia-cuda-mps-control
LIST_GPUs=$(nvidia-smi -L)
nGPU=$(echo -n $LIST_GPUs | grep -Fo "UUID" | wc -l)
echo "Stopped MPS server"
for ((i=0; i<nGPU;i++))
do
        sudo nvidia-smi -i $i -c DEFAULT
done