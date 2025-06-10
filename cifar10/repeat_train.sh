#!/bin/bash
train_times=5
for ((i=1; i<=${train_times}; i++)); do
    sh train_cifar_notadv_dropout.sh
    sh train_cifar_notadv_fixnoise_0.1.sh
done