#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=/home/sanuj/Projects/nuclei-net/caffe-scripts/nuclei/three_class/arch_4/solver.prototxt \
	2>&1 | tee /home/sanuj/Projects/nuclei-net/caffe-scripts/nuclei/three_class/arch_4/output_1.txt
#--weights=examples/nuclei/train_cifar/cifar_nuclei_quick1_iter_40000.caffemodel
#--snapshot=~/Projects/nuclei-net/caffe-scripts/nuclei/three_class_nuclei_1_iter_10000.solverstate \
