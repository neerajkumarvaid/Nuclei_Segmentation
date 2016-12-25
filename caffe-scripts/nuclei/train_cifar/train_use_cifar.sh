#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/nuclei/train_cifar/use_cifar_solver.prototxt \
	--weights=examples/nuclei/train_cifar/cifar_nuclei_quick1_iter_40000.caffemodel 2>&1 | tee examples/nuclei/train_use_cifar_7_output.txt
#--snapshot=examples/nuclei/train_cifar/use_cifar_2_iter_10000.solverstate
