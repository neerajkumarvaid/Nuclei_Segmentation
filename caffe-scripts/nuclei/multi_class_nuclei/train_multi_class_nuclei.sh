#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/nuclei/multi_class_nuclei/multi_class_nuclei_solver.prototxt \
	--snapshot=examples/nuclei/multi_class_nuclei/multi_class_nuclei_1_iter_10000.solverstate
	2>&1 | tee examples/nuclei/multi_class_nuclei/train_multi_class_nuclei_2_output.txt
#--weights=examples/nuclei/train_cifar/cifar_nuclei_quick1_iter_40000.caffemodel

