#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/nuclei/multi_class_nuclei/use_multi_class/use_multi_class_solver.prototxt \
	--snapshot=examples/nuclei/multi_class_nuclei/use_multi_class/use_multi_class_nuclei_big_1_iter_51647.solverstate \
	2>&1 | tee examples/nuclei/multi_class_nuclei/use_multi_class/train_use_multi_class_2_output.txt
#--snapshot=examples/nuclei/multi_class_nuclei/multi_class_nuclei_1_iter_10000.solverstate
#--weights=examples/nuclei/train_cifar/cifar_nuclei_quick1_iter_40000.caffemodel

