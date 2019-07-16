#!/bin/bash
caffe train -solver="solver.prototxt" \
-weights="preModels/resnet18.v2.caffemodel" \
-gpu 0,1,2,3