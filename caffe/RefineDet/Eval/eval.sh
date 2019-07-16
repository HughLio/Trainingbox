#!/bin/bash
set -eux
set -o pipefail

cp voc_eval.py /opt/caffe/test/lib/datasets/voc_eval.py
cp test.py /opt/caffe/test/lib/fast_rcnn/test.py
cp config.py /opt/caffe/test/lib/fast_rcnn/config.py
cp pascal_voc.py /opt/caffe/test/lib/datasets/pascal_voc.py
cp refinedet_test.py /opt/caffe/test/refinedet_test.py

# python  -u /opt/caffe/test/refinedet_test.py  -g 0  -p ../outputModels/ -s 320

python  -u /opt/caffe/test/refinedet_test.py  -g 0  \
        -w ../train-model_0329_VOC2007_320_v1/outputModels/VOC0712_refinedet_res18_320x320-v1_iter_42000_72_point_82.caffemodel  \
        -d ../train-model_0329_VOC2007_320_v1/lib/refinedet_res18_320x320_deploy.prototxt  \
        -s 320

# remove tmp pkl detection results
rm -r /opt/caffe/test/output/default/voc_2007_test/