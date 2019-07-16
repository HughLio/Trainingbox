# Mxnet Training
## Training
环境：
mxnet vession >=1.4
docopt， future，pyaml等

准备数据：
+ `cd train`
+ `python make_recordio_lst.py /path/to/*.txt /path/to/*.lst`
+ `python im2rec.py path/to/*.lst / --quality 100 --num-thread 64 --resize 256 --force-resize True`

准备预训练模型：
+ 以前缀的形式命名： \*-symobol.json, \*-0000.params

修改配置文件：
+ 类似example_train.yaml 配置参数

训练
`python -u mxnet_train.py /path/to/*.yaml`