# RefineDet
## training
准备工作：
+ 环境中有编译好的RefinDet
+ 以VOC的形式准备数据集，将lapmap.prototxt放在dataset目录下
+ 建立数据集软连接 `ln -s /*/*/* train-dataset`
+ 准备预训练模型，放入training的目录下

训练流程
1. `cd dataScript`
2. `./create_list.sh` 生成.txt,数据列表
3. `./create_data.sh` 生成lmdb格式数据集
4. 修改caffe路径，修改resize width 和 resize height
5. `cd training`
6. `./train.sh 2>&1 | tee train.log`
> 按需求调整预训练模型与训练参数

## evaluation
install
```bash
cd /opt/caffe/test/lib
make -j
```
create dataset link

```bash
ln -s /workspace/*/*/ VOC2007
```
Modify dataset root

修改 `pascal_voc.py`中数据集的存放路径

```python
self._devkit_path = os.environ['HOME'] + '/*/*/*/*'
self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
self._classes = ('__background__', # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
```

Modify parameters

- mean (for res18):
`/opt/caffe/test/lib/fast_rcnn/test.py` im_detect  # function

```python
def im_detect(net, im, targe_size):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_orig  = im_orig  * 0.017 # setting scale value
```

- scale:
`/opt/caffe/test/lib/fast_rcnn/config.py`

```python
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[104, 117, 123]]])
__C.PIXEL_MEANS = np.array([[[104, 117, 123]]])
```

```bash
cp voc_eval.py /opt/caffe/test/lib/datasets/voc_eval.py
cp test.py /opt/caffe/test/lib/fast_rcnn/test.py
cp config.py /opt/caffe/test/lib/fast_rcnn/config.py
cp pascal_voc.py /opt/caffe/test/lib/datasets/pascal_voc.py
cp refinedet_test.py /opt/caffe/test/refinedet_test.py
```

## eval

如果最后测试出现 IndexError: too many indices for array，那是因为你的测试数据中缺少了某些类别。请根据错误提示，找到对应的代码（$FRCN_ROOT/lib/datasets/voc_eval.py 第 148 行），前面加上一个 if 语句：
```python
if len(BB) != 0:
BB = BB[sorted_ind, :]
```


```bash
nohup sh eval.sh &
```

## parse log
生成ap的文件以及mAP文件
```bash
sh parse_eval.sh nohup.out
```
[Reference/RefineDet]: (https://github.com/sfzhang15/RefineDet.git)