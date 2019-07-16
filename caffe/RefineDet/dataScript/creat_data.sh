cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir
pwdDir=`cd $cur_dir/.. && pwd`
echo $pwdDir

redo=1
data_root_dir=$pwdDir"/dataset"
dataset_name="train-dataset"
mapfile=$pwdDir"/dataset/labelmap.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
#if [ $redo ]
#then
#  extra_cmd="$extra_cmd --redo"
#fi
for subset in trainval test
do
  python /opt/caffe/scripts/create_annoset.py \
  --anno-type=$anno_type \
  --label-map-file=$mapfile  \
  --min-dim=$min_dim  \
  --max-dim=$max_dim  \
  --resize-width=$width  \
  --resize-height=$height \
  --encode-type="jpg" \
  --encoded \
  --redo \
  --root=$data_root_dir \
  --listfile=$cur_dir/$subset.txt \
  --outdir=$data_root_dir/$db/$dataset_name"_"$subset"_"$db \
  --exampledir=examples/$dataset_name
done