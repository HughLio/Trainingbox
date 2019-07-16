#!/usr/bin/env bash

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your_eval.log"
exit
fi

LOGPATH=`realpath $1`
PARENTPATH="${LOGPATH%/*}"
LOG=${PARENTPATH}/`basename $1`

# Get AP
cat $LOG | grep -e AP -e "Start evaluating:" > $LOG.AP

# Get mAP
echo '#Iters mAP' > aux0.txt
cat $LOG | grep "Start evaluating" | awk '{split($0,a,"iter_"); print a[2]}' | awk '{split($0,a,".caffemode"); print a[1]}' > aux1.txt
cat $LOG | grep Mean | awk '{split($0,a,"= "); print a[2]}'> aux2.txt

paste aux1.txt aux2.txt| column -t >> aux0.txt

cat aux0.txt | sort -n -t 't' -k 1 > $LOG.mAP

rm aux1.txt aux2.txt aux0.txt