#!/bin/sh 
# $1 - sample


cat > $1_OutputRun.sh <<EOF1
export SCRAM_ARCH=slc6_amd64_gcc700
cd /afs/desy.de/user/m/makou/NN_BBH_Analysis
source venv/bin/activate
cd /afs/desy.de/user/m/makou/NN_BBH_Analysis/pred_sys
python XGB_predict-index_Mary.py -s $1
EOF1


chmod u+x $1_OutputRun.sh
nohup ./$1_OutputRun.sh > nohup_$1_OutputRun.out &
