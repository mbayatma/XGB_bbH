#!/bin/sh 
# $1 - filelist 

cat > $1_OutputRun.sh <<EOF1
export SCRAM_ARCH=slc6_amd64_gcc700
cd /afs/desy.de/user/m/makou/NN_BBH_Analysis
source venv/bin/activate
cd -
python XGB_Andrea.py -s $1
EOF1


chmod u+x $1_OutputRun.sh

cat > $1.submit <<EOF2
+RequestRuntime=70000

RequestMemory = 10000

executable = $1_OutputRun.sh

transfer_executable = True
universe            = vanilla
getenv              = True
Requirements        = OpSysAndVer == "CentOS7"

output              = $1.out
error               = $1.error
log                 = $1.log

queue

EOF2

chmod u+x $1.submit
condor_submit $1.submit


