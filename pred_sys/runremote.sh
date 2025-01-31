#!/bin/bash

SAMPLES="TTbar BBH Diboson DYJets ZHTT VBFHTT ggHTT QCD ST WJets WHTT"

if [ "$#" -eq 0 ]; then
    for SAMPLE in $SAMPLES; do
	echo $SAMPLE
	./HTC_submit_python.sh $SAMPLE
    done
else
    echo "Illegal number of arguments: run either with default output location with no arguments, or with the output directory as ./runOutputsnow.sh /nfs/dust/cms/user/cardinia/test_output"
fi
    