#!/bin/bash
SAMPLES="WZTo3LNu  WJetsToLNu_0J_amcatnlo  WJetsToLNu_amcatnlo  GluGluHToWWTo2L2Nu_M125  ST_tW_top_5f  bbHToTauTau_yt2_M125 W3JetsToLNu  ZZTo2L2Nu  WJetsToLNu_2J_amcatnlo  W4JetsToLNu  ZZTo2L2Q  GluGluHToTauTau_M125  MuonEG_Run2018C  MuonEG_Run2018B bbHToTauTau_ybyt_M125  VBFHToTauTau_M125  WZTo2L2Q  ZHToWWTo2L2Nu_M125  ST_tW_antitop_5f  MuonEG_Run2018D WminusHToWWTo2L2Nu_M125 DYJetsToLL_1J_amcatnlo  ZHToTauTau_M125  W1JetsToLNu  VBFHToWWTo2L2Nu_M125  DYJetsToLL_0J_amcatnlo  bbHToWWTo2L2Nu_ybyt_M125 W2JetsToLNu  bbHToWWTo2L2Nu_yb2_M125  bbHToTauTau_yb2_M125  TTToSemiLeptonic  WJetsToLNu_1J_amcatnlo  ttHToTauTau_M125  ZZTo4L WplusHToWWTo2L2Nu_M125   WplusHToTauTau_M125  WminusHToTauTau_M125  ST_t-channel_top_4f   DYJetsToLL_M-50_amcatnlo  ST_t-channel_antitop_4f  DYJetsToLL_2J_amcatnlo  TTToHadronic  WJetsToLNu  TTTo2L2Nu MuonEG_Run2018A  WWTo2L2Nu"


#SAMPLES_2017="WZTo3LNu WJetsToLNu_0J_amcatnlo  MuonEG_Run2017B  WJetsToLNu_amcatnlo GluGluHToWWTo2L2Nu_M125  ST_tW_top_5f  bbHToTauTau_yt2_M125  MuonEG_Run2017C  W3JetsToLNu  ZZTo2L2Nu  WJetsToLNu_2J_amcatnlo  W4JetsToLNu  ZZTo2L2Q GluGluHToTauTau_M125 bbHToTauTau_ybyt_M125  VBFHToTauTau_M125  WZTo2L2Q  ZHToWWTo2L2Nu_M125  ST_tW_antitop_5f  WminusHToWWTo2L2Nu_M125 DYJetsToLL_1J_amcatnlo  ZHToTauTau_M125  W1JetsToLNu  VBFHToWWTo2L2Nu_M125  DYJetsToLL_0J_amcatnlo  bbHToWWTo2L2Nu_ybyt_M125 W2JetsToLNu  bbHToWWTo2L2Nu_yb2_M125  MuonEG_Run2017D  bbHToTauTau_yb2_M125  TTToSemiLeptonic  WJetsToLNu_1J_amcatnlo ttHToTauTau_M125  ZZTo4L  MuonEG_Run2017E  WplusHToWWTo2L2Nu_M125  WplusHToTauTau_M125  WminusHToTauTau_M125  ST_t-channel_top_4f DYJetsToLL_M-50_amcatnlo  ST_t-channel_antitop_4f  DYJetsToLL_2J_amcatnlo  TTToHadronic  MuonEG_Run2017F  WJetsToLNu TTTo2L2Nu WWTo2L2Nu"
#SAMPLES="WZTo3LNu  WJetsToLNu_0J_amcatnlo  WJetsToLNu_amcatnlo  GluGluHToWWTo2L2Nu_M125  ST_tW_top_5f  bbHToTauTau_yt2_M125 W3JetsToLNu  ZZTo2L2Nu  WJetsToLNu_2J_amcatnlo  W4JetsToLNu  ZZTo2L2Q  GluGluHToTauTau_M125  MuonEG_Run2016H  bbHToTauTau_ybyt_M125  VBFHToTauTau_M125  WZTo2L2Q  ZHToWWTo2L2Nu_M125  ST_tW_antitop_5f  WminusHToWWTo2L2Nu_M125 DYJetsToLL_1J_amcatnlo  ZHToTauTau_M125  W1JetsToLNu  VBFHToWWTo2L2Nu_M125  DYJetsToLL_0J_amcatnlo  bbHToWWTo2L2Nu_ybyt_M125 W2JetsToLNu  bbHToWWTo2L2Nu_yb2_M125  bbHToTauTau_yb2_M125  TTToSemiLeptonic  WJetsToLNu_1J_amcatnlo  ttHToTauTau_M125  ZZTo4L WplusHToWWTo2L2Nu_M125  WplusHToTauTau_M125  WminusHToTauTau_M125  ST_t-channel_top_4f  MuonEG_Run2016G  DYJetsToLL_M-50_amcatnlo MuonEG_Run2016F  ST_t-channel_antitop_4f  DYJetsToLL_2J_amcatnlo  TTToHadronic  WJetsToLNu  TTTo2L2Nu WWTo2L2Nu"

#SAMPLES_2016pre="WZTo3LNu  WJetsToLNu_0J_amcatnlo WJetsToLNu_amcatnlo GluGluHToWWTo2L2Nu_M125  ST_tW_top_5f bbHToTauTau_yt2_M125 W3JetsToLNu  ZZTo2L2Nu  WJetsToLNu_2J_amcatnlo  W4JetsToLNu  ZZTo2L2Q  GluGluHToTauTau_M125  bbHToTauTau_ybyt_M125 MuonEG_Run2016B  VBFHToTauTau_M125  WZTo2L2Q  ZHToWWTo2L2Nu_M125  ST_tW_antitop_5f  WminusHToWWTo2L2Nu_M125  DYJetsToLL_1J_amcatnlo ZHToTauTau_M125  W1JetsToLNu  VBFHToWWTo2L2Nu_M125  DYJetsToLL_0J_amcatnlo  bbHToWWTo2L2Nu_ybyt_M125  W2JetsToLNu  bbHToWWTo2L2Nu_yb2_M125 bbHToTauTau_yb2_M125  TTToSemiLeptonic  WJetsToLNu_1J_amcatnlo  MuonEG_Run2016C  ttHToTauTau_M125  ZZTo4L  MuonEG_Run2016E WplusHToWWTo2L2Nu_M125  WplusHToTauTau_M125  WminusHToTauTau_M125  ST_t-channel_top_4f  DYJetsToLL_M-50_amcatnlo MuonEG_Run2016F ST_t-channel_antitop_4f  DYJetsToLL_2J_amcatnlo  MuonEG_Run2016D  TTToHadronic  WJetsToLNu  TTTo2L2Nu  WGToLNuG  WWTo2L2Nu"




if [ "$#" -eq 0 ]; then
    for SAMPLE in $SAMPLES; do
	echo $SAMPLE
	./NohupParserOutput.sh $SAMPLE
    done
else
    echo "Illegal number of arguments: run either with default output location with no arguments, or with the output directory as ./runOutputsnow.sh /nfs/dust/cms/user/cardinia/test_output"
fi
    


