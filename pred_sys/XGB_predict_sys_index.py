import sys
import uproot
import uproot3 as up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from numpy import argmax, where
import xgboost as xgb
from xgboost import XGBClassifier
from numpy import sort
import pickle
import awkward as ak
import os
from uproot_tree_utils import clone_tree, write_tree

def main(args):
    era               = args.era
    path_to_NTuples   = args.inputdir
    outdir            = args.outdir
    columns=["pzeta", "m_vis", "pt_1", "pt_2", "mt_tot","nbtag","njets","bpt_1","jpt_1","jpt_2","pt_tt" ,"jdeta","iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","evt","run","os","qcdweight", "gen_nbjets","weightEMu", "trg_muhigh_elow","trg_ehigh_mulow", "gen_noutgoing","gen_nbjets_cut" ]
    input_columns=["pzeta", "m_vis", "pt_1", "pt_2", "mt_tot","nbtag","njets","bpt_1","jpt_1","jpt_2","pt_tt" ,"jdeta"]
   
    extra_columns     = ["prob_0","prob_1","prob_2","prob_3","prob_4","pred_class","pred_proba"]
    #caution:these keys are for 2018 for other years the correct order should be checked!!!!
    keys=['WZTo3LNu', 'WJetsToLNu_0J_amcatnlo', 'WJetsToLNu_amcatnlo','GluGluHToWWTo2L2Nu_M125', 'ST_tW_top_5f', 'bbHToTauTau_yt2_M125', 'W3JetsToLNu', 'ZZTo2L2Nu', 'WJetsToLNu_2J_amcatnlo','W4JetsToLNu', 'ZZTo2L2Q', 'GluGluHToTauTau_M125', 'MuonEG_Run2018C', 'MuonEG_Run2018B', 'bbHToTauTau_ybyt_M125', 'VBFHToTauTau_M125', 'WZTo2L2Q', 'ZHToWWTo2L2Nu_M125', 'ST_tW_antitop_5f', 'MuonEG_Run2018D', 'WminusHToWWTo2L2Nu_M125', 'DYJetsToLL_1J_amcatnlo', 'ZHToTauTau_M125', 'W1JetsToLNu', 'VBFHToWWTo2L2Nu_M125', 'DYJetsToLL_0J_amcatnlo', 'bbHToWWTo2L2Nu_ybyt_M125', 'W2JetsToLNu', 'bbHToWWTo2L2Nu_yb2_M125', 'bbHToTauTau_yb2_M125', 'TTToSemiLeptonic', 'WJetsToLNu_1J_amcatnlo','ttHToTauTau_M125', 'ZZTo4L', 'WplusHToWWTo2L2Nu_M125', 'WplusHToTauTau_M125', 'WminusHToTauTau_M125', 'ST_t-channel_top_4f', 'DYJetsToLL_M-50_amcatnlo', 'ST_t-channel_antitop_4f', 'DYJetsToLL_2J_amcatnlo', 'TTToHadronic', 'WJetsToLNu', 'TTTo2L2Nu', 'MuonEG_Run2018A', 'WWTo2L2Nu']

   
    #keys_2017=['WZTo3LNu', 'WJetsToLNu_0J_amcatnlo', 'MuonEG_Run2017B', 'WJetsToLNu_amcatnlo', 'GluGluHToWWTo2L2Nu_M125', 'ST_tW_top_5f', 'bbHToTauTau_yt2_M125', 'MuonEG_Run2017C', 'W3JetsToLNu', 'ZZTo2L2Nu', 'WJetsToLNu_2J_amcatnlo', 'W4JetsToLNu', 'ZZTo2L2Q', 'GluGluHToTauTau_M125', 'bbHToTauTau_ybyt_M125', 'VBFHToTauTau_M125', 'WZTo2L2Q', 'ZHToWWTo2L2Nu_M125', 'ST_tW_antitop_5f', 'WminusHToWWTo2L2Nu_M125', 'DYJetsToLL_1J_amcatnlo', 'ZHToTauTau_M125', 'W1JetsToLNu', 'VBFHToWWTo2L2Nu_M125', 'DYJetsToLL_0J_amcatnlo', 'bbHToWWTo2L2Nu_ybyt_M125', 'W2JetsToLNu', 'bbHToWWTo2L2Nu_yb2_M125', 'MuonEG_Run2017D', 'bbHToTauTau_yb2_M125', 'TTToSemiLeptonic', 'WJetsToLNu_1J_amcatnlo', 'ttHToTauTau_M125', 'ZZTo4L', 'MuonEG_Run2017E', 'WplusHToWWTo2L2Nu_M125', 'WplusHToTauTau_M125', 'WminusHToTauTau_M125', 'ST_t-channel_top_4f', 'DYJetsToLL_M-50_amcatnlo', 'ST_t-channel_antitop_4f', 'DYJetsToLL_2J_amcatnlo', 'TTToHadronic', 'MuonEG_Run2017F', 'WJetsToLNu', 'TTTo2L2Nu', 'WWTo2L2Nu']
    #keys_2016post=['WZTo3LNu', 'WJetsToLNu_0J_amcatnlo', 'WJetsToLNu_amcatnlo', 'GluGluHToWWTo2L2Nu_M125', 'ST_tW_top_5f', 'bbHToTauTau_yt2_M125', 'W3JetsToLNu', 'ZZTo2L2Nu', 'WJetsToLNu_2J_amcatnlo', 'W4JetsToLNu', 'ZZTo2L2Q', 'GluGluHToTauTau_M125', 'MuonEG_Run2016H', 'bbHToTauTau_ybyt_M125', 'VBFHToTauTau_M125', 'WZTo2L2Q', 'ZHToWWTo2L2Nu_M125', 'ST_tW_antitop_5f', 'WminusHToWWTo2L2Nu_M125', 'DYJetsToLL_1J_amcatnlo', 'ZHToTauTau_M125', 'W1JetsToLNu', 'VBFHToWWTo2L2Nu_M125', 'DYJetsToLL_0J_amcatnlo', 'bbHToWWTo2L2Nu_ybyt_M125', 'W2JetsToLNu', 'bbHToWWTo2L2Nu_yb2_M125', 'bbHToTauTau_yb2_M125', 'TTToSemiLeptonic', 'WJetsToLNu_1J_amcatnlo', 'ttHToTauTau_M125', 'ZZTo4L', 'WplusHToWWTo2L2Nu_M125', 'WplusHToTauTau_M125', 'WminusHToTauTau_M125', 'ST_t-channel_top_4f', 'MuonEG_Run2016G', 'DYJetsToLL_M-50_amcatnlo', 'MuonEG_Run2016F', 'ST_t-channel_antitop_4f', 'DYJetsToLL_2J_amcatnlo', 'TTToHadronic', 'WJetsToLNu', 'TTTo2L2Nu','WWTo2L2Nu']
    
    #keys_2016pre=['WZTo3LNu', 'WJetsToLNu_0J_amcatnlo', 'WJetsToLNu_amcatnlo', 'GluGluHToWWTo2L2Nu_M125', 'ST_tW_top_5f', 'bbHToTauTau_yt2_M125', 'W3JetsToLNu', 'ZZTo2L2Nu', 'WJetsToLNu_2J_amcatnlo', 'W4JetsToLNu', 'ZZTo2L2Q', 'GluGluHToTauTau_M125', 'bbHToTauTau_ybyt_M125', 'MuonEG_Run2016B', 'VBFHToTauTau_M125', 'WZTo2L2Q', 'ZHToWWTo2L2Nu_M125', 'ST_tW_antitop_5f', 'WminusHToWWTo2L2Nu_M125', 'DYJetsToLL_1J_amcatnlo', 'ZHToTauTau_M125', 'W1JetsToLNu', 'VBFHToWWTo2L2Nu_M125', 'DYJetsToLL_0J_amcatnlo', 'bbHToWWTo2L2Nu_ybyt_M125', 'W2JetsToLNu', 'bbHToWWTo2L2Nu_yb2_M125', 'bbHToTauTau_yb2_M125', 'TTToSemiLeptonic', 'WJetsToLNu_1J_amcatnlo', 'MuonEG_Run2016C', 'ttHToTauTau_M125', 'ZZTo4L', 'MuonEG_Run2016E', 'WplusHToWWTo2L2Nu_M125', 'WplusHToTauTau_M125', 'WminusHToTauTau_M125', 'ST_t-channel_top_4f', 'DYJetsToLL_M-50_amcatnlo', 'MuonEG_Run2016F', 'ST_t-channel_antitop_4f', 'DYJetsToLL_2J_amcatnlo', 'MuonEG_Run2016D', 'TTToHadronic', 'WJetsToLNu', 'TTTo2L2Nu', 'WGToLNuG', 'WWTo2L2Nu']
    
    
    
    runsample         = ' '.join(args.sample)


    trees= ['TauCheck', 'TauCheck_CMS_scale_e_13TeVDown', 'TauCheck_CMS_scale_e_13TeVUp','TauCheck_CMS_scale_j_JES_13TeVDown',
    'TauCheck_CMS_scale_j_JES_13TeVUp', 'TauCheck_CMS_res_j_13TeVDown','TauCheck_CMS_res_j_13TeVUp','TauCheck_CMS_scale_met_unclustered_13TeVDown',
    'TauCheck_CMS_scale_met_unclustered_13TeVUp']


    samples_list=[]
    for samples in os.listdir(path_to_NTuples): 
        samples_list.append(path_to_NTuples+samples)
    print("samples_list",samples_list)    


    samples_dicts = {}
    for i in range(len(keys)):
            samples_dicts[keys[i]] = samples_list[i]
    print("samples_dicts",samples_dicts)

    samplefile = samples_dicts[runsample]
    print('samplefile',samplefile)
   
    #substring='MuonEG_Run2018'
    #DATA=[s for s in samplefile if substring in s]
    #DATA= {k: v for k, v in samplefile.items() if substring in v}
    #print('DATA',DATA)
    
    for tree_name in trees:
        with uproot.open(samplefile) as f:
            tree = f[tree_name]
            tree.keys()
            samplearr=tree.arrays(columns, library= "pd")
            orig_index_df = tree.arrays(['evt','run'], library='pd')

            orig_index = list(map(tuple, orig_index_df.values))

            #print("orig_index", orig_index)
            print("unique index", orig_index_df.index.is_unique)
            print("dup",orig_index_df.index.duplicated())

            samplearr=tree.arrays(columns, library= "pd")
            even= samplearr[samplearr['evt']%2==0]
            print("even",even)
            even_index= even.loc[:, 'evt':'run']

            print("even_index", even_index.shape)
            #orig_even=list(map(tuple, even_index.values))
            #print("orig_even", orig_even)
            sample_vars_even=even.loc[:,'pzeta':'jdeta']
            print("sample_vars",sample_vars_even)
            odd = samplearr[samplearr['evt']%2==1]
            odd_index= odd.loc[:, 'evt':'run']
            print("odd-index",odd_index)
            sample_vars_odd=odd.loc[:,'pzeta':'jdeta']
            index_odd_even= pd.concat([even_index,odd_index],axis=0)
            print("index_odd_even", index_odd_even)
            inputEventsH = f['inputEventsH']
            nWeightedEvents = f['nWeightedEvents']
            #saved model

        model0 = XGBClassifier()

        model0.load_model("/afs/desy.de/user/m/makou/NN_BBH_Analysis/model0_18UL_wdbclass.json")

        model1 = XGBClassifier()
        model1.load_model("/afs/desy.de/user/m/makou/NN_BBH_Analysis/model1_18UL_wdbclass.json")


        scaler0=pickle.load(open('/afs/desy.de/user/m/makou/NN_BBH_Analysis/scaler0_18UL_wdbclass.pkl','rb'))
        scaler1=pickle.load(open('/afs/desy.de/user/m/makou/NN_BBH_Analysis/scaler1_18UL_wdbclass.pkl','rb'))

        sample_std_even=scaler0.transform(sample_vars_even[input_columns])

        sample_std_odd=scaler1.transform(sample_vars_odd[input_columns]) 
        print('sample_std\n',sample_std_even.shape)
        predictions=np.empty((0,7),float)

        prediction0=model1.predict_proba(sample_std_even)
        print("prediction0", prediction0)
        pred0=pd.DataFrame(prediction0, columns=["prob_0","prob_1","prob_2","prob_3","prob_4"])
        print("pred0", pred0)

        pred0.index=even_index.index
        pred_0=pd.concat([pred0,even_index], axis=1)
        print("pred_0",pred_0)

        prediction1=model0.predict_proba(sample_std_odd)
        pred1=pd.DataFrame(prediction1, columns=["prob_0","prob_1","prob_2","prob_3","prob_4"])
        pred1.index=odd_index.index

        pred_1=pd.concat([pred1, odd_index], axis=1)
        print("pred_1", pred_1)

        prediction = pd.concat([pred_0, pred_1], sort=False).sort_index()
        print("prediction_sorted", prediction)


        index_corr=prediction.loc[:,'evt':'run']
        pred= prediction.loc[:, 'prob_0':'prob_4']
        print('pred',pred)
        prednp= pred.to_numpy()
        print(prednp)
        pred_class = np.argmax(prednp, axis=-1).astype(np.int32)
        print("pred_class_shape",pred_class)
        pred_class_proba = np.max(prednp, axis=-1).astype(np.float32)
        print("pred_class_proba.shape", pred_class_proba)

        #prediction_full= np.concatenate((prednp,pred_class,pred_class_proba),axis=1)

        prediction= np.insert(prednp,5,pred_class,axis=1)
        prediction_full= np.insert(prediction,6,pred_class_proba,axis=1)
        print("prediction_full",prediction_full.shape)

        predictions= pd.DataFrame(prediction_full, columns=["prob_0","prob_1","prob_2","prob_3","prob_4","pred_class","pred_proba"])
        print("predictions", predictions)

        predictions.index = index_corr.index
        pred_df= pd.concat([predictions, index_corr], axis=1)
        print("pred_df",pred_df)
        df_pred = pred_df.set_index(['evt','run'])
        print("df_pred",df_pred)
        print("unique index", df_pred.index.is_unique)
        print("dup",df_pred.index.duplicated())

    branches={}
    sample_dictionary={}
    for branch in extra_columns:
        print("branch name", branch)
        print("df_pred_2:column\n",np.array(df_pred[branch]))
        if branch in ['pred_class','run']:
            branches[branch]=np.int32
        elif branch in ['evt']:
            branches[branch]=np.uint64  # np.int_
        else:
            branches[branch]=np.float32
        sample_dictionary[branch]=np.array(df_pred[branch])

    #remove the year at the end to be comatible with synch Ntuples and the setting for create-dnn
    #file = up.recreate("/nfs/dust/cms/user/makou/BBH_Tuples/BBH_predictions/pred_systematics/2018/"+runsample+".root")
    file = up.recreate(outdir+runsample+".root")
    file["nWeightedEvents"] = nWeightedEvents.to_numpy()
    file['inputEventsH'] = inputEventsH.to_numpy()
    for tree_name in trees:    
        file[tree_name] = up.newtree(branches)
        file[tree_name].extend(sample_dictionary)

if __name__ == "__main__":
    from argparse import ArgumentParser
    argv = sys.argv
    description = """Create rootfiles with BDT prediction including systematics uncertainties"""
    parser = ArgumentParser(prog="XGB_predict",description=description,epilog="Good luck!")
    parser.add_argument('-y', '--era',     dest='era', nargs=1, choices=['2016pre','2016post','2017','2018'], default='2018', action='store',
                        help="set era" )
    parser.add_argument('-i', '--inputdir', dest='inputdir', nargs=1, default='/nfs/dust/cms/user/makou/BBH_Tuples/BBH_synchtuples/Synch_systematics/2018/', action='store',
                        help="Input directory for Synch Ntuples with systematics" )
    parser.add_argument('-o', '--outdir', dest='outdir', nargs=1, default='/nfs/dust/cms/user/makou/BBH_Tuples/BBH_predictions/pred_systematics/new_train_wDB/2018/', action='store',
                        help="Output directory for prediction synch Ntuples with systematics" )
    parser.add_argument('-s', '--sample', dest='sample', nargs=1, default="BBH", action='store',
                        help="Sample on which to run prediction" )
    args = parser.parse_args()
    main(args)
    print("\n>>> Done.")
