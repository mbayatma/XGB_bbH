import uproot
import pandas as pd
import numpy as np
import numpy.random as rand
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#imports for the Grid search
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.metrics import make_scorer
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras import layers, models, losses, regularizers, optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_sample_weight
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
import imblearn
from imblearn import under_sampling
from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from matplotlib import pyplot
from numpy import where
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from keras.models import model_from_yaml
import pickle
from pickle import load
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError, PrecisionRecallCurve
from yellowbrick.features import Rank2D
import os
from imblearn.pipeline import make_pipeline
import shap
import lightgbm
from lightgbm import LGBMClassifier
#open root files with uproot
#dir for the NTuples /nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/XGB_NTUPLES_18/NTuples_2018

#path_to_NTuples="/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/synch_NTuples_2018_neu/"
path_to_NTuples="/nfs/dust/cms/user/makou/BBH_Tuples/BBH_synchtuples/2018/"

columns=["pzeta", "m_vis", "pt_1", "pt_2", "mt_tot","nbtag","njets","bpt_1","jpt_1","jpt_2","pt_tt" ,"jdeta","iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","evt","run","os","qcdweight", "gen_nbjets","weightEMu", "trg_muhigh_elow","trg_ehigh_mulow", "gen_noutgoing","gen_nbjets_cut" ]


#2016_pre

#keys=["WZTo3LNu", "GluGluHToWWTo2L2Nu_M125" , "ST_tW_top_5f", "bbHToTauTau_yt2_M125", "W3JetsToLNu", "ZZTo2L2Nu", "W4JetsToLNu","ZZTo2L2Q","GluGluHToTauTau_M125", "bbHToTauTau_ybyt_M125", "MuonEG_Run2016B", "VBFHToTauTau_M125", "GluGluHToTauTau_M125_amcatnlo", "WZTo2L2Q", "ZHToWWTo2L2Nu_M125","ST_tW_antitop_5f", "WminusHToWWTo2L2Nu_M125", "DYJetsToLL_1J_amcatnlo", "ZHToTauTau_M125", "W1JetsToLNu", "VBFHToWWTo2L2Nu_M125", "DYJetsToLL_0J_amcatnlo", "bbHToWWTo2L2Nu_ybyt_M125", "W2JetsToLNu", "bbHToWWTo2L2Nu_yb2_M125", "bbHToTauTau_yb2_M125", "TTToSemiLeptonic", "MuonEG_Run2016C", "ZZTo4L", "MuonEG_Run2016E", "WplusHToWWTo2L2Nu_M125", "WplusHToTauTau_M125", "WminusHToTauTau_M125","ST_t-channel_top_4f", "DYJetsToLL_M-50_amcatnlo", "MuonEG_Run2016F", "ST_t-channel_antitop_4f", "DYJetsToLL_2J_amcatnlo","MuonEG_Run2016D", "TTToHadronic", "WJetsToLNu", "TTTo2L2Nu","WWTo2L2Nu"]


#2016 post_post

#keys= ["WZTo3LNu", "DYJetsToLL_M-50", "GluGluHToWWTo2L2Nu_M125", "ST_tW_top_5f", "DY3JetsToLL_M-50", "bbHToTauTau_yt2_M125", "W3JetsToLNu", "ZZTo2L2Nu", "W4JetsToLNu","DY2JetsToLL_M-50", "ZZTo2L2Q", "GluGluHToTauTau_M125", "MuonEG_Run2016H", "bbHToTauTau_ybyt_M125", "VBFHToTauTau_M125", "GluGluHToTauTau_M125_amcatnlo", "WZTo2L2Q", "ZHToWWTo2L2Nu_M125", "ST_tW_antitop_5f", "DY4JetsToLL_M-50", "WminusHToWWTo2L2Nu_M125", "DYJetsToLL_1J_amcatnlo", "ZHToTauTau_M125", "W1JetsToLNu", "VBFHToWWTo2L2Nu_M125", "DYJetsToLL_0J_amcatnlo", "bbHToWWTo2L2Nu_ybyt_M125","W2JetsToLNu", "bbHToWWTo2L2Nu_yb2_M125","bbHToTauTau_yb2_M125", "TTToSemiLeptonic", "ZZTo4L", "WplusHToWWTo2L2Nu_M125", "WplusHToTauTau_M125", "WminusHToTauTau_M125", "ST_t-channel_top_4f", "MuonEG_Run2016G", "DYJetsToLL_M-50_amcatnlo", "MuonEG_Run2016F", "EmbeddedElMu_Run2016H", "ST_t-channel_antitop_4f", "DYJetsToLL_2J_amcatnlo", "TTToHadronic", "WJetsToLNu", "TTTo2L2Nu", "WGToLNuG", "DY1JetsToLL_M-50", "WWTo2L2Nu", "EmbeddedElMu_Run2016G", "EmbeddedElMu_Run2016F"]


#2017 Keys

#keys= ["WZTo3LNu", "MuonEG_Run2017B", "EmbeddedElMu_Run2017E", "GluGluHToWWTo2L2Nu_M125", "ST_tW_top_5f", "bbHToTauTau_yt2_M125", "MuonEG_Run2017C", "W3JetsToLNu", "ZZTo2L2Nu", "EmbeddedElMu_Run2017F", "W4JetsToLNu", "ZZTo2L2Q", "GluGluHToTauTau_M125", "bbHToTauTau_ybyt_M125", "VBFHToTauTau_M125", "EmbeddedElMu_Run2017C", "WZTo2L2Q", "ZHToWWTo2L2Nu_M125", "ST_tW_antitop_5f", "WminusHToWWTo2L2Nu_M125", "DYJetsToLL_1J_amcatnlo", "ZHToTauTau_M125", "W1JetsToLNu", "VBFHToWWTo2L2Nu_M125", "DYJetsToLL_0J_amcatnlo", "bbHToWWTo2L2Nu_ybyt_M125", "W2JetsToLNu", "bbHToWWTo2L2Nu_yb2_M125", "MuonEG_Run2017D", "bbHToTauTau_yb2_M125", "TTToSemiLeptonic", "ZZTo4L", "MuonEG_Run2017E", "WplusHToWWTo2L2Nu_M125", "WplusHToTauTau_M125", "WminusHToTauTau_M125", "ST_t-channel_top_4f", "EmbeddedElMu_Run2017B", "EmbeddedElMu_Run2017D", "DYJetsToLL_M-50_amcatnlo", "ST_t-channel_antitop_4f", "DYJetsToLL_2J_amcatnlo", "TTToHadronic", "MuonEG_Run2017F", "WJetsToLNu", "TTTo2L2Nu", "WWTo2L2Nu"]


#keys 2018 

keys= ["WZTo3LNu","GluGluHToWWTo2L2Nu_M125", "ST_tW_top_5f", "bbHToTauTau_yt2_M125", "W3JetsToLNu", "ZZTo2L2Nu", "W4JetsToLNu","EmbeddedElMu_Run2018A", "ZZTo2L2Q", "GluGluHToTauTau_M125", "MuonEG_Run2018C", "MuonEG_Run2018B", "bbHToTauTau_ybyt_M125", "VBFHToTauTau_M125", "WZTo2L2Q", "ZHToWWTo2L2Nu_M125","ST_tW_antitop_5f", "MuonEG_Run2018D", "WminusHToWWTo2L2Nu_M125", "DYJetsToLL_1J_amcatnlo", "ZHToTauTau_M125", "EmbeddedElMu_Run2018D", "W1JetsToLNu", "VBFHToWWTo2L2Nu_M125", "DYJetsToLL_0J_amcatnlo", "bbHToWWTo2L2Nu_ybyt_M125", "W2JetsToLNu", "bbHToWWTo2L2Nu_yb2_M125", "bbHToTauTau_yb2_M125", "TTToSemiLeptonic", "ZZTo4L", "WplusHToWWTo2L2Nu_M125", "WplusHToTauTau_M125", "WminusHToTauTau_M125", "ST_t-channel_top_4f", "DYJetsToLL_M-50_amcatnlo", "ST_t-channel_antitop_4f", "DYJetsToLL_2J_amcatnlo", "EmbeddedElMu_Run2018B", "TTToHadronic", "WJetsToLNu", "TTTo2L2Nu", "MuonEG_Run2018A", "WWTo2L2Nu","EmbeddedElMu_Run2018C"]


general_cut= 'pzeta<20 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and ((pt_1 >24 and trg_ehigh_mulow > 0.5 ) | (pt_2 >24 and trg_muhigh_elow > 0.5)) and nbtag<=2'
#cut_ss = 'pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and os<0.5'

samples_list=[]
for samples in os.listdir(path_to_NTuples): 
    samples_list.append(path_to_NTuples+samples)
print("samples_list",samples_list) 
samples_dicts = {}
for i in range(len(keys)):
        samples_dicts[keys[i]] = samples_list[i]
print("samples_dicts",samples_dicts)

arr_all={}
for k , v in samples_dicts.items():
      with uproot.open(v) as sample :  
                 tree = sample['TauCheck']
                 samplearr=tree.arrays(columns, library= "pd")
                 selection=samplearr.query(general_cut)
                 even=selection[selection['evt']%2==0]
                 #even=samplearr[samplearr['evt']%2==0]
                 arr_all[k]=even                             
#print("arr_all", arr_all)
#for k,v in arr_all.items():



bbhsig=pd.concat([arr_all["bbHToTauTau_yt2_M125"], arr_all["bbHToTauTau_yb2_M125"]], axis=0)
bbHsig=bbhsig.query('gen_nbjets_cut>0 and os >0.5')
print('bbHsig',bbHsig.shape)
bbh= bbHsig.loc[:,'pzeta':'jdeta']
bbH=bbh.to_numpy()
label_bbh = 2*(np.ones(len(bbH)))
Higgs_sig = np.column_stack ((bbH,label_bbh))
print("Higgs_sig",Higgs_sig.shape)

hwwsig=pd.concat([arr_all["bbHToWWTo2L2Nu_yb2_M125"], arr_all["GluGluHToWWTo2L2Nu_M125"]], axis=0)
hWWsig=hwwsig.query('gen_nbjets_cut>0 and os >0.5')
print('hWWsig',hWWsig.shape)
Hww= hWWsig.loc[:,'pzeta':'jdeta']
HWW=Hww.to_numpy()
label_Hww = 3*(np.ones(len(HWW)))
Higgs_WW = np.column_stack ((HWW,label_Hww))
print("Higgs_WW",Higgs_WW.shape)

TT=pd.concat([arr_all["TTTo2L2Nu"], arr_all["TTToHadronic"], arr_all["TTToSemiLeptonic"]], axis=0)
TTbar=TT.query('os>0.5')
tt= TTbar.loc[:,'pzeta':'jdeta']
tt_bar=tt.to_numpy()
label_TTbar = np.zeros(len(tt_bar))
ttbar = np.column_stack ((tt_bar,label_TTbar))
print("ttbar_shape",ttbar.shape)


DYjets= pd.concat([arr_all["DYJetsToLL_M-50_amcatnlo"] ,arr_all["DYJetsToLL_0J_amcatnlo"] ,arr_all["DYJetsToLL_1J_amcatnlo"],arr_all["DYJetsToLL_2J_amcatnlo"]],axis=0)
DYJets=DYjets.query('os>0.5')
Ztt= DYJets.loc[:,'pzeta':'jdeta']
Z=Ztt.to_numpy()
label_ZTT = (np.ones(len(Z)))
ZTT = np.column_stack ((Z, label_ZTT))
print("ZTT_shape",ZTT.shape)
print("ZTT", ZTT)
Data= np.concatenate((ttbar,Higgs_WW,ZTT,Higgs_sig),axis=0)  
print('Data',Data)


Data_ran =shuffle(Data,random_state =10)
print('Data_random',Data_ran)
print('shape of data_ran', Data_ran.shape)
print('dimension', Data_ran.ndim)

#note that i mixed up scaler0 and 1 for 2017, 1 is used for even id, 0 is used for odd id
scaler0= StandardScaler()
#pickle.dump(scaler1, open('scaler1.pkl', 'wb'))
#scaler = load(open('scaler.pkl', 'rb'))
X = scaler0.fit_transform(Data_ran[:,0:12])
print('X',X)
print('shape of x',X.shape)
#np.save('X0_16post_UL_WWnobcsv.npy', X)    # .npy extension is added if not given
#pickle.dump(scaler0, open('scaler0_16post_UL_WWnobcsv.pkl', 'wb'))


Y= Data_ran[:,-1]
#np.save('Y.npy', Y)
#print('Y ',Y)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=5)
print("X_train_shape",X_train.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=5)
#sm = SMOTE(random_state=12)
#x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
#print("x_train_res",x_train_res.shape)
#fix_params ={'learning_rate': 0.1, 'n_estimators':10, 'objective':'multiclass', 'max_depth':3, 'min_child_weight':2 , 'max_delta_step':6, 'subsample': 0.8  , 'num_iterations': 10, 'num_class':5 }
#over = SMOTE(sampling_strategy={0:1137886, 1:100000 , 2:90000, 3:104544 , 4:249090}) 
#X_sm,  Y_sm = over.fit_resample(X,Y)

#under= RandomUnderSampler(sampling_strategy={0:87766, 1:74596 , 2:82268, 3:81368 })
#X_sm1, Y_sm1 = under.fit_resample(X,Y) 


fix_params ={'learning_rate': 0.1, 'n_estimators':100, 'objective':'meulti:softprob', 'max_depth':3, 'min_child_weight':2 , 'max_delta_step':6, 'subsample': 0.8  }
#classes_weight = class_weight.compute_class_weight({0:0.308,1:10., 2:9.8,3:1.5},np.unique(Y),y_train)
#print("classes_weight",classes_weight)
classes_weights = class_weight.compute_sample_weight('balanced', y_train)
#classes_weights = class_weight.compute_sample_weight({0:0.308,1:10., 2:9.8,3:1.5}, y_train)
#print ("classes_weight", classes_weights)
#model = LGBMClassifier()
model = XGBClassifier(**fix_params) 
eval_set = [(X_train, y_train), (X_val, y_val)]

#model.fit(X_train, y_train, sample_weight=classes_weights, eval_metric=["merror", "mlogloss"], eval_set=eval_set )
# evaluate model
#scores_1 = cross_val_score(model, X_train, y_train, scoring='accuracy', n_jobs=1)
#report performance
#print('Accuracy of train: %.3f (%.3f)' % (np.mean(scores_1), np.std(scores_1)))

#scores = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=1)
#report performance
#print('Accuracy of test: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#file_name = "xgb_train_model0_synchNTuples_trgcut.pkl"
# save to JSON
#model.save_model("model0_16post_UL_HWWnobcsv.json")
# save to text format
#model.save_model("model0_16post_UL_HWWnobcsv.txt")
# save
#pickle.dump(model, open(file_name, "wb"))

#model =pickle.load(open("model0_18_UL_HWWnobcsv.json","rb"))
model.load_model("model0_18_UL_HWWnobcsv.json")

result = model.score(X_test, y_test)
print('result', result)
print(classification_report(y_test, model.predict(X_test)))

result_2 = model.score(X_train, y_train)
print('result_2', result_2)
print(classification_report(y_train, model.predict(X_train)))

#comment out from here
y_pred = model.predict(X_test) 
for i in range(20):
      print(X[i], y_pred[i])
    
    
print ("y_pred", y_pred)
print("y_pred_shape", y_pred.shape)
print(classification_report(y_test, y_pred))


plt.figure(1)
cm = confusion_matrix(y_test,y_pred)
cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_df = pd.DataFrame(cmn,
                     index = ['ttbar','ZTT','Higgs_sig','HWW'], 
                    columns = ['ttbar','ZTT','Higgs_sig','HWW'])

sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
name=('CM_test_model0_16post_UL_hwwnobcsv')
plt.savefig('XGBoosted_output/{}'.format(name), dpi=500)


#Probability plot
#plt.figure(2)
#Z=model.predict_proba(X_test)
#plt.figure(figsize=(12,7))
#plt.hist(Z[:,0],bins=40,label='ttbar',color='c',alpha=0.9)
#plt.hist(Z[:,1],bins=40,label='Higgs_signal',color='m',alpha=0.8)
#plt.hist(Z[:,2],bins=40,label='ZTT',color='y',alpha=0.7)
#plt.hist(Z[:,3],bins=40,label='misc',color='r',alpha=0.5)
#plt.xlabel('Probability of each class after training on testset')
#plt.ylabel('Number of records in each bucket',fontsize=20)
#plt.legend(fontsize=15)
#name= ('Prob_Plot_model0_onsynch_trgcut')
#plt.savefig('XGBoosted_output/{}'.format(name))


#ROC Curve
plt.figure(3)
plt.title("ROC Curve and AUC")
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize= 16)

visualizer= ROCAUC(model, classes=['ttbar','ZTT','Higgs_sig','HWW'])
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
plt.legend()
name=('ROC_AUC_test0_16post_UL_HWWnobcsv')
plt.savefig('XGBoosted_output/{}'.format(name))


#Instantiate the visualizer with the Pearson ranking algorithm
#plt.figure(4)
#plt.title("Pearson Ranking of Features")
#visualizer=Rank2D(algorithm='pearson')
#visualizer.fit(X_train,y_train)
#visualizer.transform(X_train)
#name=('Pearson_ranking_test')
#plt.savefig('XGBoosted_output/{}'.format(name))

#report of classification
#plt.figure(5)
#report = ClassificationReport(model, classes=["tt","HS","ZTT","misc"])
#report.fit(X_train,y_train)
#report.score(X_test,y_test)
#name= ('XGB classification report test')
#plt.savefig('XGBoosted_output/{}'.format(name))

#class Prediction Error
#plt.figure(6)
#error=ClassPredictionError(model,classes=["tt","HS","ZTT","misc"])
#error.fit(X_train,y_train)
#error.score(X_test,y_test)
#name= ('prediction error model test')
#plt.savefig('XGBoosted_output/{}'.format(name))


plt.figure(7)
yhat = model.predict(X_test)
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % score)
# retrieve performance metrics
results = model.evals_result()
 #plot learning curves
pyplot.plot(results['validation_0']['mlogloss'], label='train')
pyplot.plot(results['validation_1']['mlogloss'], label='test')
# show the legend
pyplot.legend()
# show the plot
name=('loss-function_test_model0_16post_UL_HWWnobcsv')
plt.savefig('XGBoosted_output/{}'.format(name))


#plot feature importance
plt.figure(8)
plot_importance(model)
name=('Feature Importance test 2016post UL HWW nobcsv')
plt.savefig('XGBoosted_output/{}'.format(name))

# visualize the first prediction's explanation
#explainer = shap.Explainer(model)
#shap_values = explainer(X)
#plt.figure(9)
#shap.plots.waterfall(shap_values[0])
#name=('shape_values')
#plt.savefig('XGBoosted_output/{}'.format(name))


