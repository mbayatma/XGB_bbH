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
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError, PrecisionRecallCurve
from yellowbrick.features import Rank2D
import os
from imblearn.pipeline import make_pipeline
import shap
import lightgbm
from lightgbm import LGBMClassifier
#open root files with uproot
#dir for the NTuples /nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/XGB_NTUPLES_18/NTuples_2018

path_to_NTuples="/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/XGB_NTUPLES_18/NTuplesUL_2018/"
#columns=["mbb","dRbb","bcsv_1","bcsv_2","mt_tot","pzeta","pt_tt","m_sv","bpt_1", "mjj", "nbtag","jpt_1","puppimet","njets","dijetpt","beta_1","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","evt","run","os","qcdweight", "gen_nbjets"]


columns=["mt_tot","pzeta","pt_tt","bpt_1","mjj","nbtag","jpt_1","puppimet","njets","dijetpt","beta_1","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","evt","run","os","qcdweight", "gen_nbjets"]

keys= ["TTbar","BBH","Diboson","DYJets","ZHTT","VBFHTT","ggHTT","QCD","ST","WJets","WHTT"]
#keys= ["ggHww","VBFHww","TTbar","BBH","Diboson","DYJets","ZHTT","VBFHTT","ggHTT","ZHww","QCD","ST","WHww","WJets","WHTT"]
#this makes sense after defining the gen blabla for btag jets
general_cut= 'pzeta>-35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3'
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
                 even=selection[selection['evt']%2==1]
                 #even=samplearr[samplearr['evt']%2==0]
                 arr_all[k]=even                             
#print("arr_all", arr_all)
#for k,v in arr_all.items():

ggHsig=(arr_all["ggHTT"])[(arr_all["ggHTT"])["os"]>0.5]
print('ggHsig',ggHsig.shape)
bbHsig=(arr_all["BBH"])[(arr_all["BBH"])["os"]>0.5]
print('bbHsig',bbHsig.shape)
VBFsig=(arr_all["VBFHTT"])[(arr_all["VBFHTT"])["os"]>0.5]
print("VBFsig", VBFsig)
ZHsig = (arr_all["ZHTT"])[(arr_all["ZHTT"])["os"]>0.5]
WHsig= (arr_all["WHTT"])[(arr_all["WHTT"])["os"]>0.5]

H_sig= (pd.concat([ggHsig,bbHsig,VBFsig,ZHsig,WHsig],axis=0)).loc[:,'mt_tot':'pt_2'] 
Higgs_signal= H_sig.to_numpy()
label_Higgs_Signal= np.ones(len(Higgs_signal))
Higgs_Signal = np.column_stack ((Higgs_signal, label_Higgs_Signal))
print("Higgs_signal",Higgs_Signal.shape)


qcd=(arr_all["QCD"])[(arr_all["QCD"])["os"]<0.5]
Diboson=(arr_all["Diboson"])[(arr_all["Diboson"])["os"]>0.5]
ST=(arr_all["ST"])[(arr_all["ST"])["os"]>0.5]
WJets=(arr_all["WJets"])[(arr_all["WJets"])["os"]>0.5]
mix =(pd.concat([Diboson,ST,WJets,qcd],axis=0).loc[:,'mt_tot':'pt_2'])
misc = mix.to_numpy()
label_misc = 3*(np.ones(len(misc)))
miscelle = np.column_stack ((misc, label_misc))
print("miscelle",miscelle.shape)

TTbar=(arr_all["TTbar"])[(arr_all["TTbar"])["os"]>0.5]
tt= TTbar.loc[:,'mt_tot':'pt_2']
tt_bar=tt.to_numpy()
label_TTbar = np.zeros(len(tt_bar))
ttbar = np.column_stack ((tt_bar,label_TTbar))
print("ttbar_shape",ttbar.shape)

DYJets=(arr_all["DYJets"])[(arr_all["DYJets"])["os"]>0.5]
Ztt= DYJets.loc[:,'mt_tot':'pt_2']
Z=Ztt.to_numpy()
label_ZTT = 2*(np.ones(len(Z)))
ZTT = np.column_stack ((Z, label_ZTT))
print("ZTT_shape",ZTT.shape)

Data= np.concatenate((ttbar,Higgs_Signal,ZTT,miscelle),axis=0)  
print('Data',Data)


Data_ran =shuffle(Data,random_state =10)
print('Data_random',Data_ran)
print('shape of data_ran', Data_ran.shape)
print('dimension', Data_ran.ndim)


scaler= StandardScaler()
X = scaler.fit_transform(Data_ran[:,0:14])
print('X',X)
print('shape of x',X.shape)

Y= Data_ran[:,-1]
print('Y ',Y)



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

model.fit(X_train, y_train, sample_weight=classes_weights, eval_metric=["merror", "mlogloss"], eval_set=eval_set )
# evaluate model
scores_1 = cross_val_score(model, X_train, y_train, scoring='accuracy', n_jobs=1)
#report performance
print('Accuracy of train: %.3f (%.3f)' % (np.mean(scores_1), np.std(scores_1)))

scores = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=1)
#report performance
print('Accuracy of test: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

file_name = "xgb_train_model1_nomsv_bjetvar.pkl"

# save
pickle.dump(model, open(file_name, "wb"))

#model =pickle.load(open(file_name,"rb"))
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
                     index = ['ttbar','Higgs_Signal','ZTT','misc'], 
                    columns = ['ttbar','Higgs_Signal','ZTT','misc'])

sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
name=('CM_test_model1')
plt.savefig('XGBoosted_output/{}'.format(name))


#Probability plot
plt.figure(2)
Z=model.predict_proba(X_test)
plt.figure(figsize=(12,7))
plt.hist(Z[:,0],bins=40,label='ttbar',color='c',alpha=0.9)
plt.hist(Z[:,1],bins=40,label='Higgs_signal',color='m',alpha=0.8)
plt.hist(Z[:,2],bins=40,label='ZTT',color='y',alpha=0.7)
plt.hist(Z[:,3],bins=40,label='misc',color='r',alpha=0.5)
plt.xlabel('Probability of each class after training on testset')
plt.ylabel('Number of records in each bucket',fontsize=20)
plt.legend(fontsize=15)
name= ('Prob_Plot_model1')
plt.savefig('XGBoosted_output/{}'.format(name))


#ROC Curve
#plt.figure(3)
#plt.title("ROC Curve and AUC")
#plt.xlabel("False Positive Rate", fontsize=16)
#plt.ylabel("True Positive Rate", fontsize= 16)

#visualizer= ROCAUC(model, classes=["ttbar","Higgs_sig","ZTT","misc"])
#visualizer.fit(X_train,y_train)
#visualizer.score(X_test,y_test)
#plt.legend()
#name=('ROC_AUC_test')
#plt.savefig('XGBoosted_output/{}'.format(name))


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
name=('loss-function_test_model1')
plt.savefig('XGBoosted_output/{}'.format(name))


#plot feature importance
plt.figure(8)
plot_importance(model)
name=('Feature Importance test')
plt.savefig('XGBoosted_output/{}'.format(name))

# visualize the first prediction's explanation
explainer = shap.Explainer(model)
shap_values = explainer(X)
plt.figure(9)
shap.plots.waterfall(shap_values[0])
name=('shape_values')
plt.savefig('XGBoosted_output/{}'.format(name))


