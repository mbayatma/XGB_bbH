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
import shap

#open root files with uproot
#dir for the NTuples /nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018

ZTT = uproot.open("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_DYJets_2018.root")

ggh = uproot.open ("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_GluGluHToTauTau_2018.root")

bbH = uproot.open("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_BBH_2018.root")

TTbar= uproot.open("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_TTbar_2018.root")

Diboson= uproot.open("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_Diboson_2018.root")

WJets= uproot.open("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_WJets_2018.root")

SingleTop= uproot.open("/nfs/dust/cms/user/makou/bbh_analysis_NTuples/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_SingleTop_2018.root")


# read branches from Root as Numpy array

tree_ggh = ggh["TauCheck"]
tree_ggh.keys()
signal_ggh=tree_ggh.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")

selection_ggh = signal_ggh.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and nbtag>=1')

ggh_selec = selection_ggh.loc[:,'pzeta':'xsec_lumi_weight']

#bbH class
tree_bbH = bbH["TauCheck"]
signal_bbH=tree_bbH.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")

#print('ggh',signal)

selection_bbH = signal_bbH.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and nbtag>=1')

bbH_selec = selection_bbH.loc[:,'pzeta':'xsec_lumi_weight']


#fifth class ZTT

tree_ZTT = ZTT["TauCheck"]
ztt=tree_ZTT.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")


selection_ZTT = ztt.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and nbtag>=1 and (gen_match_1==3 and gen_match_2==4)')

ZTT_selec = selection_ZTT.loc[:,'pzeta':'xsec_lumi_weight']

#TTbar class 
tree_TTbar = TTbar["TauCheck"]
ttbar=tree_TTbar.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")


selection_TTbar = ttbar.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5  and dr_tt> 0.3 and nbtag>=1')

TTbar_selec = selection_TTbar.loc[:,'pzeta':'xsec_lumi_weight']


#forth class Diboson

tree_Diboson = Diboson["TauCheck"]
tree_Diboson.keys()
diboson=tree_Diboson.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")


selection_Diboson = diboson.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and nbtag>=1')

Diboson_selec = selection_Diboson.loc[:,'pzeta':'xsec_lumi_weight']


tree_WJets = WJets["TauCheck"]
tree_WJets.keys()
wjets=tree_WJets.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")


selection_WJets = wjets.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and nbtag>=1')

WJets_selec = selection_WJets.loc[:,'pzeta':'xsec_lumi_weight']


tree_SingleTop = SingleTop["TauCheck"]
tree_SingleTop.keys()
singletop=tree_SingleTop.arrays(["pzeta","pt_tt","m_sv","bpt_1", "mjj", "dijeteta","jpt_1","puppimet","njets","dijetpt","beta_2","d0_1","d0_2","pt_1","pt_2","xsec_lumi_weight", "iso_1", "iso_2","extraelec_veto","extramuon_veto","q_1","q_2","gen_match_1" ,"gen_match_2","dr_tt","nbtag"], library= "pd")


selection_SingleTop = singletop.query('pzeta > -35 and pt_2>15 and pt_1>15 and iso_1<0.15 and iso_2<0.2 and (pt_1 >24|pt_2 >24) and  extraelec_veto<0.5 and extramuon_veto<0.5 and dr_tt> 0.3 and nbtag>=1')

SingleTop_selec = selection_SingleTop.loc[:,'pzeta':'xsec_lumi_weight']

###misc train test set
SingleTop= SingleTop_selec.to_numpy()
SingleTop_train= SingleTop[:30604,:16]
print('SingleTop_train', SingleTop_train.shape)
SingleTop_test=SingleTop[30604:61207, :16]
print('SingleTop_test', SingleTop_test.shape)

WJets=WJets_selec.to_numpy()
WJets_train= WJets[:94,:16]
print("WJets_train", WJets_train.shape)
WJets_test=WJets[94:187,:16]
print("WJets_test",WJets_test.shape)

Diboson=Diboson_selec.to_numpy()
Diboson_train= Diboson[:17522,:16]
print("Diboson_train",Diboson_train)
Diboson_test=Diboson[17522:35043,:16]
print("Diboson_test",Diboson_test)
#misc class is the concat of Diboson, SingleTop and WJets

misc_train = np.concatenate((SingleTop_train,WJets_train,Diboson_train),axis=0)
print('misc_train', misc_train)
print ('misc_train shape', misc_train.shape)

misc_test = np.concatenate((SingleTop_test,WJets_test,Diboson_test),axis=0)
print('misc_test', misc_test)
print ('misc_test shape', misc_test.shape)


#labeling background to 0 signal to 1

label_TTbar = np.zeros(len(TTbar_selec))
ttbar = np.column_stack ((TTbar_selec,label_TTbar))
print('TTbar_label',label_TTbar)
print('TTbar.shape', ttbar.shape)
ttbar_train=ttbar[0:1025032, 0:17 ]
print("ttbar_train",ttbar_train)
ttbar_test= ttbar[1025032:2050064, 0:17]
print("ttbar_test",ttbar_test.shape)

label_ggh = np.ones(len(ggh_selec))
ggh = np.column_stack ((ggh_selec, label_ggh))
print('ggh_label', label_ggh)
ggh_train= ggh[:157,:17]
print("ggh_train", ggh_train.shape)
ggh_test=ggh[157:313,:17]
print("ggh_test", ggh_test.shape)

label_bbH = 2*(np.ones(len(bbH_selec)))
bbH = np.column_stack ((bbH_selec, label_bbH))
print('bbh_label', label_bbH)
bbH_train= bbH[:980,:17]
print("bbH_train", bbH_train)
bbH_test= bbH[980:1959, :17]
print("bbH_test",bbH_test)

label_ZTT = 3*(np.ones(len(ZTT_selec)))
ZTT = np.column_stack ((ZTT_selec, label_ZTT))
print('ZTT_label', label_ZTT)
ZTT_train= ZTT[:4971,:17]
print("ZTT_train", ZTT_train.shape)
ZTT_test=ZTT[4971:9941,:17]

label_misc_train = 4*(np.ones(len(misc_train)))
misc_lab_train = np.column_stack ((misc_train, label_misc_train))
print('label_misc_train', label_misc_train)


label_misc_test = 4*(np.ones(len(misc_test)))
misc_lab_test = np.column_stack ((misc_test, label_misc_test))
print('label_misc_train', label_misc_train)


Data_train= np.concatenate((ttbar_train,ggh_train,bbH_train,ZTT_train,misc_lab_train),axis=0)  
print('Data_train',Data_train)

Data_test= np.concatenate((ttbar_test,ggh_test,bbH_test,ZTT_test,misc_lab_test),axis=0)  
print('Data_test',Data_test)

Data = np.concatenate((Data_train,Data_test),axis=0)  
print('Data',Data)


Data_ran_train = shuffle(Data_train,random_state =10)
print('Data_random',Data_ran_train)
print('shape of data_ran', Data_ran_train.shape)
print('dimension', Data_ran_train.ndim)

Data_ran_test = shuffle(Data_test,random_state =10)
print('Data_random',Data_ran_test)
print('shape of data_ran', Data_ran_test.shape)
print('dimension', Data_ran_test.ndim)


x_train= Data_ran_train[:,0:15]
print('x_train',x_train)
print('shape of x',x_train.shape)

Y_train= Data_ran_train[:,-1]
print('Y of Data_ran',Y_train)

x_test= Data_ran_test[:,0:15]
print('x_test',x_test)
print('shape of x',x_test.shape)

Y_test= Data_ran_test[:,-1]
print('Y of Data_ran',Y_test)

Y= np.concatenate((Y_train,Y_test),axis=0)
print('Y',Y)

#preprocessing of the input data

scaler= StandardScaler()
standardized_train = scaler.fit_transform(x_train)
standardized_test = scaler.fit_transform(x_test)


#split arrays into random train and test subsets with train test split 

X_train_1= standardized_train
#print('X_train',X_train.shape)
X_test_1=standardized_test
print('X_test_1',X_test_1.shape)

X= np.concatenate((X_train_1,X_test_1),axis=0)
for i in range(20):
      print("X before training", X[i])
    

6437
smote= SMOTE(sampling_strategy={0:2050064 ,1:100000 , 2:700000 , 3:9941 ,4:96437 })
X_sm,  Y_sm = smote.fit_resample(X,Y)
for i in range(20):
    print("X_sm before training", X_sm[i])

#print('X_sm_shape', X_sm.shape)
#counter_sm = Counter(Y_sm)
#print('counter_sm',counter_sm)
#df= pd.DataFrame(X_sm)
#df['target'] = Y_sm
#plot = df.target.value_counts().plot(kind= 'bar', title= 'Count (classes)')
#fig = plot.get_figure()
#name= ('bar plot of each class oversampled with SMOTE')
#fig.savefig('NN_output/{}'.format(name))    
#for label, _ in counter_sm.items():
#    row_ix = where(Y_sm == label)[0]
#    pyplot.scatter(X_sm[row_ix, 0], X_sm[row_ix, 1],label=str(label))    
#plt.figure(1)
#pyplot.legend()
#plt.xlim(-3.5, 20)
#plt.ylim(-3.5, 20)
#name=('scatterplot_SMOTE')
#pyplot.savefig('NN_output/{}'.format(name))





X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.30, random_state=5)
#cv_params= {'max_depth': [1, 2, 3, 4, 5, 6], 'min_child_weight': [1,2,3,4]} #parameters to be tried in the grid search
#fix_params ={'learning_rate': 0.2, 'n_estimators':10 }
#my_scorer= make_scorer(f1_score, greater_is_better=True , average='micro')
#csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring= my_scorer, cv=5)
#csv.fit(X_train, y_train)
#print(csv.best_params_)

#print('y_train', y_train)
#print('y_train_shape', y_train.shape)
#print('X_train',X_train)
#print('X_train_shape', X_train.shape)
#lc = LabelEncoder() 
#lc = lc.fit(Y) 
#lc_y = lc.transform(Y)
#print('lc_y',lc_y)

#from here uncomment
#conf_matrix_list_of_arrays = []

#from sklearn.model_selection import RepeatedKFold 
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_validate, cross_val_score
#kf = KFold(n_splits= 5,random_state=None, shuffle=False)
#kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 
#for train_index, test_index in kf.split(X_train_1):    
#    X_train, X_test = X_train_1[train_index], X_train_1[test_index]
#    y_train, y_test = Y_train[train_index], Y_train[test_index]
#    print("Train:", train_index, "Validation:",test_index)        
 




classes_weights = class_weight.compute_sample_weight('balanced', y_train)

#print ('np.unique(y_train.argmax(axis=1)', np.unique(lc_y))
#print ('class_weights',classes_weights )     
model = XGBClassifier(n_estimator=5) 
eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train, sample_weight=classes_weights,early_stopping_rounds=10, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True )

 #explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_train)

# visualize the first prediction's explanation
plt.figure(1)
shap.plots.waterfall(shap_values[0])
name= ('shap_values')
plt.savefig('XGBoosted_output/{}'.format(name))



#plt.figure(1)
#yhat = model.predict(X_test)
#score = accuracy_score(y_test, yhat)
#print('Accuracy: %.3f' % score)
# retrieve performance metrics
#results = model.evals_result()
# plot learning curves
#pyplot.plot(results['validation_0']['mlogloss'], label='train')
#pyplot.plot(results['validation_1']['mlogloss'], label='test')
# show the legend
#pyplot.legend()
# show the plot
#name=('loss-function_oldmodel')
#plt.savefig('XGBoosted_output/{}'.format(name))




#plt.figure(2)
#conf_matrix = confusion_matrix(y_test, model.predict(X_test))
#cmn = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:, np.newaxis]
#cm_df = pd.DataFrame(cmn,
#                     index = ['ttbar','ggH','bbH','ZTT','misc'], 
#                    columns = ['ttbar','ggH','bbH','ZTT','misc'])

#sns.heatmap(cm_df, annot=True)
#plt.title('Confusion Matrix')
#plt.ylabel('Actual Values')
#plt.xlabel('Predicted Values')
#name=('confusion_matrix_Xtest_oldmodel')
#plt.savefig('XGBoosted_output/{}'.format(name))





#conf_matrix_list_of_arrays .append(conf_matrix)
#mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
#print('mean_of_conf_matrix_arrays',mean_of_conf_matrix_arrays)


# evaluate model
#scores = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=1)
#report performance
#print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#file_name = "xgb_oldmodel.pkl"

# save
#pickle.dump(model, open(file_name, "wb"))



# load
#xgb_model_loaded= pickle.load(open(file_name, "rb"))
#result = xgb_model_loaded.score(X_test, y_test)
#print('result', result)
#print(classification_report(y_test, xgb_model_loaded.predict(X_test)))



#comment out from here
y_pred = model.predict(X) 
for i in range(20):
      print(X[i], y_pred[i])
    
    
print ("y_pred", y_pred)
print("y_pred_shape", y_pred.shape)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

result = model.score(X, Y)
print('result', result)

print(classification_report(Y, y_pred))


plt.figure(2)
cm = confusion_matrix(Y,y_pred)
cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_df = pd.DataFrame(cmn,
                     index = ['ttbar','ggH','bbH','ZTT','misc'], 
                    columns = ['ttbar','ggH','bbH','ZTT','misc'])

sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
name=('confusion_matrix_oldmodel_predictions_wholedataset_withoutsaveloadmodel')
plt.savefig('XGBoosted_output/{}'.format(name))

bbH_train= bbH[:980,:15] 
bbH_test= bbH[980:1959, :15]
print("bbH_test",bbH_test.shape)
bbh= np.concatenate((bbH_train,bbH_test),axis=0)

scaler= StandardScaler()
bbh_standardized = scaler.fit_transform(bbh)
y_pred_bbh = model.predict(bbh_standardized) 
print("y_pred_bbh",y_pred_bbh.shape)
print("y_pred_bbh all",y_pred_bbh)
for i in range(10):
      print(bbh_standardized[i], y_pred_bbh[i])
print("y_pred_bbh", y_pred_bbh)        


ggh_train= ggh[:157,:15]
print("ggh_train", ggh_train.shape)
ggh_test=ggh[157:313,:15]
print("ggh_test", ggh_test.shape)

ggh = np.concatenate((ggh_train,ggh_test),axis=0)
scaler= StandardScaler()
ggh_standardized = scaler.fit_transform(ggh)

y_pred_ggh = model.predict(ggh_standardized) 
print("y_pred_ggh",y_pred_ggh.shape)
print("y_pred_ggh all",y_pred_ggh)
for i in range(10):
      print(ggh_standardized[i], y_pred_ggh[i])
print("y_pred_ggh", y_pred_ggh)     


#file_name = "xgb_X_train_1.pkl"

# save
#pickle.dump(model, open(file_name, "wb"))



# load
#xgb_model_loaded= pickle.load(open(file_name, "rb"))
#result = xgb_model_loaded.score(X_train_1, Y_train)
#print(result)
# test
#ind = 1
#test = X_test[ind]
#xgb_model_loaded.predict(test)[0] == xgb_model.predict(test)[0]

#y_pred_loaded = xgb_model_loaded.predict(X_train_1) 
#for i in range(10):
#    print(X_train_1[i], y_pred_loaded[i])
    
#print(classification_report(Y_train, y_pred_loaded))


#plt.figure(3)
#cm = confusion_matrix(Y_train,y_pred_loaded)
#cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#cm_df = pd.DataFrame(cmn,
#                     index = ['ttbar','ggH','bbH','ZTT','misc'], 
#                     columns = ['ttbar','ggH','bbH','ZTT','misc'])

#sns.heatmap(cm_df, annot=True)
#plt.title('Confusion Matrix')
#plt.ylabel('Actual Values')
#plt.xlabel('Predicted Values')
#name=('confusion_matrix_KFOLD_predictionontheTRAINhalf_smote')
#plt.savefig('XGBoosted_output/{}'.format(name))

    
    

#proba_class = xgb_model_loaded.predict_proba(X)
#print("all_classes_proba",proba_class)

#pred_class= np.argmax(proba_class, axis=1)
#print("pred_class", pred_class)

#predicted_class_proba= np.max(proba_class, axis=1)
#print("predicted_class_proba",predicted_class_proba)


#proba_class_0= xgb_model_loaded.predict_proba(X)[:,0]
#print("proba_class_0",proba_class_0)

#proba_class_1= xgb_model_loaded.predict_proba(X)[:,1]
#print("proba_class_1",proba_class_1)

#proba_class_2= xgb_model_loaded.predict_proba(X)[:,2]
#print("proba_class_2",proba_class_2)

#proba_class_3= xgb_model_loaded.predict_proba(X)[:,3]
#print("proba_class_3",proba_class_3)

#proba_class_4= xgb_model_loaded.predict_proba(X)[:,4]
#print("proba_class_4",proba_class_4)

#file = uproot.update("/afs/desy.de/user/m/makou/NN_BBH_Analysis/em-NOMINAL_ntuple_DYJets_2018.root")

#file= uproot.recreate("em-NOMINAL_ntuple_DYJets_2018.root")
#file["TauCheck"]= 
#file["tree"] = uproot.newtree({"branch1" : int,
#                               "branch2" : int,
#                                "branch3": np.array()})

#file["TauCheck"].extend({"branch1":proba_class_0, "branch2":proba_class_1,"branch3":proba_class_2})

#print("X", X.shape)
#print ("y_pred", y_pred)
#print("y_pred_shape", y_pred.shape)
#predictions = [round(value) for value in y_pred]
#accuracy = accuracy_score(y_test, predictions) 
#print("Accuracy: %.2f%%" % (accuracy * 100.0))



# serialize model to YAML
#model_yaml = model.to_yaml()
#with open("model.yaml", "w") as yaml_file:
#    yaml_file.write(model_yaml)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
 
# later...
 
# load YAML and create model
#yaml_file = open('model.yaml', 'r')
#loaded_model_yaml = yaml_file.read()
#yaml_file.close()
#loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(sample_weight=classes_weights,eval_metric=["merror", "mlogloss"])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))





#y_pred = model.predict(X_test) 
#print ("y_pred", y_pred)
#predictions = [round(value) for value in y_pred]
#accuracy = accuracy_score(y_test, predictions) 
#print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Fit model using each importance as a threshold
#thresholds = sort(model.feature_importances_)
#for thresh in thresholds:
    # select features using threshold
 #   selection = SelectFromModel(model, threshold=thresh, prefit=True)
   # select_X_train = selection.transform(X_train)
    # train model
    #selection_model = XGBClassifier()
    #selection_model.fit(select_X_train, y_train)
    # eval model
    #select_X_test = selection.transform(X_test)
    #predictions = selection_model.predict(select_X_test)
    #accuracy = accuracy_score(y_test, predictions)
    #print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))



#Classification Report
#print(classification_report(y_test,y_pred))

# plot feature importance

#plt.figure(1)
#plot_importance(model)
#name=('Feature Importance balanced weight removed features')
#plt.savefig('XGBoosted_output/{}'.format(name))



#cm_df = pd.DataFrame(cmn,index = ['ttbar','ggH','bbH','ZTT','misc'], columns = ['ttbar','ggH','bbH','ZTT','misc'])


#plt.figure(2)
#cm = confusion_matrix(y_test,y_pred)
#cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#cm_df = pd.DataFrame(cmn,
#                     index = ['ttbar','ggH','bbH','ZTT','misc'], 
#                     columns = ['ttbar','ggH','bbH','ZTT','misc'])

#sns.heatmap(cm_df, annot=True)
#plt.title('Confusion Matrix')
#plt.ylabel('Actual Values')
#plt.xlabel('Predicted Values')
#name=('confusion_matrix_XGBoosted_balanced_weight_OVERSAMPLE_SMOTE')
#plt.savefig('XGBoosted_output/{}'.format(name))


