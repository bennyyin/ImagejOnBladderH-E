# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:07:43 2018

@author: Benny  Yin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:19:30 2018

@author: Benny  Yin
"""
from neupy import algorithms, environment, estimators
from sklearn.decomposition import PCA
from sklearn import datasets, metrics
from sklearn.preprocessing import normalize;
from scipy import interp
import random
import neupy 
import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, auc

#readPath = 'C:/Life/Thesis/output2/crack_imagej_features2.csv' # crack 376vs469 old images
#readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized_836.csv'#376vs469 old images

readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized_1177.csv'#(717+376)vs460 new IN images (with ex_time with x125)
readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized2_1177.csv'#(717+376)vs460 new IN images (- ex_time with x125)
readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized3_1177.csv'#(717+376)vs460 new IN images (60 features: - ex_time -x125 )

readPath ='C:/Life/Thesis/output2/Nuclei_processed1210_1177.csv'#(717+376)vs460 new IN images nuclei features 1206 features (with nuclei features from cellPro)
readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized700_1177.csv'#(717+376)vs460 new IN images 696 features (with CNT features from cellPro)
df=pd.read_csv(readPath);#csv-> dataframe
feature_names=np.array(list(df)[5:])#getting colnames
#feature_names=np.array(list(df))#getting colnames
data_r=df.values#numpy object
data_r=np.array(data_r)
row=[x for x in range(data_r.shape[0])] 
def shuffle_Data(data_r):    
    data=data_r[[a for a in row if a not in random.sample(range(717), 250)],:] #excluding 250 cases from invasive
    #truth=data[:,1]#type: nin vs in #arrayObeject from numpy
    truth=data[:,2]#type: nin vs in #arrayObeject from numpy
    truth_raw=truth.astype('str')#### Y
    col=[x for x in range(data.shape[1])] 
    HE_pattern_raw=data[:,[a for a in col if a not in [0,1,2,3,4]]]#excluding unwanted col ex: row_number, type, name
    HE_pattern_raw=HE_pattern_raw.astype('float32')#for DT
    HE_pattern_raw=np.nan_to_num(HE_pattern_raw, copy=True)### X
    truth_raw=truth_raw=="nin"
    truth_raw=truth_raw.astype(int)
    return HE_pattern_raw, truth_raw


HE_pattern_raw, truth_raw = shuffle_Data(data_r)

X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw, truth_raw)
scaler = StandardScaler()
scaler.fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test = scaler.transform(X_test_r)

X_train_n = normalize(X_train)
X_test_n = normalize(X_test)

#feature selection from random forest
clf = RandomForestClassifier(n_estimators=20,random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))#all features
goodFeatures_idx=list(np.where(clf.feature_importances_[:]>0.001)[0])
goodFeatures_idx=list(np.where(clf.feature_importances_[:]>0)[0])

clf.fit(X_train[:,goodFeatures_idx], y_train)
print(clf.score(X_test[:,goodFeatures_idx], y_test))#good features
clf.fit(X_train_r, y_train)

#feature selection from bagging
clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_features = 1.0, max_samples=0.3)
clf.fit(X_train, y_train)
feature_importances = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
accuracy = []; fscore = []
for i in range(1,20):
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)
    
    clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_features = 1.0, max_samples=0.3)
    clf.fit(X_train, y_train)
    feature_importances += np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

goodFeatures_idx=list(np.where(feature_importances[:]>0.01)[0])
len(goodFeatures_idx)




#PNN
pnn=algorithms.PNN()
pnn.fit(X_train_n, y_train)
predicted_y=pnn.predict(X_test_n)
print(metrics.accuracy_score(predicted_y, y_test))
predicted_y=pnn.predict(X_train_n)
print(metrics.accuracy_score(predicted_y, y_train))



def printing_list_row(ls):
    for i in ls:
        print(i)

for batch in range(1,10):
    pnn=algorithms.PNN(std=1/batch)
    pnn.fit(X_train_n, y_train)
    predicted_y=pnn.predict(X_test_n)
    print(1/batch)
    print(metrics.accuracy_score(predicted_y, y_test))
#SVM
clf = SVC(kernel='poly', degree=2)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
X_test.shape
X_test[:,goodFeatures_idx].shape
clf.fit(X_train[:,goodFeatures_idx], y_train)
print(clf.score(X_test[:,goodFeatures_idx], y_test))
#DT
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

#ADA boost
clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100,learning_rate=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Bagging
clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_features = 1.0, max_samples=0.5)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

#LR
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
clf.fit(X_train_n, y_train)
print(clf.score(X_test_n, y_test))

#voting
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=100,random_state=0)
clf3 = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_features = 1.0, max_samples=0.5)

evc=VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
evc.fit(X_train, y_train)
print(evc.score(X_test, y_test))

#MLP deep learning:ANN
clf = MLPClassifier(activation= 'tanh',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16, 6), random_state=1)            
clf = MLPClassifier(activation= 'tanh',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16, 23), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
#print(clf.score(X_train, y_train))



#X20#
#=========================================#
def draw_one_Avg_ROC_curve(tprs,aucs,mlModel,score):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #mean curve
    #plt.plot(mean_fpr, mean_tpr, color=color, label=r''+mlModel+' Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=r''+mlModel+' Mean ROC (AUC = %0.2f $\pm$ %0.2f) accuracy: %0.2f' % (mean_auc, std_auc, score))

#=============================================================================#
def drawAvg_ROC_curve(tprs,aucs,mlModel):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #mean curve
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #sd range
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')    
    #plot axises
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic '+mlModel)
    plt.legend(loc="lower right")



#feature selection from random forest
isAllone=True;score=0
tprsX20 = [];aucsX20 = []
#PNN
tprs = [];aucs = [];mean_fpr = np.linspace(0, 1, 100);score=0
for i in range(0,20):
    HE_pattern_raw, truth_raw = shuffle_Data(data_r)
    HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
    
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)

    X_train_n = normalize(X_train)
    X_test_n = normalize(X_test)
    pnn=algorithms.PNN()
    pnn.fit(X_train_n, y_train)
    #predicted_y=pnn.predict(X_test_n)
    probas_=pnn.predict_proba(X_test_n)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    predicted_y=pnn.predict(X_test_n)
    score+=metrics.accuracy_score(predicted_y, y_test)
    print(metrics.accuracy_score(predicted_y, y_test))
    if not isAllone:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
if isAllone:
    draw_one_Avg_ROC_curve(tprs,aucs,"'PNN'",score/20)
else:    
    drawAvg_ROC_curve(tprs,aucs,"'PNN'")


#SVM
tprs = [];aucs = [];mean_fpr = np.linspace(0, 1, 100);score=0
for i in range(0,20):
    HE_pattern_raw, truth_raw = shuffle_Data(data_r)
    HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
    
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)

    clf = SVC(kernel='poly', degree=2,probability=True,)
    clf.fit(X_train, y_train)
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    score+=clf.score(X_test, y_test)
    if not isAllone:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
if isAllone:
    draw_one_Avg_ROC_curve(tprs,aucs,"'SVM'",score/20);tprsX20=tprsX20+tprs
else:
    drawAvg_ROC_curve(tprs,aucs,"'SVM'") 
  

# Bagging
tprs = [];aucs = [];mean_fpr = np.linspace(0, 1, 100);score=0
for i in range(0,20):
    HE_pattern_raw, truth_raw = shuffle_Data(data_r)
    HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
    
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)
    
    clf = BaggingClassifier(AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100),n_estimators=20,max_features = 1.0, max_samples=0.5)
    clf.fit(X_train, y_train)
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    score+=clf.score(X_test, y_test)
    if not isAllone:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
if isAllone:
    draw_one_Avg_ROC_curve(tprs,aucs,"'Bagging(Adaboost)'",score/20);tprsX20=tprsX20+tprs
else:
    drawAvg_ROC_curve(tprs,aucs,"'Bagging(Adaboost)'")

#RDF
tprs = [];aucs = [];mean_fpr = np.linspace(0, 1, 100);score=0
for i in range(0,20):
    HE_pattern_raw, truth_raw = shuffle_Data(data_r)
    HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
    
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)

    clf = RandomForestClassifier(n_estimators=20,random_state=0)
    probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    if not isAllone:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    score+=clf.score(X_test, y_test)
if isAllone:
    draw_one_Avg_ROC_curve(tprs,aucs,"'Random Forest'",score/20);    
else:
    drawAvg_ROC_curve(tprs,aucs,"'Random Forest'");

#LR
tprs = [];aucs = [];mean_fpr = np.linspace(0, 1, 100);score=0
for i in range(0,20):
    HE_pattern_raw, truth_raw = shuffle_Data(data_r)
    HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
    
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    score+=clf.score(X_test, y_test)
    if not isAllone:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
if isAllone:
    draw_one_Avg_ROC_curve(tprs,aucs,"'Logistic Regression'",score/20);tprsX20=tprsX20+tprs
else:
    drawAvg_ROC_curve(tprs,aucs,"'Logistic Regression'")

#MLP deep learning:ANN
tprs = [];aucs = [];mean_fpr = np.linspace(0, 1, 100);score=0
for i in range(0,20):
    HE_pattern_raw, truth_raw = shuffle_Data(data_r)
    HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
    
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)
    
    X_train_n = normalize(X_train)
    X_test_n = normalize(X_test)

    
    #clf = MLPClassifier(activation= 'tanh',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16, 6), random_state=1)            
    clf = MLPClassifier(activation= 'tanh',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16, 23), random_state=1)
    #clf.fit(X_train, y_train)
    #print(clf.score(X_test, y_test))
    #print(clf.score(X_train, y_train))
    clf.fit(X_train, y_train)
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    score+=clf.score(X_test, y_test)

    
    predicted_y = clf.predict(X_test)
    score2=metrics.accuracy_score(predicted_y, y_test)
    if not isAllone:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
if isAllone:
    draw_one_Avg_ROC_curve(tprs,aucs,"'MLP'",score/20);tprsX20=tprsX20+tprs
else:
    drawAvg_ROC_curve(tprs,aucs,"'MLP'")




    

#=============================================================================#
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
mean_tpr = np.mean(tprsX20, axis=0)#all ML mean
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)#all ML standard deviation
#mean curve
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Total Mean ROC from all ML classifiers (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
std_tpr = np.std(tprsX20, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#sd range
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')    
#plot axises
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.tick_params(axis='both', which='major', labelsize=26)
plt.xlabel('False Positive Rate',fontsize=28)
plt.ylabel('True Positive Rate',fontsize=28)
title_font = {'fontname':'Arial', 'size':'30', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}#dictionary
#plt.title('Receiver Operating Characteristic \n(ALL ML classifiers with all 696 features)',**title_font)
#plt.title('Receiver Operating Characteristic \n(ALL ML classifiers with top 100 features)',**title_font)
#plt.title('Receiver Operating Characteristic \n(ALL ML classifiers with artificial retraction features)',**title_font)

plt.title('Receiver Operating Characteristic \n(ALL ML classifiers with desmoplastic reaction features)',**title_font)
plt.title('Receiver Operating Characteristic \n(ALL ML classifiers with cytoplasmic features)',**title_font)
#plt.title('Receiver Operating Characteristic \n(ALL ML classifiers)',**title_font)
plt.legend(loc="lower right",fontsize = 14)
plt.show()


#======================================================================#
    #voting
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=100,random_state=0)
clf3 = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_features = 1.0, max_samples=0.5)
evc=VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
evc.fit(X_train, y_train)
print(evc.score(X_test, y_test))


