# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:18:33 2019

@author: Benny  Yin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:39:54 2019

@author: Benny  Yin
"""

#from neupy import algorithms, environment, estimators
from sklearn.decomposition import PCA
from sklearn import datasets, metrics
from sklearn.preprocessing import normalize;
from neupy import algorithms, environment, estimators
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
from sklearn.metrics import roc_auc_score, fbeta_score

#readPath = 'C:/Life/Thesis/output2/crack_imagej_features2.csv' # crack 376vs469 old images
#readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized_836.csv'#376vs469 old images

#readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized_1177.csv'#(717+376)vs460 new IN images (with ex_time with x125)
#readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized2_1177.csv'#(717+376)vs460 new IN images (- ex_time with x125)
#readPath = 'C:/Life/Thesis/output2/crack_eosin_nucleo_cellPro_nuclei_normalized3_1177.csv'#(717+376)vs460 new IN images (60 features: - ex_time -x125 )

#readPath ='C:/Life/Thesis/output2/Nuclei_processed1210_1177.csv'#(717+376)vs460 new IN images nuclei features 1206 features (with nuclei features from cellPro)

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
#HE_pattern_raw, truth_raw = shuffle_Data(data_r)
HE_pattern_raw0, truth_raw = shuffle_Data(data_r)
X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw0, truth_raw, test_size=0.3)
scaler = StandardScaler()
scaler.fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test = scaler.transform(X_test_r)

X_train_n = normalize(X_train)
X_test_n = normalize(X_test)

#bagging on DT
clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_features = 1.0, max_samples=0.3)
clf.fit(X_train, y_train)
feature_importances = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
accuracy = []; fscore = []
for i in range(0,20):
    HE_pattern_raw0, truth_raw = shuffle_Data(data_r)
    X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw0, truth_raw, test_size=0.3)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train = scaler.transform(X_train_r)
    X_test = scaler.transform(X_test_r)
    
    clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_features = 1.0, max_samples=0.3)
    clf.fit(X_train, y_train)
    feature_importances += np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

len(feature_importances[:])#696
from scipy import stats
stats.describe(feature_importances[:])
stats.describe(feature_importances[np.where((feature_importances[:]>0.015)&(feature_importances[:]<0.05))[0]])

#len(list(np.where(feature_importances[:]==0.0)[0]))# around 27
len(list(np.where(feature_importances[:]>0.01)[0]))# around 270
len(list(np.where(feature_importances[:]>0.025)[0]))# around 100
len(list(np.where(feature_importances[:]>0.039)[0]))# around 70
696-28
topNidx= 660
goodFeatures_idx=feature_importances.argsort()[-topNidx:][::-1];len(goodFeatures_idx)
feature_names[goodFeatures_idx]
#feature
len(performance)

performance=feature_importances[goodFeatures_idx]
objects=feature_names[goodFeatures_idx]
y_pos = np.arange(len(objects))

performance=feature_importances[goodFeatures_idx]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects,rotation=90,)
plt.ylabel('Feature importance')
plt.subplots_adjust(bottom=0.72)
plt.rc('xtick', labelsize=12) 
plt.margins(x=0.01)
plt.show()

def findIncrease(acclist):
    increased_list=[]
    for idx in range(1,len(acclist)):
        increased_list.append(acclist[idx]-acclist[idx-1])
    return increased_list



#HE_pattern_raw=HE_pattern_raw[:,goodFeatures_idx]
HE_pattern_raw2=HE_pattern_raw0[:,goodFeatures_idx]
X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw, test_size=0.3)
scaler = StandardScaler()
scaler.fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test = scaler.transform(X_test_r)

X_train_n = normalize(X_train)
X_test_n = normalize(X_test)

from neupy import algorithms, environment, estimators

#PNN
pnn=algorithms.PNN()
pnn.fit(X_train_n, y_train)
predicted_y=pnn.predict(X_test_n)
pnnacc=metrics.accuracy_score(predicted_y, y_test)
print(pnnacc)
allacc.append(pnnacc)
predicted_y=pnn.predict(X_train_n)
#print(metrics.accuracy_score(predicted_y, y_train))

xacc=range(0,671)

topNidx= 670
#feature_importances=np.random.normal(size=topNidx)#random 
rdfAccMean=[];pnnAccMean=[];xacc=np.arange(0,topNidx+1);
rdfAccMean=[0];pnnAccMean=[0];svmAccMean=[0];baggingAccMean=[0];lrAccMean=[0];mlpAccMean=[0];
for topNidx in xacc[1:]:
    print(topNidx)    
    goodFeatures_idx=feature_importances.argsort()[-topNidx:][::-1]
    #goodFeatures_idx=feature_importances.argsort()[-110:][::-1]
    #goodFeatures_idx=goodPNNfeatureindex[:topNidx]
    
    pnnaccuracy = []; pnnfscore = []
    rdfaccuracy = []; rdffscore = []
    svmaccuracy = []; svmfscore = []
    baggingaccuracy = []; baggingfscore = []
    lraccuracy = []; lrscore = []
    mlpaccuracy = []; mlpfscore = []
    for i in range(0,40):
        HE_pattern_raw, truth_raw = shuffle_Data(data_r)
        HE_pattern_raw2=HE_pattern_raw[:,goodFeatures_idx]
                
        X_train_r, X_test_r, y_train, y_test = train_test_split(HE_pattern_raw2, truth_raw)
        scaler = StandardScaler()
        scaler.fit(X_train_r)
        X_train = scaler.transform(X_train_r)
        X_test = scaler.transform(X_test_r)
        X_train_n = normalize(X_train)
        X_test_n = normalize(X_test)
        
   #PNN
        pnn=algorithms.PNN()
        pnn.fit(X_train_n, y_train)
        predicted_y=pnn.predict(X_test_n)
        pnnaccuracy.append(metrics.accuracy_score(predicted_y, y_test))
        #pnnfscore.append(fbeta_score(y_test, predicted_y, average='binary', beta=0.5))
   #Rd
        clf = RandomForestClassifier(n_estimators=20,random_state=0)
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        rdfaccuracy.append(metrics.accuracy_score(predicted_y, y_test))        
        #rdffscore.append(fbeta_score(y_test, predicted_y, average='binary', beta=0.5))
   #svm
        clf = SVC(kernel='poly', degree=2)
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        svmaccuracy.append(metrics.accuracy_score(predicted_y, y_test))        
    #Bagging
        clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_features = 1.0, max_samples=0.5)
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        baggingaccuracy.append(metrics.accuracy_score(predicted_y, y_test))        
   #LR
        clf = LogisticRegression()
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        lraccuracy.append(metrics.accuracy_score(predicted_y, y_test))        
   #MLP
        clf = MLPClassifier(activation= 'tanh',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16, 23), random_state=1)
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        mlpaccuracy.append(metrics.accuracy_score(predicted_y, y_test))        
        
        
    meanacc=np.mean(pnnaccuracy)
    print('PNN accuracy: ',meanacc)
    pnnAccMean.append(meanacc)
    
    meanacc=np.mean(rdfaccuracy)
    print('random forest accuracy: ',meanacc)
    rdfAccMean.append(meanacc)
    
    meanacc=np.mean(baggingaccuracy)
    print('bagging accuracy: ',meanacc)
    baggingAccMean.append(meanacc)
    
    meanacc=np.mean(svmaccuracy)
    print('SVM accuracy: ',meanacc)
    svmAccMean.append(meanacc)
    
    meanacc=np.mean(lraccuracy)
    print('logistic regression accuracy: ',meanacc)
    lrAccMean.append(meanacc)
    
    meanacc=np.mean(mlpaccuracy)
    print('Multi-layer perceptron: ',meanacc)
    mlpAccMean.append(meanacc)
#save the output to csv so we don't need to run everything again. too time consuming~
classifiers_df= pd.DataFrame({'PNN accuracy':pnnAccMean[:],
                              'Random forest accuracy':rdfAccMean[:],
                              'bagging (decision tree) accuracy':baggingAccMean[:],
                              'Support vector machine accuracy':svmAccMean[:],
                              'Logistic regression accuracy':lrAccMean[:],
                              'Multi-layer perceptron accuracy':mlpAccMean[:]})
classifiers_df.to_csv('C:/Life/Thesis/output2/0_670featureXacc2.csv', index=False)





title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}#dictionary
plt.title('General classifier accuracy over number of features',**title_font)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.plot(xacc[:],pnnAccMean[:],label='PNN accuracy')
plt.plot(xacc[:],rdfAccMean[:],label='Random forest accuracy')
plt.plot(xacc[:],baggingAccMean[:],label='bagging (decision tree) accuracy')
plt.plot(xacc[:],svmAccMean[:],label='Support vector machine accuracy')
plt.plot(xacc[:],lrAccMean[:],label='Logistic regression accuracy')
plt.plot(xacc[:],mlpAccMean[:],label='Multi-layer perceptron accuracy')
plt.scatter(np.where(pnnAccMean==max(pnnAccMean)),max(pnnAccMean),s=10,marker='o',c='g')
plt.legend(loc="lower right")
plt.show()
np.where(pnnAccMean==max(pnnAccMean))[0]

increaselist=np.array(findIncrease(pnnAccMean))
goodPNNfeatureindex=increaselist.argsort()[-topNidx:][::-1]+1#0th element is 0~1



































rdfAccMean2=[];pnnAccMean2=[];xacc2=np.arange(100,300,5)
for topNidx in xacc2:
    print(topNidx)    
    goodFeatures_idx=feature_importances.argsort()[-topNidx:][::-1]
    #goodFeatures_idx=feature_importances.argsort()[-110:][::-1]
    pnnaccuracy = []; pnnfscore = []
    rdfaccuracy = []; rdffscore = []
    for i in range(0,40):
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
        predicted_y=pnn.predict(X_test_n)
        pnnfscore.append(fbeta_score(y_test, predicted_y, average='binary', beta=0.5))
        pnnaccuracy.append(metrics.accuracy_score(predicted_y, y_test))
    
        clf = RandomForestClassifier(n_estimators=20,random_state=0)
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        rdffscore.append(fbeta_score(y_test, predicted_y, average='binary', beta=0.5))
        rdfaccuracy.append(metrics.accuracy_score(predicted_y, y_test))
    print('PNN accuracy: ',np.mean(pnnaccuracy))
    pnnAccMean2.append(np.mean(pnnaccuracy))
    print('random forest accuracy: ',np.mean(rdfaccuracy))
    rdfAccMean2.append(np.mean(rdfaccuracy))

xacc3=np.append(xacc,xacc2);rdfAccMean3=np.append(rdfAccMean,rdfAccMean2);pnnAccMean3=np.append(pnnAccMean,pnnAccMean2)
#xacc3=np.insert(xacc3,0,0);rdfAccMean3=np.insert(rdfAccMean3,0,0);pnnAccMean3=np.insert(pnnAccMean3,0,0)

#plot with highest points
plt.plot(xacc3,pnnAccMean3)
plt.plot(xacc3,rdfAccMean3)
plt.scatter(np.where(pnnAccMean3==max(pnnAccMean3[50:100])),max(pnnAccMean3[50:100]),s=12,marker='o',c='g')
feature_idx=(np.where(pnnAccMean3==max(pnnAccMean3))[0]-100)*5+100
plt.scatter(feature_idx,max(pnnAccMean3),s=12,marker='o',c='r')
plt.yticks(np.arange(0,1.05,0.05))
plt.show()

rdfAccMean4=[];pnnAccMean4=[];xacc4=np.arange(300,660,10)
for topNidx in xacc4:
    print(topNidx)    
    goodFeatures_idx=feature_importances.argsort()[-topNidx:][::-1]
    #goodFeatures_idx=feature_importances.argsort()[-110:][::-1]
    pnnaccuracy = []; pnnfscore = []
    rdfaccuracy = []; rdffscore = []
    for i in range(0,20):
        HE_pattern_raw, truth_raw = shuffle_Data()
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
        predicted_y=pnn.predict(X_test_n)
        pnnfscore.append(fbeta_score(y_test, predicted_y, average='binary', beta=0.5))
        pnnaccuracy.append(metrics.accuracy_score(predicted_y, y_test))
    
        clf = RandomForestClassifier(n_estimators=20,random_state=0)
        predicted_y = clf.fit(X_train, y_train).predict(X_test)
        rdffscore.append(fbeta_score(y_test, predicted_y, average='binary', beta=0.5))
        rdfaccuracy.append(metrics.accuracy_score(predicted_y, y_test))
    print('PNN accuracy: ',np.mean(pnnaccuracy))
    pnnAccMean4.append(np.mean(pnnaccuracy))
    print('random forest accuracy: ',np.mean(rdfaccuracy))
    rdfAccMean4.append(np.mean(rdfaccuracy))

xacc5=np.append(xacc3,xacc4);rdfAccMean5=np.append(rdfAccMean3,rdfAccMean4);pnnAccMean5=np.append(pnnAccMean3,pnnAccMean4)

plt.plot(xacc5,pnnAccMean5)
plt.plot(xacc5,rdfAccMean5)
plt.scatter(np.where(pnnAccMean3==max(pnnAccMean3[50:100])),max(pnnAccMean3[50:100]),s=10,marker='o',c='g')
feature_idx=(np.where(pnnAccMean3==max(pnnAccMean3))[0]-100)*5+100
plt.scatter(feature_idx,max(pnnAccMean3),s=12,marker='o',c='r')
plt.yticks(np.arange(0,1.05,0.05))
plt.show()


plt.plot(xacc3[0:2],pnnAccMean3[0:2])
plt.plot(xacc3[0:2],rdfAccMean3[0:2])
plt.ylabel('some numbers')
plt.show()

#X20#
#=========================================#

#feature selection from bagging
#=============================================================================#
    