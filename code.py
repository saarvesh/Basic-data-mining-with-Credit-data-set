
# coding: utf-8

# In[107]:

import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')


# In[4]:

train = pd.read_csv('C:/Users/ssssa/Desktop/Graman_credit/german_credit.csv')


# In[108]:

# Data Exploration
import matplotlib.pylab as plt
#plotting histogram
train.hist()
plt.show()


# In[110]:


train.head()   #Displaying the first 10 rows of data """


# In[111]:


train.tail()


# In[112]:

#Statistical analysis of the Mean median mode
train.describe() 


# In[113]:


train.dtypes



# In[114]:

train.info()



# In[115]:

train.ix[: ,0:5].boxplot()


# In[120]:

train.ix[:,0:1].plot()


# In[129]:

train.hist(bins=50)


# In[21]:




# In[22]:

plt.show()


# In[140]:

train.ix[: ,0:2].plot()


# In[145]:

train.ix[: ,1:4].plot()


# In[152]:

train.ix[: ,6:10].hist()


# In[147]:

train.ix[: ,7:10].plot()


# In[148]:

train.ix[: ,4:5].plot()


# In[151]:

train.ix[: ,1:5].hist()


# In[153]:

train.ix[: ,4:7].plot()


# In[155]:

train.ix[: ,15:20].hist()


# In[27]:

# Data Preprocessing


# In[ ]:


from sklearn import preprocessing


# In[30]:

#Standardization of the data
X_scaled = preprocessing.scale(train)
np.savetxt("C:/Users/ssssa/Desktop/standardized.txt",X_scaled, delimiter=",")


# In[29]:

from sklearn.preprocessing import LabelEncoder


# In[134]:

# Label encoding
le=LabelEncoder()
train = pd.read_csv('C:/Users/ssssa/Desktop/Graman_credit/german_credit.csv')
train.head(5)    


# In[33]:

le.fit(train.values.ravel())
a= le.transform(train.values.ravel())


# In[138]:

df=np.savetxt("C:/Users/ssssa/Desktop/Encoded.csv",a, delimiter=",")
a.head()


# In[37]:

# Standardization of the data
train=pd.read_csv("C:/Users/ssssa/Desktop/Encoded.csv")



# In[157]:

std_scale = preprocessing.StandardScaler().fit(train)
df_std = std_scale.transform(train)
b= np.asarray(df_std)
np.savetxt("C:/Users/ssssa/Desktop/German_Encoded_Standard.csv",b, delimiter=",")


# In[161]:

#Imputation Putting the mean values in the missing column
from sklearn.preprocessing import Imputer
a=pd.read_csv("C:/Users/ssssa/Desktop/Encoded.csv")
a.tail()
a


# In[160]:

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(a)
imp


# In[47]:

#Min Max normalization
min_max_scaler = preprocessing.MaxAbsScaler()


# In[48]:

X_train_minmax = min_max_scaler.fit_transform(a)


# In[50]:

b= np.asarray(df_std)
np.savetxt("C:/Users/ssssa/Desktop/German_Encoded_Normalized.csv",b, delimiter=",")


# In[204]:

# Principal Component Analysis on  Normalized

a=pd.read_csv("C:/Users/ssssa/Desktop/Graman_credit/Normalized/german_N.csv")
from sklearn.decomposition import PCA


# In[164]:

pca = PCA().fit(train)
X_train = pca.transform(train)
np.savetxt("C:/Users/ssssa/Desktop/German_Encoded_Normalized_PCA.csv",b, delimiter=",")


# In[205]:


mean_vec = np.mean(a, axis=0)
cov_mat = (a - mean_vec).T.dot((a - mean_vec)) / (a.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)



# In[206]:

print('NumPy covariance matrix: \n%s' %np.cov(a.T))


# In[207]:

cov_mat = np.cov(a.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)


# In[208]:

print('\nEigenvalues \n%s' %eig_vals)


# In[215]:

# Make a list of (eigenvalue, eigenvector) tuples


# In[ ]:

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]


# In[211]:

# Sort the (eigenvalue, eigenvector) tuples from high to low


# In[ ]:

eig_pairs.sort(key=lambda x: x[0], reverse=True)


# In[216]:

# Visually confirm that the list is correctly sorted by decreasing eigenvalues


# In[217]:

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[213]:

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[214]:

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(21, 21))
    plt.bar(range(21), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(21), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:

# Principal Component Analysis on standardized data


# In[218]:

b=pd.read_csv("C:/Users/ssssa/Desktop/Graman_credit/german_S.csv")


# In[60]:

pca = PCA().fit(b)


# In[219]:

mean_vec = np.mean(a, axis=0)
cov_mat = (a - mean_vec).T.dot((a - mean_vec)) / (a.shape[0]-1)
print('NumPy covariance matrix: \n%s' %np.cov(a.T))


# In[220]:

cov_mat = np.cov(a.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# In[221]:

# Make a list of (eigenvalue, eigenvector) tuples


# In[222]:

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]


# In[223]:

# Sort the (eigenvalue, eigenvector) tuples from high to low


# In[ ]:

eig_pairs.sort(key=lambda x: x[0], reverse=True)


# In[224]:

# Visually confirm that the list is correctly sorted by decreasing eigenvalues


# In[226]:

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[227]:

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[228]:

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(21, 21))
    plt.bar(range(21), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(21), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[63]:

#Implemntign the algorithms 
#1. Logistic Regression


# In[ ]:

from sklearn.linear_model import LogisticRegression


# In[64]:

inputtrain=pd.read_csv("C:/Users/ssssa/Desktop/Graman_credit/Normalized/inputtrain.csv")


# In[65]:

trainoutput = pd.read_csv("C:/Users/ssssa/Desktop/Graman_credit/Normalized/trainoutput.csv")
testinput=pd.read_csv("C:/Users/ssssa/Desktop/Graman_credit/Normalized/testinput.csv")
testoutput=pd.read_csv("C:/Users/ssssa/Desktop/Graman_credit/Normalized/testoutput.csv")


# In[66]:

aa = LogisticRegression()


# In[67]:

aa.fit(inputtrain,trainoutput)


# In[69]:

coef_l1_LR = aa.coef_.ravel()


# In[72]:

aa.class_weight


# In[74]:

p=aa.coef_


# In[75]:

from sklearn.metrics import accuracy_score


# In[76]:

y_pred= aa.predict(testinput)


# In[77]:

logpredict=aa.predict_log_proba(testinput)


# In[78]:

accuracy_score(y_pred, testoutput)


# In[84]:

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


# In[83]:

f1_score(y_pred, testoutput)


# In[85]:

classification_report(y_pred, testoutput)


# In[86]:

precision_score(y_pred, testoutput)


# In[87]:

recall_score(y_pred, testoutput)


# In[229]:

aa.coef_


# In[231]:

y_pred


# In[ ]:




# In[88]:

#2. SVM


# In[89]:

from sklearn import svm


# In[90]:

clf = svm.SVC()


# In[91]:

clf.fit(inputtrain,trainoutput) 


# In[92]:

y_pred=clf.predict(testinput)


# In[93]:

y_pred


# In[94]:

clf.support_vectors_


# In[95]:

clf.support_


# In[96]:

clf.class_weight


# In[98]:

clf.classes_


# In[99]:

clf.decision_function_shape


# In[100]:

clf.dual_coef_


# In[101]:

clf.fit_status_


# In[102]:

clf.intercept_


# In[103]:

clf.n_support_


# In[106]:

clf.support_


# In[232]:

clf.degree


# In[233]:

clf.coef0


# In[236]:

clf.decision_function


# In[237]:

#3.Naive Bayes


# In[238]:

from sklearn.naive_bayes import GaussianNB


# In[239]:

gnb = GaussianNB()


# In[241]:

clf= gnb.fit(inputtrain,trainoutput)


# In[242]:

y_pred= clf.predict(testinput)


# In[243]:

y_pred


# In[244]:

clf.score


# In[245]:

clf.sigma_


# In[246]:

clf.theta_


# In[247]:

clf.class_count_


# In[248]:

clf.class_prior_


# In[249]:

clf.get_params


# In[250]:


accuracy_score(y_pred, testoutput)


# In[251]:

f1_score(y_pred, testoutput)


# In[ ]:




# In[252]:

classification_report(y_pred, testoutput)


# In[253]:

precision_score(y_pred, testoutput)


# In[254]:

recall_score(y_pred, testoutput)


# In[ ]:




# In[255]:

#4. K -Nearest Neighbour


# In[266]:

from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors


# In[268]:

clf = neighbors.KNeighborsClassifier()


# In[269]:

clf


# In[270]:

clf.algorithm


# In[271]:

clf.fit(inputtrain,trainoutput)


# In[273]:

y_predict= clf.predict(testinput)


# In[274]:

y_predict


# In[275]:

clf.effective_metric_


# In[276]:

clf.kneighbors_graph


# In[277]:

clf.kneighbors_graph.im_class


# In[278]:

accuracy_score(y_pred, testoutput)


# In[279]:

f1_score(y_pred, testoutput)


# In[280]:

classification_report(y_pred, testoutput)


# In[281]:

precision_score(y_pred, testoutput)


# In[282]:

recall_score(y_pred, testoutput)


# In[ ]:




# In[ ]:

# SGD 


# In[283]:

from sklearn.linear_model import SGDClassifier


# In[284]:

clf = SGDClassifier()


# In[285]:


clf.fit(inputtrain,trainoutput)


# In[286]:

clf.C


# In[287]:

clf.class_weight


# In[288]:

clf.classes_


# In[289]:

clf.coef_


# In[290]:

clf.decision_function


# In[291]:

clf.loss_function


# In[292]:

clf.verbose


# In[293]:

clf.score


# In[294]:

clf.learning_rate


# In[295]:

clf.densify


# In[296]:

accuracy_score(y_pred, testoutput)


# In[297]:

f1_score(y_pred, testoutput)


# In[298]:

classification_report(y_pred, testoutput)


# In[299]:

precision_score(y_pred, testoutput)


# In[300]:

recall_score(y_pred, testoutput)


# In[ ]:




# In[301]:

# TREES CART Modelling


# In[302]:

from sklearn import tree


# In[303]:

clf = tree.DecisionTreeClassifier()


# In[304]:


clf.fit(inputtrain,trainoutput)


# In[305]:

clf.class_weight


# In[306]:

clf.classes_


# In[307]:

clf.feature_importances_


# In[308]:

clf.fit_transform


# In[309]:

clf.max_depth


# In[310]:

y_predict=clf.predict(testinput)


# In[311]:

accuracy_score(y_pred, testoutput)


# In[312]:

f1_score(y_pred, testoutput)


# In[313]:

clf.min_weight_fraction_leaf


# In[314]:

classification_report(y_pred, testoutput)


# In[315]:

precision_score(y_pred, testoutput)


# In[316]:

recall_score(y_pred, testoutput)


# In[317]:

from sklearn.ensemble import RandomForestClassifier


# In[318]:

clf = DecisionTreeClassifier()


# In[319]:

clf = RandomForestClassifier()


# In[321]:

clf.fit(inputtrain,trainoutput)


# In[322]:

clf.apply


# In[323]:

clf.base_estimator


# In[324]:

clf.bootstrap


# In[325]:

clf.classes_


# In[326]:

clf.min_weight_fraction_leaf


# In[327]:

clf.n_classes_


# In[328]:

y_pred= clf.predict(testinput)


# In[329]:

accuracy_score(y_pred, testoutput)


# In[330]:

f1_score(y_pred, testoutput)


# In[331]:

classification_report(y_pred, testoutput)


# In[332]:

precision_score(y_pred, testoutput)


# In[333]:

recall_score(y_pred, testoutput)


# In[ ]:




# In[334]:

#Next phase is for validation the alogorthim to check for Over fitting


# In[335]:

# Cross Validation


# In[338]:

from sklearn.cross_validation import cross_val_score


# In[339]:

clf = svm.SVC()


# In[340]:

# for K=5


# In[344]:


asd= trainoutput
c= np.asarray(asd)
asd=c.ravel()


# In[346]:

scores = cross_val_score(clf, inputtrain, asd, cv=5)


# In[347]:

scores.all


# In[348]:

scores


# In[349]:

scores.max


# In[351]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[352]:

# for K=10


# In[353]:

scores = cross_val_score(clf, inputtrain, asd, cv=10)


# In[354]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[355]:

#Logistic regression


# In[357]:

clf= LogisticRegression()


# In[358]:

scores = cross_val_score(clf, inputtrain, asd, cv=5)


# In[359]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[360]:

scores


# In[362]:

#for k=10


# In[363]:

scores = cross_val_score(clf, inputtrain, asd, cv=10)


# In[364]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[365]:

scores


# In[367]:

#Trees


# In[368]:

clf = tree.DecisionTreeClassifier()


# In[369]:

#k=10


# In[370]:

scores = cross_val_score(clf, inputtrain, asd, cv=10)


# In[371]:

scores


# In[372]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[373]:

# K=5


# In[374]:

scores = cross_val_score(clf, inputtrain, asd, cv=5)


# In[375]:

scores


# In[376]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



