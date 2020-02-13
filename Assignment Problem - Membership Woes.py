#!/usr/bin/env python
# coding: utf-8

# ### Importing all the required libraries

# In[857]:


import numpy as np
import pandas as pd
import itertools
import warnings
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, classification_report


# In[828]:


warnings.filterwarnings(action='ignore')


# ### Reading the given excel file and getting the sheets it has

# In[755]:


excel_file = pd.ExcelFile('Assignment- Membership woes.xlsx')
excel_file.sheet_names  # see all sheet names


# ### Read and manipulated the problem statement sheet to get the whole problem statement here

# In[756]:


prob_state = pd.read_excel("Assignment- Membership woes.xlsx", sheet_name='Problem statement')
prob_state.head(3)


# In[757]:


prob_state['Unnamed: 1'].fillna(value = '', inplace=True)
prob_state['Unnamed: 0'].fillna(value = '', inplace=True)
prob_state["prbState"] = prob_state["Unnamed: 1"] + prob_state["Unnamed: 0"]
for i in prob_state['prbState']:
    print(i)


# ### Reading the 'data' sheet

# In[758]:


com_data = pd.read_excel("Assignment- Membership woes.xlsx", sheet_name='Data')
com_data.head(3)


# ### Getting column names

# In[759]:


com_data.columns


# ### Description about the data (describe() brings out a lot of information from the dataframe)

# In[760]:


com_data.info()


# In[761]:


com_data.describe()


# ### Collecting metadata about the data at hand

# In[966]:


print(len(com_data))
print(len(com_data['END_DATE  (YYYYMMDD)'].unique()))
print(len(com_data['START_DATE (YYYYMMDD)'].unique()))
print(com_data['MEMBERSHIP_STATUS'].unique()) 
print() 
print(com_data['MEMBER_GENDER'].unique()) 
print()
print(com_data['PAYMENT_MODE'].unique()) 
print()
print(com_data[com_data['MEMBERSHIP_STATUS']=='INFORCE'].count())


# In[968]:


# Check the proportion of data belonging to each of the classes.
print(com_data['MEMBERSHIP_STATUS'].value_counts(normalize=True)) 


# In[763]:


print(com_data['END_DATE  (YYYYMMDD)'].isnull().sum())
print(com_data['START_DATE (YYYYMMDD)'].isnull().sum())
# Number of nan(None/Null) values in the column 'END_DATE' is too much for it to be taken into account


# In[905]:


# There is a large number of agents that lead the users. There count is 4317 for a whole ~10000 different users. 
# Hence this feature will be of no use in our prediction.
len(onehot_req_com_data['AGENT_CODE'].unique())


# In[906]:


# The membership number of a particular user won't we of any use in predicting he's cancelling the subscription or not.
# And then we have already seen that their are too many null values in the END_DATE column for it to be used.

req_com_data = com_data.drop(['MEMBERSHIP_NUMBER', 'END_DATE  (YYYYMMDD)', 'AGENT_CODE'],  axis=1)
req_com_data.head(3)


# In[907]:


### Nearly ~16% of values in column 'MEMBER_ANNUAL_INCOME' are null, so decided to fill those with mean of column values

print(req_com_data['MEMBER_ANNUAL_INCOME'].isna().sum())
print(req_com_data['MEMBER_ANNUAL_INCOME'].mean())


# In[908]:


req_com_data['MEMBER_ANNUAL_INCOME'].fillna(req_com_data['MEMBER_ANNUAL_INCOME'].mean(), inplace=True)


# In[909]:


for i in req_com_data['MEMBER_OCCUPATION_CD'].unique():
    print(i)


# In[910]:


# The null values in the respective columns can be filled with Other/NA etc.

req_com_data['MEMBER_MARITAL_STATUS'].fillna(value = 'Other', inplace=True)
req_com_data['MEMBERSHIP_STATUS'].fillna(value = 'Other', inplace=True)
req_com_data['MEMBERSHIP_PACKAGE'].fillna(value = 'Other', inplace=True)
req_com_data['PAYMENT_MODE'].fillna(value = 'Other', inplace=True)
req_com_data['MEMBER_GENDER'].fillna(value = 'Other', inplace=True)
req_com_data['MEMBER_OCCUPATION_CD'].fillna(value = 7.0, inplace=True)


# In[911]:


# From the date column, we can have the month as it may be a factor
req_com_data['START_DATE (YYYYMMDD)'] = pd.DatetimeIndex(req_com_data['START_DATE (YYYYMMDD)']).month


# ### Applying one-hot encoding on 'MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'MEMBERSHIP_PACKAGE', 'PAYMENT_MODE', 'MEMBERSHIP_STATUS', as they are columns with categorical values

# In[912]:


onehot_req_com_data = pd.get_dummies(req_com_data, columns=
     ['MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'MEMBERSHIP_PACKAGE', 'PAYMENT_MODE', 'MEMBERSHIP_STATUS'], drop_first=True)
onehot_req_com_data.head(3)


# In[915]:


# All of the columns in our dataframe, ready to be worked on.
print(onehot_req_com_data.columns)


# ### Exploratory analysis

# In[916]:


# Plotting the scatter plot and box plot for a few features that we got as important features from the algorithms. 

sns.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_OCCUPATION_CD", data=onehot_req_com_data)
sns.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_OCCUPATION_CD", data=onehot_req_com_data)


# In[971]:


sns.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_AGE_AT_ISSUE", data=onehot_req_com_data)
sns.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_AGE_AT_ISSUE", data=onehot_req_com_data)


# In[917]:


sns.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBERSHIP_TERM_YEARS", data=onehot_req_com_data)
sns.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBERSHIP_TERM_YEARS", data=onehot_req_com_data)


# In[918]:


sns.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_ANNUAL_INCOME", data=onehot_req_com_data)
sns.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="MEMBER_ANNUAL_INCOME", data=onehot_req_com_data)


# In[919]:


sns.catplot(x="MEMBERSHIP_STATUS_INFORCE", y="ANNUAL_FEES", data=onehot_req_com_data)
sns.boxplot(x="MEMBERSHIP_STATUS_INFORCE", y="ANNUAL_FEES", data=onehot_req_com_data)


# ### We infer from the above plots that there is too much of overlapping among the data from both the classes and hence, much can't be said about the cruciality of a feature. And this might be the reason for low accuracy of our models. 

# In[921]:


# Separating data into two parts, X is independent variables (features) and y is our target variable
data_copy = onehot_req_com_data.copy()
X = onehot_req_com_data.drop(['MEMBERSHIP_STATUS_INFORCE'], axis=1)
y = onehot_req_com_data.drop(['MEMBERSHIP_TERM_YEARS', 'ANNUAL_FEES', 'MEMBER_ANNUAL_INCOME',
                        'MEMBER_OCCUPATION_CD', 'MEMBER_AGE_AT_ISSUE', 'ADDITIONAL_MEMBERS',
                        'START_DATE (YYYYMMDD)', 'MEMBER_MARITAL_STATUS_M',
                        'MEMBER_MARITAL_STATUS_Other', 'MEMBER_MARITAL_STATUS_S',
                        'MEMBER_MARITAL_STATUS_W', 'MEMBER_GENDER_M', 'MEMBER_GENDER_Other',
                        'MEMBERSHIP_PACKAGE_TYPE-B', 'PAYMENT_MODE_MONTHLY',
                        'PAYMENT_MODE_QUARTERLY', 'PAYMENT_MODE_SEMI-ANNUAL',
                        'PAYMENT_MODE_SINGLE-PREMIUM'], axis=1)


# In[922]:


#apply SelectKBest class to extract top 10 best features using chi2 as score_func
bestfeatures = SelectKBest(score_func=chi2, k=5) 
fit = bestfeatures.fit(X,y.to_numpy())
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']
# Printing the feature names and the scores for the top 10 
print(featureScores.nlargest(5,'Score')) 


# In[923]:


#apply SelectKBest class to extract top 10 best features using f_classif as score_func
bestfeatures = SelectKBest(score_func=f_classif, k=5) 
fit = bestfeatures.fit(X,y.to_numpy())
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  
# Printing the feature names and the scores for the top 10 
print(featureScores.nlargest(5,'Score')) 


# In[924]:


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[925]:


#get correlations of each features in dataset
corrmat = onehot_req_com_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(onehot_req_com_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ### Applying logistic regression and getting the coefficients

# In[970]:


model = LogisticRegression(C=10**2)
ytest = ytest.to_numpy()
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.10, random_state=0, stratify=y) 
model.fit(xtrain, ytrain)
predicted_classes = model.predict(xtest)
accuracy = accuracy_score(ytest.to_numpy().flatten(), predicted_classes)
parameters = model.coef_
print("Accuracy: ", accuracy)
print("Parameters: ", parameters) # printing the coefficients 
cm = confusion_matrix(ytest, predicted_classes) 
print (cm) 


# ### Getting important features using Random Forrest

# In[954]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(xtrain, ytrain)


# In[955]:


# True for the features whose importance is greater than the mean importance and False for the rest.
sel.get_support()


# In[956]:


selected_feat=xtrain.columns[(sel.get_support())]
selected_feat


# In[957]:


xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(xtrain, ytrain)
print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(xtrain, ytrain)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(xtest[xtrain.columns], ytest)))


# In[958]:


y_pred = xgb_model.predict(xtest)
print(classification_report(ytest, y_pred))


# In[960]:


from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(xgb_model, ax=ax)


# In[961]:


# Shows the churn probability of each user, and hence can be taken care accordingly.
data_copy['Churn_probability'] = xgb_model.predict_proba(data_copy[xtrain.columns])[:,1]
data_copy.Churn_probability[:5]


# In[962]:


pipe = Pipeline([('classifier' , RandomForestClassifier())])

# Creating parameter grid.
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))}
    ]
# Create grid search object
gso = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
# Fit on data

best_lrm = gso.fit(xtrain, ytrain)


# In[963]:


y_pred_rf = best_lrm.predict(xtest)


# In[964]:


print(best_lrm.best_params_)
print(classification_report(ytest, y_pred_rf))
print(accuracy_score(ytest.to_numpy().flatten(), y_pred_rf))


# In[965]:


gso1 = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(gso1, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(xtrain, ytrain)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(xtest)

# New Model Evaluation metrics 
print(classification_report(ytest, y_pred_acc))

#Logistic Regression (Grid Search) Confusion matrix
print(confusion_matrix(ytest,y_pred_acc))
print(accuracy_score(ytest.to_numpy().flatten(), y_pred_acc))


# In[ ]:




