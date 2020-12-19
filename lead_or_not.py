# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:33:32 2020

@author: Ankita
"""
#--import libraries
#from imblearn.over_sampling import SMOTE 
#from imblearn.over_sampling import KMeansSMOTE
import random
from sklearn.feature_selection import RFE
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import roc_auc_score, roc_curve, auc
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import calendar
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 10000)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import ensemble
import gc
import numpy as np
h = .02  # step size in the mesh
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger('CHURN_PRED')
logging.basicConfig(filename='log_leads.log',level=logging.DEBUG)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info('logger established')
#-------------------------------------------------------------------------------------

names = ["XGB","GBM","Neural Net","Naive Bayes","Decision Tree", "Random Forest","Nearest Neighbors","LogisticRegression"]

classifiers = [XGBClassifier(max_delta_step=2,scale_pos_weight=100),
               GradientBoostingClassifier(n_estimators=100,random_state=0,verbose=1),
    MLPClassifier(alpha=0.00001,max_iter=27577,activation='logistic',batch_size=100,verbose=True,random_state=0),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=7, n_estimators=100,min_samples_split=2, max_features=3, class_weight='balanced'),
    KNeighborsClassifier(2),
    LogisticRegression(penalty='l2',class_weight='balanced', random_state=0, 
                                        solver='lbfgs', max_iter=100, multi_class='auto', verbose=0,
                                        warm_start=False, n_jobs=None, l1_ratio=None)]
#import csv
lead_data=pd.read_csv('Leads.csv')
print(lead_data.describe())
print(lead_data.info())
print(lead_data.columns)

#changing some 

categorical_vars= lead_data.select_dtypes(exclude=['float64', 'int64']).columns
numerical_vars = lead_data.select_dtypes(include=['float64', 'int64']).columns

for cat in categorical_vars.tolist():
    print(cat)
    print(lead_data[cat].value_counts(dropna=False))
    
cat_to_drop=['Magazine','Get updates on DM Content','Update me on Supply Chain Content',
             'I agree to pay the amount through cheque','Receive More Updates About Our Courses']
lead_data.drop(columns=cat_to_drop,axis=1,inplace=True)
categorical_vars=categorical_vars.drop(cat_to_drop)
categories_yes_no=['Do Not Email','Do Not Call','Search','Newspaper Article',
                   'X Education Forums','Newspaper','Digital Advertisement',
                   'Through Recommendations','A free copy of Mastering The Interview']
categories_other=['Lead Source','Last Activity','Specialization',
                  'Tags','Lead Quality','']
dict_ordinal={'01.High':1,'02.Medium':2,'03.Low':3}
dict={'Yes':1,'No':0}

for cat in categories_yes_no:
    print(cat)
    lead_data[cat]=lead_data[cat].replace(dict)

   
#Checking percentage of null values present in training dataset 
missing_num= lead_data[lead_data.columns].isna().sum().sort_values(ascending=False)
missing_perc= (lead_data[lead_data.columns].isna().sum()/len(lead_data)*100).sort_values(ascending=False)
missing= pd.concat([missing_num,missing_perc],keys=['Total','Percentage'],axis=1)
missing_df= missing[missing['Percentage']>0]
print(missing_df)


lead_data['Tags'].fillna('Other',inplace=True)
lead_data['Lead Quality'].fillna('Other',inplace=True)
lead_data['Lead Profile'].fillna('Select',inplace=True)
lead_data['What matters most to you in choosing a course'].fillna(lead_data['What matters most to you in choosing a course'].mode()[0],inplace=True)
lead_data['What is your current occupation'].fillna(lead_data['What is your current occupation'].mode()[0],inplace=True)
lead_data['Country'].fillna(lead_data['Country'].mode()[0],inplace=True)
lead_data['City'].fillna(lead_data['City'].mode()[0],inplace=True)
lead_data['Specialization'].fillna(lead_data['Specialization'].mode()[0],inplace=True)
lead_data['How did you hear about X Education'].fillna(lead_data['How did you hear about X Education'].mode()[0],inplace=True)
lead_data['Last Activity'].fillna(lead_data['Last Activity'].mode()[0],inplace=True)
lead_data['Lead Source'].fillna(lead_data['Lead Source'].mode()[0],inplace=True)




lead_data['Asymmetrique Profile Index'].fillna(lead_data['Asymmetrique Profile Index'].mode()[0],inplace=True)
lead_data['Asymmetrique Activity Index'].fillna(lead_data['Asymmetrique Activity Index'].mode()[0],inplace=True)
lead_data['Asymmetrique Profile Score'].fillna(round(lead_data['Asymmetrique Profile Score'].mean()),inplace=True)
lead_data['Asymmetrique Activity Score'].fillna(round(lead_data['Asymmetrique Activity Score'].mean()),inplace=True)

Q1 = lead_data['Total Time Spent on Website'].quantile(0.25)
Q2 = lead_data['Total Time Spent on Website'].quantile(0.5)
Q3 = lead_data['Total Time Spent on Website'].quantile(0.75)

q1_total_visits=round(lead_data[lead_data['Total Time Spent on Website']<Q1]['Total Time Spent on Website'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']<Q1) & (lead_data['TotalVisits'].isnull()),'TotalVisits']=q1_total_visits
q2_total_visits=round(lead_data[lead_data['Total Time Spent on Website']<Q2]['Total Time Spent on Website'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']<Q2) & (lead_data['TotalVisits'].isnull()),'TotalVisits']=q2_total_visits
q3_total_visits=round(lead_data[lead_data['Total Time Spent on Website']<Q3]['Total Time Spent on Website'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']<Q3) & (lead_data['TotalVisits'].isnull()),'TotalVisits']=q3_total_visits
q4_total_visits=round(lead_data[lead_data['Total Time Spent on Website']>=Q3]['Total Time Spent on Website'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']>=Q3) & (lead_data['TotalVisits'].isnull()),'TotalVisits']=q4_total_visits

q1_page_visits=round(lead_data[lead_data['Total Time Spent on Website']<Q1]['Page Views Per Visit'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']<Q1) & (lead_data['Page Views Per Visit'].isnull()),'Page Views Per Visit']=q1_page_visits
q2_page_visits=round(lead_data[lead_data['Total Time Spent on Website']<Q2]['Page Views Per Visit'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']<Q2) & (lead_data['Page Views Per Visit'].isnull()),'Page Views Per Visit']=q2_page_visits
q3_page_visits=round(lead_data[lead_data['Total Time Spent on Website']<Q3]['Page Views Per Visit'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']<Q3) & (lead_data['Page Views Per Visit'].isnull()),'Page Views Per Visit']=q3_page_visits
q4_page_visits=round(lead_data[lead_data['Total Time Spent on Website']>=Q3]['Page Views Per Visit'].mean())
lead_data.loc[(lead_data['Total Time Spent on Website']>=Q3) & (lead_data['Page Views Per Visit'].isnull()),'Page Views Per Visit']=q4_page_visits
#-------------------missing values imputation done
#--------------------------------------------------------

categorical_vars= lead_data.select_dtypes(exclude=['float64', 'int64']).columns
numerical_vars = lead_data.select_dtypes(include=['float64', 'int64']).columns

for i in numerical_vars:
    plt.figure(figsize=(8,6))
    plt.hist(lead_data[i])
    plt.show()