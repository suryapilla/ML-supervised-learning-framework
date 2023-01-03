##prediction

# this is for testing the flame data
# feed the features
# output to excel the fire/no fire output
# use the outfile in matlab and show the plots
import joblib
import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
import os
#%%
start_time = datetime.now()

ModelName = input("Enter Model name: ")
Model_directoryPath = input("Enter path of Models folder: ")
ModelFilename = Model_directoryPath + "/" + ModelName +'.sav'

DTModel = joblib.load(ModelFilename)
DatasetPath = input("Enter Dataset directory path: ")

features_name = ['Feature1','Feature2','Feature3','Feature4','Feature5','Feature6','Feature7','Feature8']

filelist = []
accuracylist = []

for fileName in os.listdir(DatasetPath):
    scenarioPath = DatasetPath + "/" + fileName
    for scenarioName in os.listdir(scenarioPath):
        if scenarioName.endswith(".csv"):
            Data_in = pd.read_csv(scenarioPath + "/" + scenarioName ,delimiter=',')# Read csv
            features = Data_in[features_name] # copy all the features into features variable
            labels = Data_in.Label # copy labels in labels variable
            pred_output = DTModel.predict(features)
            accuracy_dt = 100*accuracy_score(labels, pred_output)
            accuracylist.append(accuracy_dt)
            filelist.append(scenarioName)
            
time_elapsed = datetime.now() - start_time
DAT = np.column_stack((filelist,accuracylist))

ab = np.zeros(np.size(filelist), dtype=[('var1', 'U100'), ('var2', float)])

ab['var1'] = filelist
ab['var2'] = accuracylist

AccuracyFilePath = input("Enter path of Accuracy folder: ")
np.savetxt(AccuracyFilePath + '/' + 'acc_' + ModelName + '.csv',ab,fmt = '%s, %f',delimiter=',')
print ('Total execution time (hh.mm.ss) {}'.format(time_elapsed) )
