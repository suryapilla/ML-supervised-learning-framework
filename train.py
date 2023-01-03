#%%

import os # to scan through directories

# Reference to pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
import pandas as pd # library for manipulation dataframe
import numpy as np # library to perform array operations


from datetime import datetime # library to caluculate time taken
from sklearn.model_selection import train_test_split # Import train_test_split function
#%%
#>>>>>>>Step 1: Give path to Primary directory of dataset
path = input("Enter path of Dataset folder: ")

# Give feature names
features_name = ['Feature1','Feature2','Feature3','Feature4','Feature5','Feature6','Feature7','Feature8']

#%%
#>>>>>>>Step 2: Reading data into a dataframe
# This cell performs below functionalities
# i.   Scan through each sub-directory 
# ii.  Reads csv files and seperate out features and labels
# iii. Append the data to consume all the dataset into a single variable
features_train_tot=[]
features_test_tot=[]
target_train_tot=[]
target_test_tot=[]

for fileName in os.listdir(path):
    scenarioPath = path + "/" + fileName
    for scenarioName in os.listdir(scenarioPath):
        
        if scenarioName.endswith(".csv"):
           
            Data_in = pd.read_csv(scenarioPath + "/" + scenarioName ,delimiter=',')# Read csv
            features = Data_in[features_name] # copy all the features into features variable
            labels = Data_in.Label # copy labels in labels variable
         
            features_train, features_test, target_train, target_test = train_test_split(features,labels,test_size = 0.3,shuffle=False,stratify = None)
            
            features_train_tot.append(features_train)
            features_test_tot.append(features_test)
            target_train_tot.append(target_train)
            target_test_tot.append(target_test)
      
#%%
# Preprocessing Features and Labels to train data
# X: Input
# y: Label

X_train = np.vstack(features_train_tot)
X_test = np.vstack(features_test_tot)

y_train = np.hstack(target_train_tot)
y_train = np.array(y_train,ndmin=2).transpose()

y_test = np.hstack(target_test_tot)
y_test = np.array(y_test,ndmin=2).transpose()
#%%
#>>>>>>>Step 3: Model generation
print('Model Generation in progress........................')
start_time = datetime.now()
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score  #Import scikit-learn metrics module for accuracy calculation

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion = "gini", random_state = 42,max_depth=7)

# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train,)

time_elapsed = datetime.now() - start_time
print ('Time Taken to Generate Model (hh.mm.ss) {}'.format(time_elapsed) )
print('Model Generated')
#%%
#>>>>>>>Step 4: Training accuracy
# Predict DT model on Train data
train_prediction = clf.predict(X_train)
accuracy_dt = accuracy_score(y_train, train_prediction)
print('DT Train accuracy',accuracy_dt)   
#%%
#>>>>>>>Step 5: Testing accuracy
test_prediction = clf.predict(X_test)
accuracy_dt_test = accuracy_score(y_test, test_prediction)
print('DT Test accuracy',accuracy_dt_test)   
   
#%%
#>>>>>>>Step 6: Save model in .sav file
import joblib
ModelName = input("Enter Model name: ")
Model_directoryPath = input("Enter path of Models folder: ")
filename = Model_directoryPath + "/" + ModelName +'.sav'
joblib.dump(clf, filename)
#%%
#>>>>>>>Step 7: Determine feature importance and save it in csv file
FeatureImportance_directoryPath = input("Enter path of FeatureImportance folder: ")
feature_importances = pd.DataFrame(clf.feature_importances_)
feature_importances = 100*feature_importances
ab = np.zeros(np.size(np.array(feature_importances,ndmin=1)), dtype=[('var1', float),('var2', 'U100')])
                                
ab['var1'] = np.array(feature_importances,ndmin=1).transpose()
ab['var2'] = features_name

np.savetxt(FeatureImportance_directoryPath + "/" + ModelName + '.csv',ab,fmt ='%f, %s',delimiter=',')

#%%
#>>>>>>>Step 8: DT visualization and saving it in txt file
from sklearn.tree import export_graphviz
from sklearn import tree
DT_directoryPath = input("Enter path of DT folder: ")
tree.export_graphviz(clf,out_file= DT_directoryPath + "/" + ModelName + '.txt',label='all',filled=True,rounded=True,class_names=True)
# To visualize tree use following link: http://graphviz.it/#/gallery/grammar.gv
