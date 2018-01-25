#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from enron_other_script import *
from sklearn.feature_selection import SelectKBest, f_classif

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)


### Task 1: Select what features to use. -----------------------------------
### features_list is a list of strings, each of which is a feature name.

### Below is a list of all features. Final feature list will be created after
### running SelectKBest in the section, 'Task 3', below.

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees' ,'to_messages',
'from_poi_to_this_person', 'from_messages','from_this_person_to_poi', 
'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Missing values, 'NaN', noted within many features. Set 'NaN' to zero before

for k in data_dict.values()[0].keys():
	ctr = 0
	for p in data_dict:
		nas = data_dict[p][k]
		if  nas !='NaN' and data_dict[p] != 'email_address':
			continue
		else:
			data_dict[p][k] = 0

### Task 2: Remove outliers ----------------------------------------------------

### Plot before removing outliers need to save the results on the report
# plot_features(data_dict, 'salary', 'bonus')

### Remove outliters
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Plot before removing outliers
# plot_features(data_dict, 'salary', 'bonus')




### Task 3: Create new feature(s) ----------------------------------------------

### Create New Feature
create_poi_email_ratio(data_dict, features_list)

### Feature selection: SelectKBest
feature_scores, my_features = select_features(data_dict, features_list, k=6)

# ### Plot feature scores

# y_pos = np.arange(len(feature_scores))
# performance = [score[1] for score in feature_scores]
 
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, [score[0] for score in feature_scores], rotation=50)
# plt.ylabel('Scores')
# plt.title('SelectKBest Scores')
 
# plt.show()


### Ensure features_list only includes the selected features from SelectKBest
features_list =['poi'] + my_features

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Feature Scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)


### Task 4: Try a variety of classifiers ----------------------------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



### ----------------------------------------------------------------------------
### Task 5: Tune the classifier to achieve better than .3 precision and recall 
### using our testing script.

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.cross_validation import train_test_split


### Using train_test_split to validate our model

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### GaussianNB

clf = GaussianNB()
print "Gaussian"
evaluate_classifier(clf, features, labels)


### SVM: 5 C values are used to tune the algorithm

# for c in [1, 10, 100, 1000, 10000]:
#     clf = SVC(kernel="rbf", C=c)
#     print "SVM with C =", c
#     evaluate_classifier(clf, features, labels)


### Decision Tree: 5 min_sample_split values are used to tune the algorithm

# for split in [2, 5, 8, 10, 15]:
#     clf = DecisionTreeClassifier(min_samples_split = split)
#     print "DecisionTreeClassifier with min_samples_split =", split
#     evaluate_classifier(clf, features, labels)


# Final Pick: GaussianNB
clf = GaussianNB()




###-----------------------------------------------------------------------------
### Task 6: Dump the classifier, dataset, and features_list so anyone can
### check the results. 

dump_classifier_and_data(clf, my_dataset, features_list)    