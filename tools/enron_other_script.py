
"""
This file contains the following:
- General data exploration
- function to visualise the data
- Creation of new features
- Function for feature selection () 

"""

import sys
import pickle
sys.path.append("../tools/")
from pprint import pprint
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# General ----------------------------------------------------------------------

### Total number of data point

def exploration(data_dict):
	print "Total number of observations: ", len(data_dict.keys())
	"""
	Counting POI in the data
	"""

	poi_counter = 0
	for p in data_dict:
		if data_dict[p]["poi"]==1:
			poi_counter += 1
	print "Number of POI: ", poi_counter
	print "Percentage of POI: ", "{0:.1f}%".format(float(poi_counter)/
		len(data_dict.keys()) * 100)
	"""
	Check for missing value: the for loop below checks the number of missing 
	"""

	for k in data_dict.values()[0].keys():
		ctr = 0
		for p in data_dict:
			if data_dict[p][k]=='NaN':
				ctr += 1
		print k, ": " ,ctr



# Data Visualisation -----------------------------------------------------------

# Extract features and labels from dataset for plotting

# plot salary and bonus for each employee
def plot_features(data_dict, feature_x, feature_y):
	features_to_plot = ['poi',feature_x, feature_y]
	data = featureFormat(data_dict, features_to_plot, sort_keys = True) 
	for point in data:
	    (poi, x, y) = point
	    color = 'red' if poi else 'blue'
	    plt.scatter(x, y, color=color)

	plt.xlabel(feature_x)
	plt.ylabel(feature_y)
	plt.show()



# Feature Creation and Selection -----------------------------------------------

# Create new features
def create_poi_email_ratio(data_dict, features_list):
    features_used = ['from_messages', 'to_messages', 'from_poi_to_this_person',
                'from_this_person_to_poi']

    for p in data_dict:
    	employee = data_dict[p]
    	total_from_email = employee['from_poi_to_this_person']
    	+ employee['from_messages']
    	total_to_email = employee['from_this_person_to_poi']
    	+ employee['to_messages']
    	total_email = total_from_email + total_to_email
    	poi_total_interaction = employee['from_poi_to_this_person']
    	+ employee['from_this_person_to_poi']

    	if poi_total_interaction != 0 and total_email != 0:
    		employee['poi_email_ratio']=float(poi_total_interaction)/total_email

    	else:
    		employee['poi_email_ratio'] = 0

    features_list.append('poi_email_ratio')


# SelectKBest
def select_features(data_dict, features_list, k):
	data = featureFormat(data_dict, features_list)
	labels, features = targetFeatureSplit(data)

	selector = SelectKBest(f_classif, k = k)
	selector.fit(features, labels)
	scores = selector.scores_
	tuples = zip(features_list[1:], scores)
	feature_scores = sorted(tuples, key=lambda x: x[1], reverse=True)

	return feature_scores[:k]


# Validation and Evaluation----------------------------------------------------

from sklearn.cross_validation import train_test_split

# Precision, recall and f1 statistics for the given classifier
def evaluate_classifier(clf, features, labels, n=300, test_size = 0.3):
    avg_acc = 0
    avg_prec = 0
    avg_rec = 0
    avg_f1 = 0
    for i in range(n):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        avg_acc += accuracy_score(pred, labels_test)


        avg_prec += precision_score(labels_test, pred, average = "binary")

        avg_rec += recall_score(labels_test, pred, average= "binary")

        avg_f1 += f1_score(labels_test, pred, average="binary")

    print '\tAccuracy: ', avg_acc / n
    print '\tPrecision:', avg_prec / n
    print '\tRecall:   ', avg_rec / n
    print '\tF1 Score:   ', avg_f1 / n
