#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Loading all features available initially for feature list

features_list = ['poi','salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
'from_poi_to_this_person','from_messages', 'from_this_person_to_poi',
'shared_receipt_with_poi']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# manually added from investigation from Lesson 7 mini project
outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)


### Task 3: Create new feature(s)


def computeFraction( feature1, feature2 ):
    """
    given two continuous feature values, calculate the ratio between them
   """
    # NaN will be treated as 0
    if feature1 == "NaN" or feature2 == "NaN":
        fraction = 0
    else:
        fraction = float(feature1)/float(feature2)

    return fraction


for name in data_dict:

    data_point = data_dict[name]

    # Create Bonus to Salary ratio and add to data
    bonus_to_salary = computeFraction(data_point["bonus"], data_point["salary"])
    data_dict[name]["bonus_to_salary"] = bonus_to_salary


# add created features to list
added_features = ["bonus_to_salary"]
# added_features = [] # uncomment to just use default features

for feature in added_features:
    features_list.append(feature)





### Store to my_dataset for easy export below.
my_dataset = data_dict
print len(data_dict)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


def trainTestClassifier(features_train, labels_train, clf_type, params):
    classifiers = {
    "lr" : LogisticRegression(),
    "svm": LinearSVC(),
    "nb" : GaussianNB(),
    }

    # uncomment for non-pca
    # estimator = []
    estimator = [('pca',PCA(n_components=1))]

    estimator.append( (clf_type, classifiers[clf_type]) )

    pipeline = Pipeline(estimator)

    clf = GridSearchCV(pipeline, param_grid=params, scoring="f1")
    clf = clf.fit(features_train, labels_train)
    best_params = clf.best_params_
    clf = clf.best_estimator_


    return clf, best_params



classifiers = {
                'nb': {},
                "svm" : {'svm__C':[1e3, 5e3, 1e4, 5e4, 1e5, 1.0, 0.5, 0.0001],
                        'svm__random_state' : [42]
                        },
                "lr": {
                        'lr__C':[0.005,0.05, 0.5, 1, 10, 100],
                        'lr__penalty': ['l2','l1'],
                        'lr__class_weight': ['balanced'],
                        'lr__tol': [0.001, 0.0005, 10**-10],
                        'lr__random_state':[42]
                      },

               }

best_clf = None
best_f1 = 0


for clf_type, clf_params in classifiers.iteritems():
        clf, params = trainTestClassifier(features_train, labels_train, clf_type, clf_params)
        pred = clf.predict(features_test)
        f1 = f1_score(labels_test, pred)
        recall = recall_score(labels_test, pred)
        precision = precision_score(labels_test, pred)
        print clf_type.upper() + " Results:"
        print "F1: ", f1
        print "Recall: ", recall
        print "Precision: ", precision
        print "Params: ", params
        if f1 > best_f1:
            best_f1 = f1
            best_clf = clf



print "Best Classifier: ", best_clf
clf = best_clf

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
