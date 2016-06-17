import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

##reading student data
student_data = pd.read_csv("student-data.csv")
print "successfull reading student data"
##print student_data.head()

##data exploration
totalstudents = student_data.shape[0]
totalfeatures = student_data.shape[1] -1
totalpassed = student_data[student_data["passed"]=="yes"].shape[0]
totalfailed = student_data[student_data["passed"]=="no"].shape[0]
graduation_rate = (float)(totalpassed)/(totalstudents)*100
print "total number of students:", totalstudents
print "total number of features:",totalfeatures
print "number of students who passed:",totalpassed
print "number of students who failed:", totalfailed
print "graduation rate:{:.2f}%".format(graduation_rate)

##identify feature and targets
features_columns = list(student_data.columns[:-1])

##splitting testing and training data

target_column = student_data.columns[-1] 
print "feature columns:{}" .format(features_columns)
print "target column:{}" .format(target_column)

x_all = student_data[features_columns]
y_all = student_data[target_column]
print "feature values", x_all.head()

##preprocessing i.e. converting non-numeric data into numeric data
def preprocess_features(ft):
     output = pd.DataFrame(index = ft.index)
     for column,columndata in ft.iteritems():
         if columndata.dtype == object:
                 columndata = columndata.replace(['yes', 'no'], [1, 0])
         if columndata.dtype == object:
                 columndata = pd.get_dummies(columndata, prefix=column)
         output = output.join(columndata)
     return output
x_all = preprocess_features(x_all)
print "Processed features : {} total features".format(len(x_all.columns))
print list(x_all.columns)

total = student_data.shape[0]
train = 300
test = total  - train
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = test, random_state = 5)
print "training set: {} items".format(x_train.shape[0])
print "test set: {} items" .format(x_test.shape[0])

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
clf1 = GaussianNB()
clf2 = GradientBoostingClassifier()
clf3  = svm.SVC()

x_train_100 = x_train[:100]
y_train_100 = y_train[:100]

x_train_200 = x_train[:200]
y_train_200 = y_train[:200]

x_train_300 = x_train[:300]
y_train_300 = y_train[:300]

print "set-size:100"
print "1.>naive bias  2.>gradient boosting 3.>support vector machine"
train_predict(clf1,x_train_100,y_train_100,x_test,y_test)
train_predict(clf2,x_train_100,y_train_100,x_test,y_test)
train_predict(clf3,x_train_100,y_train_100,x_test,y_test)

print "set-size:200"
print "1.>naive bias  2.>gradient boosting 3.>support vector machine"
train_predict(clf1,x_train_200,y_train_200,x_test,y_test)
train_predict(clf2,x_train_200,y_train_200,x_test,y_test)
train_predict(clf3,x_train_200,y_train_200,x_test,y_test)

print "set-size:300"
print "1.>naive bias  2.>gradient boosting 3.>support vector machine"
train_predict(clf1,x_train_300,y_train_300,x_test,y_test)
train_predict(clf2,x_train_300,y_train_300,x_test,y_test)
train_predict(clf3,x_train_300,y_train_300,x_test,y_test)

## model tuning
def performance_metric(y_true, y_predict):
    error = f1_score(y_true, y_predict, pos_label='yes')
    return error
def fit_model(x, y):
    classifier = svm.SVC()
    parameters = {'kernel':['poly','rbf','sigmoid'], 'degree':[1,2,3], 'C':[0.1, 1, 10]}
    f1_scorer = make_scorer(performance_metric, greater_is_better=True)
    clf = GridSearchCV(classifier, param_grid = parameters, scoring = f1_scorer)
    clf.fit(x, y)
    return clf

clf = fit_model(x_train, y_train)
print "successfully fit a model."
print "the best parameters were:"
print clf.best_params_
print "final score after tuning for training {}".format(predict_labels(clf, x_train, y_train))
print "final score after tuning for testing {}".format(predict_labels(clf, x_test, y_test))




