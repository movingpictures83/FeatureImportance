import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model

def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
    'feature_importance': importances}).sort_values('feature_importance', ascending = False).reset_index(drop = True)
    return df

def drop_col_feat_imp(model, X_train, y_train, X_valid, y_valid, random_state=42):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
        importances.append(benchmark_score - drop_col_score)
        print("column {}, importance: {}".format(col, benchmark_score - drop_col_score))
        print(benchmark_score - drop_col_score)

    importances_df = imp_df(X_train.columns, importances)
    return importances_df

def random_forest(data_df, y_col, cl="Classifier", test_size=0.3 ,validate=True):
    y = data_df[y_col]

    X = data_df.drop([y_col], axis=1)

    # Create random variable for benchmarking
    X["random"] = np.random.random(size= len(X))

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = test_size, random_state = 42)

    if cl == "Classifier":
        rf = RandomForestClassifier(n_estimators = 100,
                                   n_jobs = -1,
                                   oob_score = True,
                                   bootstrap = True,
                                   random_state = 42)
    else:
        rf = RandomForestRegressor(n_estimators=100,
                                    n_jobs=-1,
                                    oob_score=True,
                                    bootstrap=True,
                                    random_state=42)
    rf.fit(X_train, y_train)

    print('Training accuracy: {:.2f} \nOOB Score: {:.2f} \nTest Accuracy: {:.2f}'.format(rf.score(X_train, y_train),
                                                                                                 rf.oob_score_,
                                                                                                 rf.score(X_valid, y_valid)))
    # scores = cross_val_score(rf, X, y, cv=5)
    # print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # a = rf.predict(X_valid)

    # importances_df = drop_col_feat_imp(rf, X, y, X_valid, y_valid)
    # importances_df.to_csv("/Users/stebliankin/Desktop/SabrinaProject/FeatureSelection/importance_df.csv")
    if validate:
        scores = cross_val_score(rf, X, y, cv=25)
        print("CV Accuracy: %0.2f " % (scores.mean()))
    return rf

import PyPluMA

class FeatureImportancePlugin:
    def input(self, infile):
        paramfile = open(infile, 'r')
        self.parameters = dict()
        for line in paramfile:
           contents = line.strip().split('\t')
           self.parameters[contents[0]] = contents[1]

    def run(self):
        pass

    def output(self, outfile):
        data_path = PyPluMA.prefix()+"/"+self.parameters["datapath"]
        data_df = pd.read_csv(data_path)

        cleanupfile = open(PyPluMA.prefix()+"/"+self.parameters["cleanup"], 'r')
        cleanup_nums = dict()
        for line in cleanupfile:
            contents = line.strip().split('\t')
            cleanup_nums[contents[0]] = dict()
            specificfile = open(PyPluMA.prefix()+"/"+contents[1], 'r')
            for line2 in specificfile:
                contents2 = line2.strip().split('\t')
                cleanup_nums[contents[0]][contents2[0]] = int(contents2[1])

        data_df.replace(cleanup_nums, inplace=True)

        # Drop id column:
        data_df = data_df.drop([self.parameters["dropid"]], axis=1)

        # remove NaN:
        data_df = data_df.fillna(0)

        y_col = self.parameters["ycol"]
        rf = random_forest(data_df, y_col, "Classifier", validate=False)

#y= "interleukin6"
#test_size = 0.25
#validate = True


