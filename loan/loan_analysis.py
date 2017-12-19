# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:58:56 2017

@author: shshriv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
    
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

## Encoder for features
le = LabelEncoder()
## Encoder for target
lsle = LabelEncoder()

def fixTypeAndNA(df):
    """
    Data Cleaning
    """
    #Making applicant income as float instead of int
    df['ApplicantIncome'] = df['ApplicantIncome'].astype(float)
    # some values of dependents have '+' in it, removing those and keeping the number. '3+' -> 3
    df['Dependents'] = df.Dependents.str.replace('+', '')
    # filling missing dependents value with the mode of column
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    # changing column type to int
    df['Dependents'] = df['Dependents'].astype(int)
    # filling missing gender value with the back fill
    df['Gender'] = df['Gender'].fillna(method='bfill').astype('category')
    # filling missing Married value with the back fill
    df['Married'] = df['Married'].fillna(method='bfill').astype('category')
    # filling missing Self_Employed value with the back fill
    df['Self_Employed'] = df['Self_Employed'].fillna(method='bfill').astype('category')
    # filling missing Education value with the back fill
    df['Education'] = df['Education'].fillna(method='bfill').astype('category')
    # filling missing Property_Area value with the back fill
    df['Property_Area'] = df['Property_Area'].fillna(method='bfill').astype('category')
    # filling missing LoanAmount value with the mean of column
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    # filling missing Loan_Amount_Term value with the mode of column
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    # filling missing Credit_History value with 0.5, which is better than 0 and worse than having credit history
    df['Credit_History'] = df['Credit_History'].fillna(value=0.5)
    print("Missing value treatment done")
    return df

def featureScale(df):
    """
    FEATURE SCALING
    """
    scaler = MinMaxScaler()
    #print(df['ApplicantIncome'].head())
    df[['ApplicantIncome']] = scaler.fit_transform(df[['ApplicantIncome']])
    df[['CoapplicantIncome']] = scaler.fit_transform(df[['CoapplicantIncome']])
    df[['LoanAmount']] = scaler.fit_transform(df[['LoanAmount']])
    df[['Loan_Amount_Term']] = scaler.fit_transform(df[['Loan_Amount_Term']])
    print("Scaling Done")
    #print(df['ApplicantIncome'].head())
    return df

def encodeCategoricalData(df):
    """
    Encoding
    """
    df.Gender = le.fit_transform(df.Gender)
    df.Married = le.fit_transform(df.Married)
    df.Education = le.fit_transform(df.Education)
    df.Self_Employed = le.fit_transform(df.Self_Employed)
    df.Property_Area = le.fit_transform(df.Property_Area)
    print("Encoding Done")
    return df

def preProcess(df):
    
    df = fixTypeAndNA(df)
    df = featureScale(df)
    df = encodeCategoricalData(df)
    del(df['Loan_ID'])
#    print("processed df ")
#    print(df.head())
#    print("Describe:", df.describe())
    return df

def preProcessTrainData(df):
    #marking Loan_Status as category
    df['Loan_Status'] = df['Loan_Status'].astype('category')
    
    # Encode Loan_Status of train data
    df.Loan_Status = lsle.fit_transform(df.Loan_Status)
    #print("Dtypes after preprocessing:", df.dtypes)
    df.to_csv('v3_cleaned.csv')
    return df

def exploreData(df):
    """
    Data Exploration
    """
    import seaborn as sns
    print("Describe:", df.describe())
    print(df.columns)
    print("Dtypes:", df.dtypes)
    
    # Scatter plot to show all attribute relations
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal="kde")
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal="hist")
    plt.tight_layout()
    plt.show()
    
    #Using seaborn to plot all charts to understand relation
    sns.set(style="ticks", color_codes=True)
    ##sns.pairplot(data=df.dropna(), hue="Loan_Status", size=2.5 )
    print
    sns.lmplot("Credit_History", "CoapplicantIncome", data=df.dropna(), hue="Loan_Status", fit_reg=False)
    sns.lmplot("Credit_History", "LoanAmount", data=df.dropna(), hue="Loan_Status", fit_reg=False)
    sns.lmplot("Loan_Amount_Term", "LoanAmount", data=df.dropna(), hue="Loan_Status", fit_reg=False)
    
    print(pd.crosstab(df.Education, df.Self_Employed))
    edu_empl = pd.crosstab(index=df.Education, columns=df.Self_Employed, margins=True)
    print("edu_empl", edu_empl)
    
    df[['Credit_History', 'Loan_Amount_Term', 'Loan_Status']].plot.bar(stacked=True)
    print("Training data size:",len(df)) #614
    print("Training data size without NaNs:",len(df.dropna())) #480
    
    # Plotting numeric columns to understand their ranges
    df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")].index.values].hist(figsize=[11,11])
    return df

def splitData(df, fraction):
    """
    Splitting
    """
    from sklearn.model_selection import train_test_split
    # define the target variable (dependent variable) as y
    y = df.Loan_Status
    
    #splitting data with 20% as test data
    X_train, X_test, y_train, y_test = train_test_split(df[['Gender', 'Married', 'Dependents', 'Education',
           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area']], y, test_size=fraction, random_state=7)
#    print (X_train.shape, y_train.shape)
#    print (X_test.shape, y_test.shape)
    #print("Y = ",y)
    #print("y_train = ",y_train)
    #print("X_train = ",X_train)
    ## standardize the data attributes (accuracy dropped after this)
    #from sklearn.preprocessing import scale
    #standardized_X = scale(X_train)
    #X_train = standardized_X
    #print("standardized_X = ",X_train)
    return X_train, X_test, y_train, y_test


def fitNevaluate(X_train, X_test, y_train, y_test, clf, name):
    #start_time = time.time()
    #print("Fitting started...", start_time)
    model = clf.fit(X_train, y_train)
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    #predictions = clf.predict(X_test)
    
#    ## The line / model
#    plt.scatter(y_test, predictions)
#    plt.xlabel("True Values")
#    plt.ylabel("Predictions")
#    plt.show()
    acc = model.score(X_test, y_test)
    print("======================")
    print("Accuracy score for", name, acc)
    return clf, acc

def getBestModel(X_train, X_test, y_train, y_test):
    """
    Model Building
    """
    algos = {}
    from sklearn import svm
    from sklearn.linear_model import SGDClassifier, Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    classifiers = [
        ("SVM", svm.SVC(C= 10, gamma= 0.1, kernel= 'rbf')),
        ("SVM-Default", svm.SVC()),
        ("SGD", SGDClassifier(max_iter=10, tol=None)),
        ("ASGD", SGDClassifier(average=True, max_iter=10, tol=None)),
        ("Perceptron", Perceptron(max_iter=10, tol=None)),
        ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                             C=1.0, max_iter=10, tol=None)),
        ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                              C=1.0, max_iter=10, tol=None)),
        ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X_train.shape[0])),
        ("XGBoost", XGBClassifier()),
        ("GaussianNB", GaussianNB()),
        ("DecisionTreeClassifier",DecisionTreeClassifier(random_state=0)),
        ("AdaBoostClassifier",AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                                                 algorithm="SAMME", n_estimators=200)),
        ("RandomForestClassifier",RandomForestClassifier(max_depth=2, random_state=0)),
        ("neural_network.MLPClassifier", MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                       hidden_layer_sizes=(5, 2), random_state=1))
    ]

    for name, clf in classifiers:
        #print("training %s" % name)
        clf, acc = fitNevaluate(X_train, X_test, y_train, y_test, clf, name)
        algos[acc] = (clf, name)
    
    top = sorted(algos.items())
    #print(top)
    clf
    for key, value in top:
        clf = value[0]
        acc = key
    
    print("Best accuracy", acc)
    print("Best Model", clf)
    return clf

def svc_param_selection(X, y, nfolds):
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm
    
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    start_time = time.time()
    print("GridSearchCV started...", start_time)
    grid_search.fit(X, y)
    print("--- %s seconds ---" % (time.time() - start_time), grid_search.n_splits_)
    grid_search.best_params_
    
    # View the accuracy score
    print('Best score for data:', grid_search.best_score_) 
    # View the best parameters for the model found using grid search
    print('Best C:',grid_search.best_estimator_.C) 
    print('Best Kernel:',grid_search.best_estimator_.kernel)
    print('Best Gamma:',grid_search.best_estimator_.gamma)
    return grid_search.best_params_

def prepareTestPrediction(clf, testcsv, pca):
    """
    Testing
    """
    
    tdf = pd.read_csv(testcsv)
    ids = tdf['Loan_ID']
    tdf = preProcess(tdf)
    
    if pca:
        tdf = pca.fit_transform(tdf)
        tpredicts = clf.predict(tdf)
    else:
        tpredicts = clf.predict(tdf[['Gender', 'Married', 'Dependents', 'Education',
           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area']])
        
    tpredicts = lsle.inverse_transform(tpredicts)
    out = np.column_stack((ids,tpredicts))
#    print("transformed")
#    print(tpredicts[:10])
#    print(tpredicts.shape, ids.shape)
#    print(out[:10])
    
    import csv
    header = [['Loan_ID','Loan_Status']]
    with open('preds.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(header)
        writer.writerows(out)
    print("Writing Done")


def doPCA(X_train, X_test):
    """
    PCA
    """
    pca = PCA(n_components=9)
    pca.fit(X_train)
    
    #The amount of variance that each PC explains
    var= pca.explained_variance_ratio_
    #print ("var", var)
    #Cumulative Variance explains
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    #print ("var1",var1)
    plt.plot(var1)
    
    X1_train=pca.fit_transform(X_train)
    X1_test=pca.fit_transform(X_test)
#    print ("X_train",X_train)
#    print
#    print ("X1_train",X1_train, X1_train.shape)
    print("PCA Done")
    return X1_train, X1_test, pca


### START
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
### Exploration
exploreData(df)
### Pre processing starts
df = preProcess(df)
### Pre processing for training data only
df = preProcessTrainData(df)

### Data splitting
X_train, X_test, y_train, y_test = splitData(df, 0.2)
### PCA
pca=None
#All the accuracies came down after PCA so don't do it
#X_train, X_test, pca = doPCA(X_train, X_test)

print("Best Params",svc_param_selection(X_train, y_train, 10))

### GEtting trained model
clf = getBestModel(X_train, X_test, y_train, y_test)
### Generating results for submission
prepareTestPrediction(clf, 'test_Y3wMUE5_7gLdaTN.csv', pca)






    


