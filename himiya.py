import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics as sm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

warnings.simplefilter(action='ignore', category=FutureWarning)

def regression_model(numbers, properties):
    numbers, properties = make_unique(numbers, properties)
    X, y = numbers, properties
   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    ##X  = preprocessing.scale(X)
    X= preprocessing.normalize(X)
    
    slr = LinearRegression(normalize=True)
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    
    print("LINEAR REGRESSION \n")
    print('Slope: {:.2f}'.format(slr.coef_[0]))
    print('Intercept: {:.2f}'.format(slr.intercept_))
    
    regression_output(X_test, X_train, y_test, y_train, y_test_pred, slr)

    print("\n\n\n SVM \n")
    model = SVR()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    
    regression_output(X_test, X_train, y_test, y_train, y_test_pred,  model)

    print("\n\nDECISION TREE REGRESSION \n")
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    #print("Важности признаков:\n{}".format(model.feature_importances_))
    

    regression_output(X_test, X_train, y_test, y_train,y_test_pred ,  model)
    plot_feature_importances(X,y,model)
    
def plot_feature_importances(X,y,model):
     n_features = len(X[1])
     plt.barh(range(n_features), model.feature_importances_, align='center')
     plt.yticks(np.arange(n_features), range(len(X[1])))
     plt.xlabel("Важность признака")
     plt.ylabel("Признак")
     plt.show()
       

def regression_output(X_test, X_train, y_test, y_train,y_test_pred,  model):
    print("Regressor model performance:")
    print('Test score: {:.2f}'.format(model.score(X_test, y_test)))
    print('Train score: {:.2f}'.format(model.score(X_train, y_train)))
    print("Mean absolute error(MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error(MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

    print('y_test\ty_test_pred')
    for i in range(min(len(y_test),15)):
        print(y_test[i], '\t', '{0:.2f}'.format(float(y_test_pred[i])))

def make_unique(numbers, properties):
    numbers_unq, properties_unq = [], []
    for i in range(len(numbers)):
        if numbers[i] not in numbers[i + 1:]:
            numbers_unq.append(numbers[i])
            properties_unq.append(properties[i])
    return np.asarray(numbers_unq), np.asarray(properties_unq)




numbers = []
with open('C:/Users/Logge/OneDrive/Рабочий стол/Chains/Experiment-Space/DB_SOIL/matrk2m1.txt') as file:
    for line in file.readlines():
        if line.isspace(): break
        numbers.append(list(map(float, line.split())))

properties = []
with open('C:/Users/Logge/OneDrive/Рабочий стол/Chains/SOIL/soil-1-1.set') as file:
    for line in file.readlines():
        if line.startswith('A3'):
            properties.append(float(line.split()[1]))

regression_model(numbers,properties)



##slr = LinearRegression(normalize = True)
##slr.fit(X, y)
##scr = slr.score(X, y)
##y_pred = slr.predict(X)
##print('Slope: {:.2f}'.format(slr.coef_[0]))
##print('Intercept: {:.2f}'.format(slr.intercept_))
##print('Score: {:.2f}'.format(scr))

##print('y_test:', y_test)


##plt.scatter(X, y)
##plt.plot(X, slr.predict(X), color='red', linewidth=2);

##neigh = KNeighborsClassifier(n_neighbors=1)
##neigh.fit(X_train, y_train) 
##y_train_pred = neigh.predict(X_train)
##y_test_pred = neigh.predict(X_test)
##print("\n\n\nKNN \n")
##print('Score: {:.2f}'.format(neigh.score(X, y)))
##print('y_test\ty_test_pred')
##for i in range(len(y_test)):
##    print(y_test[i], '\t', '{.2f}'.format(y_test_pred[i]))


##model = ExtraTreesClassifier()
##model.fit(X_train, y_train)
### display the relative importance of each attribute
##print(model.feature_importances_)

##model = LogisticRegression()
### create the RFE model and select 3 attributes



##model = LogisticRegression()
##model.fit(X_train, y_train)
##print(model)
### make predictions
##y_test_pred = model.predict(X_test)
### summarize the fit of the model
####print(metrics.classification_report(y_test, y_test_pred))
####print(metrics.confusion_matrix(y_test, y_test_pred))
##
##print("\n\n\n Logistic Regression\n")
##print('Score: {:.2f}'.format(model.score(X, y)))
##print('y_test\ty_test_pred')
##for i in range(len(y_test)):
##    print(y_test[i], '\t', float(y_test_pred[i]))
##
##print("\n\n\n Naive Baes\n")
##model = GaussianNB()
##model.fit(X_train, y_train)
##y_test_pred = model.predict(X_test)
##print('Score: {:.2f}'.format(model.score(X, y)))
##print('y_test\ty_test_pred')
##for i in range(len(y_test)):
##    print(y_test[i], '\t', float(y_test_pred[i]))

##print("\n\n\n Decision Tree \n")
##model = DecisionTreeClassifier()
##model.fit(X_train, y_train)
##y_test_pred = model.predict(X_test)
##print('Score: {:.2f}'.format(model.score(X, y)))
##print('y_test\ty_test_pred')
##for i in range(len(y_test)):
##    print(y_test[i], '\t', float(y_test_pred[i]))




##rfe = RFE(model, 5)
##rfe = rfe.fit(X_train, y_train)
### summarize the selection of the attributes
##print(rfe.support_)
##print(rfe.ranking_)
##
##y_test_pred=rfe.predict(X_test)
##
##print("\n\n\nRFE \n")
##print('Score: {:.2f}'.format(rfe.score(X, y)))
##print('y_test\ty_test_pred')
##for i in range(len(y_test)):
##    print(y_test[i], '\t','{0:.2f}'.format(float(y_test_pred[i])))
##
##

    
