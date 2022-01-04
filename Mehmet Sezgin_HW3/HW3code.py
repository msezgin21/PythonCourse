import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer

CSES4 = pd.read_csv('cses4_cut.csv')
X = CSES4.iloc[:,:-1]
y = CSES4.iloc[:,-1]
CSES4

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

##### Firstly, I will select Random Forest Classifier
RandomForest = RandomForestClassifier()
RandomForest_accuracy=cross_val_score(RandomForest, X, y, cv=cv).mean()

##### Secondly, I will select Linear Discriminant Analysis
LiDiAn = LinearDiscriminantAnalysis()
LiDiAn_accuracy=cross_val_score(LiDiAn, X, y, cv=cv).mean()

##### Thirdly, I will select K-Nearest Neighbors
KNeNeig = KNeighborsClassifier()
KNeNeig_accuracy=cross_val_score(KNeNeig, X, y, cv=cv).mean()

##### Fourthly, I will select Naive Bayes
NaBayes = GaussianNB()
NaBayes_accuracy=cross_val_score(NaBayes, X, y, cv=cv).mean()

##### Fifthly, I will selecet Support Vector Machine
SuVeMa = SVC(probability = True)
SuVeMa_accuracy=cross_val_score(SuVeMa, X, y, cv=cv).mean()

##### Sixtly, I will select Decision Tree
DeTr = DecisionTreeClassifier()
DeTr_accuracy=cross_val_score(DeTr, X, y, cv=cv).mean()

##### Sevently, I will select Logistic Regression
LogReg = LogisticRegression()
LogReg_accuracy=cross_val_score(LogReg, X, y, cv=cv).mean()

##### Finally, I will select Quadratic Discriminant Analysis
QuDiAn = QuadraticDiscriminantAnalysis()
QuDiAn_accuracy=cross_val_score(QuDiAn, X, y, cv=cv).mean()

pd.options.display.float_format = '{:,.2f}%'.format
accuracies1 = pd.DataFrame({
    'Model'       : ['Random Forest', 'Linear Discriminant Analysis', 'K-Nearest Neighbors', 'Bayes', 'Support Vector Machine', 'Decision Tree', 'Logistic Regression', 'Quadratic Discriminant Analysis'],
    'Accuracy'    : [100*RandomForest_accuracy, 100*LiDiAn_accuracy, 100*KNeNeig_accuracy, 100*NaBayes_accuracy, 100*SuVeMa_accuracy, 100*DeTr_accuracy, 100*LogReg_accuracy, 100*QuDiAn_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies1.sort_values(by='Accuracy', ascending=False)



#####Data: not having normal distrubution

variables = CSES4[
    ['D2011', 'D2015', 'D2016', 'D2021', 'D2022', 'D2023', 'D2026', 'D2027', 'D2028', 'D2029', 'D2030', 'age']]
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
plotnumber = 1

for column in variables:
    if plotnumber <= 12:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(variables[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()

#####Making data to have normal distrubution

quantTrans = preprocessing.QuantileTransformer(random_state=0)
trans = quantTrans.fit_transform(variables)
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
plotnumber = 1

for column in range(trans.shape[1]):
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(trans[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()


