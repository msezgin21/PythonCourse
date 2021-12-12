import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from linear_regression import linear_regression


def linear_regression(mydata):

    if len(mydata) == 0:
        return

    if not isinstance(mydata, pd.DataFrame) :
        return


    for column_name in mydata.columns:
        mydata[column_name] = pd.to_numeric(mydata[column_name], errors="c")

    clear_mydata = mydata.dropna(axis=0)

    clear_mydata = clear_mydata.reset_index(drop=True)

    X = clear_mydata.iloc[:, :-1]
    y = clear_mydata[clear_mydata.columns[-1]]

    coefficients = ((np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y))

    yhat = X.dot(coefficients)

    Errors = y - yhat

    var1 = ((Errors.T.dot(Errors)) / (len(clear_mydata) - len(X.columns) - 1)) * (np.linalg.inv(X.T.dot(X)))

    t_st = stats.t.ppf(0.95, len(clear_mydata) - 1)

    Conf_intervals = []
    for i in range(0, len(coefficients)):
        Conf_intervals.append([coefficients[i]-t_st*var1[i][i], coefficients[i]+t_st*var1[i][i]])


    return coefficients, Errors, Conf_intervals




    