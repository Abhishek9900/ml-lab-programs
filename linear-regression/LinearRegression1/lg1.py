import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import linear_model, neighbors
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf


def estimate_coefficients(x, y):
    # size of the dataset OR number of observations/points
    n = np.size(x)

    # mean of x and y
    # Since we are using numpy just calling mean on numpy is sufficient
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x - n * mean_y * mean_x)
    SS_xx = np.sum(x * x - n * mean_x * mean_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1 * mean_x

    return (b_0, b_1)

    # x,y are the location of points on graph
    # color of the points change it to red blue orange play around


def plot_regression_line(x, y, b):
    # plotting the points as per dataset on a graph
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels for x and y axis
    plt.xlabel('Dose')
    plt.ylabel('Response')

    # function to show plotted graph
    plt.show()


def plot_error(x, e):
    # plotting the points as per dataset on a graph
    plt.scatter(x, e, color="m", marker="o", s=30)

    # plotting the regression line
    #plt.plot(x, y_pred, color="g")

    # putting labels for x and y axis
    plt.xlabel('Dose')
    plt.ylabel('Error')

    # function to show plotted graph
    plt.show()


def main():

    df = pd.read_csv('./drug2.csv')
    df.drop('sex', axis=1, inplace=True)
    print(df.head())
    print("\nData Description: ")
    print(df.describe())

    df.head()

    print("\nCorrelation:")
    print(df.corr())

    # Linear regression model
    # X = np.array(df.drop(['sex'],1))
    # X = np.array(df['dose'])
    # = np.array(df['response'])
    X = df.iloc[:, :-1].values  # dose
    y = df.iloc[:, 1].values  # response
    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Linear function
    reg_clf = linear_model.LinearRegression()
    # Training
    reg_clf.fit(X_train, y_train)
    # Testing
    """
    # when X = np.array(df.drop(['response'], 1))
    acc = reg_clf.score(X_test, y_test)
    acc = acc * 100
    print('Linear regression accuracy score: ', acc)
    """

    print("\nIntercept: ")
    print(reg_clf.intercept_)
    print("\nCoefficient: ")
    print(reg_clf.coef_)

    results = sm.OLS(y, X).fit()
    print(results.summary())

    # Making predictions
    # estimating coefficients
    # b = estimate_coefficients(X, y)
    b = [reg_clf.intercept_, reg_clf.coef_]
    print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(X, y, b)
    y_pred = b[0] + b[1] * X
    residual_error = y - y_pred
    print(residual_error)
    plot_error(X, residual_error)


if __name__ == "__main__":
    main()
