import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import linear_model, neighbors
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf


def plot_regression_line(x, y, b, x_label, y_label):
    # plotting the points as per dataset on a graph
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels for x and y axis
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # function to show plotted graph
    plt.show()


def plot_error(error):
    plt.hist(np.asarray(error, dtype='float'), bins=30)  # plot of frequency of errors vs range of errors.
    plt.ylabel('Probability')
    plt.xlabel('Bins')

    # function to show plotted graph
    plt.show()


def main():

    df = pd.read_csv('./dmf.csv')
    print(df.head())
    print("\nData Description: ")
    print(df.describe())

    df.head()

    print("\nCorrelation:")
    print(df.corr())

    # Linear regression model
    X = df.iloc[:, :-1].values  # dmf
    y = df.iloc[:, 1].values  # flor
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
    #plot_regression_line(X, y, b, 'dmf', 'flor')
    y_pred = b[0] + b[1] * X
    print(y_pred)
    residual_error = y - y_pred
    print(residual_error)
    #plot_error(residual_error)
    # plot_error(X, residual_error)


if __name__ == "__main__":
    main()
