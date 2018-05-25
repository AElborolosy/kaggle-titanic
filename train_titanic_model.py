"""
Train and save a logistic regression model for the titanic data"
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Train the logistic regression model
def train_log_reg(X, y):
    # Set a random seed and reshape the data
    np.random.seed(42)
    y.ravel()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Initialize and train the model
    log_reg = LogisticRegression(verbose=0, max_iter=200)
    log_reg.fit(X_train, y_train)

    # Evaluate the model using the test set
    performance = 100 *log_reg.score(X_test, y_test)
    print("Accuracy: %.2f%%" % performance)

    predictions = log_reg.predict(X_test)
    print(pd.crosstab(predictions, y_test))
    return log_reg


def main():
    data = pd.read_csv('training_data.csv', index_col='PassengerId')
    model = train_log_reg(data.drop(['Survived'], axis=1), data['Survived'])
    with open('titanic_model.pickle', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    return


if __name__ == '__main__':
    main()
