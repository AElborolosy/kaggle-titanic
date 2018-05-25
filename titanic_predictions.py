"""
Make predictions using a pre-existing logistic regression model.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    # Load the data and the model
    data = pd.read_csv('testing_data.csv', index_col='PassengerId')
    model = pickle.load(open('titanic_model.pickle', 'rb'))

    # Generate the list of passenger IDs
    PassengerId = np.arange(892, 892 + len(data))
    PId = pd.Series(PassengerId)

    # Make predictions
    predictions = model.predict(data)
    pred = pd.Series(predictions)
    results = PId.to_frame(name='PassengerId')
    results['Survived'] = pred
    results.to_csv('submission.csv', index=False)
    return


if __name__ == '__main__':
    main()
