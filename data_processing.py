"""
Script that loads data from train.csv and cleans it by encoding categorical data
and normalizing numeric data
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


def process_data(df):
    """
    Noramlizes/encodes numeric/categorical data in the given df.
    Uses two helper functions (normalize_numeric_data & encode_categorical_data)
    """
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col] = normalize_numeric_data(df[col])
        else:
            df[col] = encode_categorical_data(df[col])
    return df


def normalize_numeric_data(feature):
    """
    Helper function for process_data. Normalizes numeric data,
    centered on 0 with range of [-1, 1]
    """
    # Reshape data for imputing and normalizing
    feature = feature.values.reshape(-1, 1)

    # Replace null values with the median
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    feature = imp.fit_transform(feature)

    # Normalize the data
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    return feature


def encode_categorical_data(feature):
    """
    Helper function for process_data
    Encode and replace values in categorical data
    """
    # Store list of unique values and replace them accordingly
    values = [val for val in feature.unique()]
    encoding = list(range(0, len(values)))
    feature.replace(to_replace=values, value=encoding, inplace=True)
    return feature


def main():
    # Load the file and prepare the data for scoring
    file_path = 'train.csv'
    data = pd.read_csv(file_path, index_col = 'PassengerId')
    p_data = process_data(data.drop(['Survived', 'Name', 'Cabin'], axis=1))
    p_data['Survived'] = data['Survived']
    p_data.to_csv('testing_data.csv')
    return


if __name__ == '__main__':
    main()
