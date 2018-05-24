"""
Script that loads data from train.csv and cleans it by encoding categorical data
and normalizing numeric data
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize


# Given a Series object of numeric data, normalize it
def process_data(df):
    for col in df.columns:
        print(col, df[col].dtype)
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col] = normalize_numeric_data(df[col])
        else:
            df[col] = encode_categorical_data(df[col])
    return df


# Reshape and normalize numeric data
def normalize_numeric_data(feature):
    feature.dropna(inplace=True)
    temp_feature = np.array(feature.values)
    reshaped_feature = temp_feature.reshape(-1, 1)
    normalized_feature = normalize(reshaped_feature, axis=1, return_norm=True)
    final_feature = pd.Series(normalized_feature)
    return final_feature


# Encode and replace values in categorical data
def encode_categorical_data(feature):
    values = [val for val in feature.unique()]
    encoding = list(range(0, len(values)))
    feature.replace(to_replace=values, value=encoding, inplace=True)
    return feature


def main():
    # Load the file and prepare the data for scoring
    file_path = 'train.csv'
    data = pd.read_csv('train.csv', index_col = 'PassengerId')
    processed_data = process_data(data.drop(['Survived', 'Name', 'Cabin'], axis=1))
    print(processed_data.head(3))

    # Score the features based on univariate predictive power
    selector = SelectKBest(k='all', score_func='f_classif')
    # selector.fit_transform(data.drop(['Survived'], axis=1), data['Survived'])
    # print(selector.scores_)
    return


if __name__ == '__main__':
    main()
