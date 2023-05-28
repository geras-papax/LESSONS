import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def FeatureInfo(data: pd.DataFrame=None):
  
    '''
    Display feature information and visualizations.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).

    Returns
    -------
    None
        This function does not return any value. It displays visualizations.

    '''

    categorical_cols, non_categorical_cols = checkCategoricalColumns(data)
    # Get statistics of the features
    statistics = data.describe()
    print(statistics)

    # Number of rows and columns for the grid
    num_features = len(data.columns) - 1  
    num_rows = (num_features - 1) // 3 + 1
    num_cols = min(3, num_features)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 16))
    axes = axes.flatten()

    # Distribution of each feature
    for i, column in enumerate(data.columns):
        if column not in categorical_cols:  
            sns.histplot(data[column], kde=True, ax=axes[i])
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {column}')

    plt.tight_layout()
    plt.show()

    # Label's class distribution

    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='label')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

    #   # Plot pair-wise interactions between features with label
    #   sns.pairplot(data      = data, 
    #              hue       = 'label',
    #              palette   = 'Set1',
    #              kind      = 'scatter', # 'scatter', 'kde', 'hist', 'reg'
    #              diag_kind = 'hist');   # 'auto', 'hist', 'kde', None
    #   plt.show()


def checkOutliers(data: pd.DataFrame=None, numerical_columns:list=None):
    '''
    Check for outliers in each numerical feature and visualize the results.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).
    numerical_columns : list, optional
        List of numerical columns to check for outliers (default: None).

    Returns
    -------
    None
        This function does not return any value. It displays visualizations.

    '''
    plt.figure(figsize=(8, 3 * len(numerical_columns)))
    for i, column in enumerate(numerical_columns):
        plt.subplot(len(numerical_columns), 2, 2*i+1)
        sns.boxplot(x=data[column])
        plt.xlabel(column)
        plt.title('Boxplot - {}'.format(column))

        plt.subplot(len(numerical_columns), 2, 2*i+2)
        sns.histplot(data[column], kde=False)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title('Distribution Plot - {}'.format(column))

    plt.tight_layout()
    plt.show()


def isCategorical(column: pd.DataFrame=None) -> bool:
    '''
    Check if a column is categorical based on the number of unique values.

    Parameters
    ----------
    column : pandas.DataFrame, optional
        Input column for which to check if it is categorical (default: None).

    Returns
    -------
    bool
        True if the column is determined to be categorical, False otherwise.

    '''
    unique_values = column.unique()
    num_unique_values = len(unique_values)
    num_total_values = len(column)
    
    threshold = 0.05  # threshold based on dataset

    if num_unique_values / num_total_values <= threshold:
        return True
    else:
        return False

def checkCategoricalColumns(data: pd.DataFrame=None) -> '''tuple[list, list]''':
    '''
    Check categorical and non-categorical columns in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame to check for categorical columns (default: None).

    Returns
    -------
    tuple
        A tuple containing two lists: 
        - The first list contains the names of the categorical columns.
        - The second list contains the names of the non-categorical columns.

    '''
    categorical_columns = []
    non_categorical_columns = []

    for column in data.columns:
        is_cat = isCategorical(data[column])
        if is_cat:
            categorical_columns.append(column)
        else:
            non_categorical_columns.append(column)

    return categorical_columns, non_categorical_columns