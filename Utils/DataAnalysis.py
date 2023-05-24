import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def FeatureInfo(data: pd.DataFrame=None):
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

  # Plot pair-wise interactions between features with label
  sns.pairplot(data      = data, 
             hue       = 'label',
             palette   = 'Set1',
             kind      = 'scatter', # 'scatter', 'kde', 'hist', 'reg'
             diag_kind = 'hist');   # 'auto', 'hist', 'kde', None
  plt.show()


def T_Test(data: pd.DataFrame=None, categorical_cols: list=None, non_categorical_cols: list=None):
  feature_columns = data.drop(categorical_cols[0], axis=1)
  categorical_column = data[categorical_cols[0]]
  '''
  Perform T-Test to compare the means of the numerical feature column 
  for the two categories of the binary categorical column
  '''
  ttest_results = []
  for column in feature_columns.columns:
      category1 = feature_columns[categorical_column == 0][column]
      category2 = feature_columns[categorical_column == 1][column]
      t_stat, p_value = ttest_ind(category1, category2)
      print(t_stat, p_value)
      ttest_results.append({'Feature': column, 'T-Statistic': t_stat, 'P-Value': p_value})

  ttest_results = pd.DataFrame(ttest_results)

  plt.figure(figsize=(10, 6))
  sns.barplot(x='Feature', y='T-Statistic', data=ttest_results, color='blue', alpha=0.5, label='T-Statistic')
  plt.xticks(rotation=90)
  plt.xlabel('Feature')
  plt.ylabel('Value')
  plt.title('T-Test Results: Feature Columns vs. Categorical Column')
  plt.legend()
  plt.show()


def checkOutliers(data: pd.DataFrame=None, numerical_columns:list=None):
  '''
  Checking for outliers in each feature
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
    unique_values = column.unique()
    num_unique_values = len(unique_values)
    num_total_values = len(column)
    '''
    If the number of unique values is relatively small 
    compared to the total values, it's likely categorical
    '''
    threshold = 0.05  # threshold based on dataset

    if num_unique_values / num_total_values <= threshold:
        return True
    else:
        return False

def checkCategoricalColumns(data: pd.DataFrame=None) -> '''tuple[list, list]''':
    categorical_columns = []
    non_categorical_columns = []

    for column in data.columns:
        is_cat = isCategorical(data[column])
        if is_cat:
            categorical_columns.append(column)
        else:
            non_categorical_columns.append(column)

    return categorical_columns, non_categorical_columns