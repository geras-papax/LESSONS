import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from ennemi import estimate_mi
from ennemi import pairwise_mi

def calcDrop(res:pd.DataFrame=None) -> list:
    '''
        Calculate which variables to drop
    '''
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
    
    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))
     
    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop 
    poss_drop = list(set(poss_drop).difference(set(drop)))
    
    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        
    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)
         
    return drop



def corrX(df:pd.DataFrame=None, cut:int=0.9) -> list:
    '''
        Log the variable states based on the original logic
    '''

    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))
    
    dropcols = list()
    
    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 'v2.target','corr', 'drop' ]))
    
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else: 
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]
                
                s = pd.Series([ corr_mtx.index[row],
                up.columns[col],
                avg_corr[row],
                avg_corr[col],
                up.iloc[row,col],
                drop],
                index = res.columns)
        
                res = res.append(s, ignore_index = True)
    
    dropcols_names = calcDrop(res)
    
    return(dropcols_names)


'''
Example - How to use

df = pd.read_csv( 'Data/boston_corrected.txt' )

corrX_new(df, cut = 0.7)

'''

def PearsonCorrelation(data:pd.DataFrame=None):
  '''
  Calculate correlation between feature columns
  '''
  correlation_matrix = abs(data.iloc[:, :-1].corr())

  # Set the range between 0 and 1
  # correlation_matrix = (correlation_matrix + 1)/2
  # Visualize the correlations
  plt.figure(figsize=(19, 15))
  sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
  plt.title("Pearson's Correlation Heatmap")
  plt.show()

def FiCorrelation(data:pd.DataFrame=None):
<<<<<<< Updated upstream
    X = data.iloc[:,:-1]  # Features
    y = data['label']  # Labels

    # Calculate feature importance correlation using f_classif
    f_scores, p_values = f_classif(X, y)

    # Create a DataFrame to store the feature importance correlation results
    feature_importance = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores, 'p-value': p_values})
=======
>>>>>>> Stashed changes

    # Calculate phi_k correlation matrix
    correlation_matrix = data.iloc[:, :-1].phik_matrix()

<<<<<<< Updated upstream
    # Plot the feature importance correlation results
    plt.figure(figsize=(12, 7))
    sns.barplot(data=feature_importance, x='F-Score', y='Feature')
    plt.xlabel('F-Score')
    plt.ylabel('Feature')
    plt.title('Feature Importance Correlation')
    plt.tight_layout()
=======
    plt.figure(figsize=(19, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Phi_k's Correlation Heatmap")
>>>>>>> Stashed changes
    plt.show()


def MutualInformation(data:pd.DataFrame=None):
    # Compare target variable against all other variables
    #
    estimate = estimate_mi(y= data['label'].values, 
                        x= data.iloc[:,:-1],
                        normalize = True)
    # Pairwise comparisons between a set of variables
    #
    pairwise = pairwise_mi( data, normalize=True )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=estimate, orient='h')
    # Visualize Mutual Information matrix
    #
    plt.figure(figsize = (22, 15))

    sns.heatmap(pairwise, annot=True, cmap="coolwarm")

    plt.title('Mutual Information matrix showing correlation coefficients')
    plt.show()


def CorrelatingFiltering(data: pd.DataFrame=None, correlation_matrix: pd.DataFrame=None) -> '''tuple[pd.DataFrame, bool]''':
  # Avg correlation for each feature
  average_correlations = correlation_matrix.abs().mean()

  # Highest avg correlation
  feature_with_max_average_correlation = average_correlations.idxmax()
  max_average_correlation_value = average_correlations.max()

  print("Feature with the highest average correlation:", feature_with_max_average_correlation)
  print("Max average correlation value:", max_average_correlation_value)

  # Get the columns with correlation lower than 0.8 with the specific column
  low_correlation_columns = correlation_matrix[(correlation_matrix[feature_with_max_average_correlation] < 0.8) | (correlation_matrix[feature_with_max_average_correlation] == 1)].index

  # Last column with labels
  low_correlation_columns = low_correlation_columns.append(pd.Index([data.columns[-1]]))
  
  # Filter the columns and keep the desired ones
  data_filtered = data[low_correlation_columns]

  # Confirm the dropped columns
  dropped_columns = data.columns.difference(data_filtered.columns)
  if dropped_columns.empty:
    return data, False
  else:
    print("Dropped columns:", dropped_columns)
    return data_filtered, True

<<<<<<< Updated upstream
def FeatureSelection(data: pd.DataFrame=None) :
=======
def FeatureSelection(data: pd.DataFrame=None, num: int=5) :
>>>>>>> Stashed changes
    # Separate the features and target variable
    X = data.iloc[:, :-1]  # Features
    y = data['label']  # Target variable

    # Select the number of top features to keep
<<<<<<< Updated upstream
    k = 5

    # Perform feature selection using mutual information
    # mutual_info_scores = estimate_mi(y=y, x= X, normalize = True)
    # print(mutual_info_scores.values)
    # top_mutual_info_indices = mutual_info_scores.values.argsort()[:k][::-1]
    # top_mutual_info_features = X.columns[top_mutual_info_indices]

    # Perform feature selection using fi-correlation
    correlation_scores, p_values = f_classif(X, y)
    top_correlation_indices = correlation_scores.argsort()[-k:][::-1]
=======
    k = num

    # Perform feature selection using mutual information
    mutual_info_scores = estimate_mi(y=y, x= X, normalize = True)
    top_mutual_info_indices = mutual_info_scores.values[0].argsort()[-k:][::-1]
    top_mutual_info_features = X.columns[top_mutual_info_indices]

    # Perform feature selection using fi-correlation
    phik_correlation_scores = data.phik_matrix()['label'].drop('label')
    top_correlation_indices = phik_correlation_scores.argsort()[-k:][::-1]
>>>>>>> Stashed changes
    top_correlation_features = X.columns[top_correlation_indices]

    # Perform feature selection using Pearson correlation
    pearson_correlation_scores = abs(data.corr(method='pearson')['label']).drop('label')
    top_pearson_correlation_indices = pearson_correlation_scores.argsort()[-k:][::-1]
    top_pearson_correlation_features = X.columns[top_pearson_correlation_indices]

    # Print the top features selected by each method
<<<<<<< Updated upstream
    # print("Top", k, "features selected by Mutual Information:", top_mutual_info_features)
    print("Top", k, "features selected by Correlation:", top_correlation_features)
    print("Top", k, "features selected by Pearson Correlation:", top_pearson_correlation_features)
=======
    print("Top", k, "features selected by Mutual Information:", top_mutual_info_features)
    print("Top", k, "features selected by Phi_k Correlation:", top_correlation_features)
    print("Top", k, "features selected by Pearson Correlation:", top_pearson_correlation_features)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=pd.DataFrame(pearson_correlation_scores).T, orient='h')
    plt.title('Pearson Correlation on Label')
    plt.show()
    plt.figure(figsize=(12, 7))
    sns.barplot(data=pd.DataFrame(phik_correlation_scores).T, orient='h')
    plt.title('Phi_k Correlation on Label')
    plt.show()
>>>>>>> Stashed changes
