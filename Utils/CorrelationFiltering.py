import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from ennemi import estimate_mi
from ennemi import pairwise_mi

def calcDrop(res:pd.DataFrame=None) -> list:
    '''
    Calculate which variables to drop based on the correlation analysis.

    Parameters
    ----------
    res : pandas.DataFrame, optional
        DataFrame containing the correlation analysis results (default: None).

    Returns
    -------
    list
        A list of column names to be dropped based on the correlation analysis.

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
    Identify highly correlated variables and recommend columns to drop.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).
    cut : int, optional
        Threshold for correlation value (default: 0.9).

    Returns
    -------
    list
        A list of column names to be dropped based on high correlation.

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
    Calculate and visualize the Pearson's correlation matrix.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).

    Returns
    -------
    None
        This function does not return any value. It displays the Pearson's correlation heatmap.

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
    '''
    Calculate and visualize the Phi_k correlation matrix.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).

    Returns
    -------
    None
        This function does not return any value. It displays the Phi_k correlation heatmap.

    '''
    # Calculate phi_k correlation matrix
    correlation_matrix = data.iloc[:, :-1].phik_matrix()

    plt.figure(figsize=(19, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Phi_k's Correlation Heatmap")
    plt.show()


def MutualInformation(data:pd.DataFrame=None):
    '''
    Calculate and visualize mutual information between variables.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).

    Returns
    -------
    None
        This function does not return any value. It displays the mutual information results.

    '''
    # Compare target variable against all other variables
    estimate = estimate_mi(y= data['label'].values, 
                        x= data.iloc[:,:-1],
                        normalize = True)
    # Pairwise comparisons between a set of variables
    pairwise = pairwise_mi( data, normalize=True )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=estimate, orient='h')
    # Visualize Mutual Information matrix
    plt.figure(figsize = (22, 15))

    sns.heatmap(pairwise, annot=True, cmap="coolwarm")

    plt.title('Mutual Information matrix showing correlation coefficients')
    plt.show()

def FeatureSelection(data: pd.DataFrame=None, num: int=5) :
    '''
    Perform feature selection using mutual information and phi_k correlation methods.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame containing the data (default: None).
    num : int, optional
        Number of top features to select (default: 5).

    Returns
    -------
    None
        This function does not return any value. It displays the top features selected by each method.

    '''
    # Separate the features and target variable
    X = data.iloc[:, :-1]  # Features
    y = data['label']  # Target variable

    # Select the number of top features to keep
    k = num

    # Perform feature selection using mutual information
    mutual_info_scores = estimate_mi(y=y, x= X, normalize = True)
    top_mutual_info_indices = mutual_info_scores.values[0].argsort()[-k:][::-1]
    top_mutual_info_features = X.columns[top_mutual_info_indices]

    # Perform feature selection using fi-correlation
    phik_correlation_scores = data.phik_matrix()['label'].drop('label')
    top_correlation_indices = phik_correlation_scores.argsort()[-k:][::-1]
    top_correlation_features = X.columns[top_correlation_indices]

    # Print the top features selected by each method
    print("Top", k, "features selected by Mutual Information:", top_mutual_info_features)
    print("Top", k, "features selected by Phi_k Correlation:", top_correlation_features)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=pd.DataFrame(phik_correlation_scores).T, orient='h')
    plt.title('Phi_k Correlation on Label')
    plt.show()

