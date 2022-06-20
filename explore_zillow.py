#zillow explore module
#stephen fitzsimon

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations, product
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression

ALPHA = 0.05

def return_iqr_outlier_tuple(df, column, k = 1.5):
    '''compute the IQR based off of a passed k value'''
    #get Q3
    upper_limit = df[column].describe()['75%']
    #get Q1
    lower_limit = df[column].describe()['25%']
    # get IGR range
    iqr = upper_limit - lower_limit
    #return tuple with range and multiple of iqr
    return (lower_limit-iqr*k, upper_limit+iqr*k)

def plot_histogram(df, column_name, title_str='', hue_column=None):
    '''Plots a histogram of a column from a dataframe'''
    #set size
    sns.set(rc={"figure.figsize":(15, 6)}) 
    #make plot
    sns.histplot(data=df, x=column_name, hue=hue_column)
    #set title
    plt.title(title_str)
    #show plot
    plt.show()

def plot_boxplot(df, column_name, title_str=''):
    ''' Plots a boxplot of a column from a dataframe '''
    #set size
    sns.set(rc={"figure.figsize":(15, 6)}) 
    #make plot
    sns.boxplot(data=df, x=column_name)
    #set title
    plt.title(title_str)
    #show plot
    plt.show()

def plot_variable_pairs(df,
                        columns_x = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt','taxvaluedollarcnt', 'lotsizesquarefeet', 'latitude', 'longitude'],
                        columns_y = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt','taxvaluedollarcnt', 'lotsizesquarefeet', 'latitude', 'longitude'],
                        sampling = 1000):
    '''plots a lmplot plot of all pairs of all columns passed in two lists'''
    #make the pairs
    pairs = product(columns_x, columns_y)
    for pair in pairs:
        #make a plot for every pair
        sns.lmplot(x=pair[0], y=pair[1], data=df.sample(sampling), line_kws={'color': 'red'})
        plt.show()

def plot_categorical_and_continuous_vars(df,
                                         columns_cat=['county'],
                                         columns_cont=['calculatedfinishedsquarefeet', 'yearbuilt', 'bedroomcnt', 'bathroomcnt', 'taxvaluedollarcnt', 'latitude', 'longitude'],
                                         sampling = 1000):
    '''plots a strip plot, a box plot, and a barplot for all the combinations passed
    from columns_cat, and columns_cont'''
    #make all the pairs
    pairs = product(columns_cat, columns_cont)
    for pair in pairs:
        #set up for subplots
        sns.set(rc={"figure.figsize":(15, 6)}) 
        fig, axes = plt.subplots(1, 3)

        #make the plots 
        sns.stripplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[0])
        sns.boxplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[1])
        sns.barplot(x=pair[0], y=pair[1], data=df.sample(sampling), ax = axes[2])
        plt.show

def r_values_vars(df,
                columns = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet','yearbuilt', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'latitude', 'longitude']):
    '''performs a correlation test for all pairs of columns passed
    returns a dataframe with results'''
    #make all the pairs
    pairs = combinations(columns, 2)
    outputs = []
    #perform a pearson r test on all the pairs
    for pair in pairs:
        #perform the test
        corr, p = stats.pearsonr(df[pair[0]], df[pair[1]])
        #store the output
        output = {
            'correlation':f"{pair[0]} x {pair[1]}",
            'r' : corr,
            'p-value' : p,
            'reject_null' : p < ALPHA
        }
        outputs.append(output)
    # return the results
    corr_tests = pd.DataFrame(outputs)
    return corr_tests

def t_test_by_cat(df,
                columns_cat,
                columns_cont
                ):
    '''Performs a t-test for all subcategories of columns_cat and paored with every column in columns cat
    returns results as a dataframe'''
    outputs = []
    pairs_by_cat = {}
    #get pairs for every sub_Cat
    for category in columns_cat:
        #get subcategory names
        subcats = df[category].unique().tolist()
        #make the pairs
        pairs = list(product(subcats, columns_cont))
        pairs_by_cat[category] = pairs
    for category in columns_cat:
        pairs = pairs_by_cat[category]
        for pair in pairs:
            #subset into county_x and not county_x
            category_x = df[df[category] == pair[0]][pair[1]]
            not_category_x = df[~(df[category] == pair[0])][pair[1]].mean()
            #do the stats test
            t, p = stats.ttest_1samp(category_x, not_category_x)
            output = {
                'category_name':pair[0],
                'column_name':pair[1],
                't-test':t,
                'p-value':p,
                'reject_null': p < ALPHA
            }
            outputs.append(output)
    #return as a dataframe
    return pd.DataFrame(outputs)

def t_test_by_cat_greater(df,
                columns_cat,
                columns_cont,
                ):
    '''Performs a t-test for all subcategories of columns_cat and paored with every column in columns cat
    returns results as a dataframe'''
    outputs = []
    pairs_by_cat = {}
    #get pairs for every sub_Cat
    for category in columns_cat:
        #get subcategory names
        subcats = df[category].unique().tolist()
        #make the pairs
        pairs = list(product(subcats, columns_cont))
        pairs_by_cat[category] = pairs
    for category in columns_cat:
        pairs = pairs_by_cat[category]
        for pair in pairs:
            #subset into county_x and not county_x
            category_x = df[df[category] == pair[0]][pair[1]]
            not_category_x = df[~(df[category] == pair[0])][pair[1]].mean()
            #do the stats test
            t, p = stats.ttest_1samp(category_x, not_category_x)
            output = {
                'category_name':pair[0],
                'column_name':pair[1],
                't-stat':t,
                'p-value':p,
                'reject_null': p/2 < ALPHA and t > 0
            }
            outputs.append(output)
    #return as a dataframe
    return pd.DataFrame(outputs)

def t_test_greater(df, column_cat, subcat_val, column_cont):
    '''Perform a t-test that mean is greater than pop'''
    #get subsets
    category_x = df[df[column_cat] == subcat_val][column_cont]
    not_category_x = df[~(df[column_cat] == subcat_val)][column_cont].mean()
    #perform test
    t, p = stats.ttest_1samp(category_x, not_category_x)
    #organize results
    output = {
        'category_name':column_cat,
        'category_value':subcat_val,
        't-stat': t,
        'p-value':p,
        'reject_null': p/2 < ALPHA and t > 0
    }
    #return results
    return pd.DataFrame([output])