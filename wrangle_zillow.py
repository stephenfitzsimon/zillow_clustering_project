#ziilow wrangle module
#stephen fitzsimon

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from env import get_db_url

RAND_SEED = 739
FILENAME = 'zillow_clustering_data.csv'

def wrangle_data(df):
    df = filter_properties(df)
    df = handle_missing_values(df, 0.01, 0.01)
    df = clearing_fips(df)
    df['latitude'] = df['latitude']*10**-6
    df['longitude'] = df['longitude']*10**-6
    df.drop(columns=['calculatedbathnbr', 'fullbathcnt', 'finishedsquarefeet12', 'assessmentyear', 'roomcnt'], inplace=True)
    return df

def get_zillow_data(query_db=False):
    '''Acquires the zillow data from the database or the .csv file if if is present

    Args:
        query_db = False (Bool) :  Forces a databse query and a resave of the data into a csv.
    Return:
        df (DataFrame) : a dataframe containing the data from the SQL database or the .csv file
    '''
    #file name string literal
    #check if file exists and query_dg flag
    if os.path.isfile(FILENAME) and not query_db:
        #return dataframe from file
        print('Returning saved csv file.')
        return pd.read_csv(FILENAME).drop(columns = ['Unnamed: 0'])
    else:
        #query database 
        print('Querying database ... ')
        query = '''
        SELECT predictions_2017.logerror, e.transdate, 
                properties_2017.*, 
                typeconstructiontype.typeconstructiondesc, 
                storytype.storydesc, 
                propertylandusetype.propertylandusedesc, 
                heatingorsystemtype.heatingorsystemdesc, 
                airconditioningtype.airconditioningdesc, 
                architecturalstyletype.architecturalstyledesc, 
                buildingclasstype.buildingclassdesc
            FROM (SELECT max(transactiondate) AS transdate, parcelid FROM predictions_2017 GROUP BY parcelid) AS e
                JOIN predictions_2017 ON predictions_2017.transactiondate = e.transdate AND predictions_2017.parcelid = e.parcelid
                JOIN properties_2017 ON properties_2017.parcelid = predictions_2017.parcelid
                LEFT JOIN airconditioningtype USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
                LEFT JOIN storytype USING (storytypeid)
                LEFT JOIN typeconstructiontype USING (typeconstructiontypeid);
        '''
        #get dataframe from a 
        df = pd.read_sql(query, get_db_url('zillow'))
        print('Got data from the SQL database')
        #save the dataframe as a csv
        df.to_csv(FILENAME)
        print('Saved dataframe as a .csv!')
        #return the dataframe
        return df

def return_col_percent_null(df, max_null_percent = 1.0):
    '''Returns a dataframe with columns of the column of df, the percent nulls in the column, and the count of nulls.

    Args:
        df (dataframe) : a dataframe 
        max_null_percent = 1.0 (float) : returns all columns with percent nulls less than max_null_percent
    Return:
        (dataframe) : dataframe returns with df column names, percent nulls, and null count
    '''
    outputs = [] #to store output
    for column in df.columns: #loop through the columns
        #store and get information
        output = {
            'column_name': column,
            'percent_null' : round(df[column].isna().sum()/df[column].shape[0], 4),
            'count_null' : df[column].isna().sum()
        }
        #append information
        outputs.append(output)
    #make a dataframe
    columns_percent_null = pd.DataFrame(outputs)
    #return the dataframe with the max_null_percent_filter
    return columns_percent_null[columns_percent_null.percent_null <= max_null_percent]

def handle_missing_values(df, prop_required_column, prop_required_row):
    '''Returns a dataframe that is filtered by the percent of non-null values in the columns and the rows
    
    Args:
        df (dataframe) : a zillow dataframe
        prop_required_column (float) : the percent of the column values that are not null
        prop_required_row (float) : the percent of the row values that are not null
    Returns:
        df (dataframe) : a filtered dataframe
    '''
    #get the proportion of nulls in each column
    null_proportion_df = return_col_percent_null(df)
    #get the columns to keep
    columns_to_keep = null_proportion_df[null_proportion_df['percent_null'] < (prop_required_column)]['column_name'].tolist()
    # get the columns from the dataframe
    df = df[columns_to_keep]
    #filter the rows
    df = df[(df.isnull().sum(axis=1)/df.shape[1] < (prop_required_row))]
    return df

def filter_properties(df):
    filter_cols = ['Single Family Residential', 'Mobile Home', 'Manufactured, Modular, Prefabricated Homes', 'Residential General', 'Townhouse']
    df = df[df['propertylandusedesc'].isin(filter_cols)]
    return df

def clearing_fips(df):
    '''This function takes in a DataFrame of unprepared Zillow information and generates a new
    'county' column, with the county name based on the FIPS code. Drops the 'fips' column and returns
    the new DataFrame.

    Args:
        df (Dataframe) : dataframe containing zillow data
    Return:
        (DataFrame) : a dataframe with a cleared fips columns
    '''
    # create a list of our conditions
    fips = [
        (df['fips'] == 6037.0),
        (df['fips'] == 6059.0),
        (df['fips'] == 6111.0)
        ]
    # create a list of the values we want to assign for each condition
    counties = ['Los Angeles', 'Orange', 'Ventura']
    # create a new column and use np.select to assign values to it using our lists as arguments
    df['county'] = np.select(fips, counties)
    df = df.drop(columns = 'fips')
    return df

def split_data(df, return_info=False):
    '''splits the zillow dataframe into train, test and validate subsets
    
    Args:
        df (DataFrame) : dataframe to split
        return_info (bool) : returns a dataframe with the number of rows and columns of the returned dataframes.
    Return:
        train, test, validate (DataFrame) :  dataframes split from the original dataframe
    '''
    #make train and test
    train, test = train_test_split(df, train_size = 0.8, random_state=RAND_SEED)
    #make validate
    train, validate = train_test_split(train, train_size = 0.7, random_state=RAND_SEED)
    #get info about the dataframes
    train_info = {
        'subset':'train',
        'rows': train.shape[0],
        'columns': train.shape[1]
    }
    validate_info = {
        'subset':'validate',
        'rows': validate.shape[0],
        'columns': validate.shape[1]
    }
    test_info = {
        'subset':'test',
        'rows': test.shape[0],
        'columns': test.shape[1]
    }
    #put info into a dataframe
    df_info = pd.DataFrame([train_info, validate_info, test_info])
    # get appropriate return statement
    if return_info:
        return train, validate, test, df_info
    else:
        return train, validate, test

def scale_and_encode(df, columns_to_encode = None, columns_to_scale=None):
    '''Scales and encodes the column lists passed to the function'''
    # if nothing is passed, then do not call the function
    if columns_to_encode is not None:
        df = encode_columns(df, columns_to_encode)
    if columns_to_scale is not None:
        df = zillow_scale(df, columns_to_scale)
    return df

def year_cats_non_geo(df):
    '''Bins the year column into categories'''
    #make bins
    year_bins = [1870, 1940, 1975, 1980, 2020]
    #make column with categorical variable
    df['year_bin'] = pd.cut(df['yearbuilt'], year_bins, labels=['old_homes', 'war_post_war_homes', 'eighties_homes', 'new_homes'])
    return df

def year_cats_geo(df):
    '''Bins the year column into categories'''
    #make bins
    year_bins = [1870, 1945, 1970, 1985, 2020]
    #make column with categorical variable
    df['year_bin'] = pd.cut(df['yearbuilt'], year_bins, labels=['war_pre_war_homes', 'post_war_homes', 'eight_to_sixties_homes', 'new_homes'])
    return df

def encode_columns(df,
                    column_names):
    '''encodes columns as passed in column_names'''
    #make dummies
    dummy_df = pd.get_dummies(df[column_names], drop_first=True)
    #add to the existing dataframe
    df = pd.concat([df, dummy_df], axis=1).drop(columns = column_names)
    return df

def make_geo_cluster(df):
    '''adds column for the non-geo clusters'''#subset to the columns to cluster
    df_sub = df[['parcelid','logerror','county', 'latitude', 'longitude', 'yearbuilt', 'taxvaluedollarcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    #scale and encode the data
    df_sub_scaled = scale_and_encode(df_sub, columns_to_encode=['county'], columns_to_scale=['latitude', 'longitude', 'yearbuilt', 'taxvaluedollarcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet'])
    X = df_sub_scaled #['yearbuilt_scaled', 'taxvaluedollarcnt_scaled', 'bathroomcnt_scaled', 'calculatedfinishedsquarefeet_scaled']
    kmeans = KMeans(n_clusters=6, random_state=RAND_SEED)
    kmeans.fit(X)
    #add to the dataframe
    df['cluster_geo'] = kmeans.predict(X)
    return df

def make_non_geo_cluster(df):
    '''adds column for the non-geo clusters'''#subset to the columns to cluster
    df_sub = df[['parcelid','logerror', 'yearbuilt', 'taxvaluedollarcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    #scale and encode the data
    df_sub_scaled = scale_and_encode(df_sub, columns_to_encode=None, columns_to_scale=['yearbuilt', 'taxvaluedollarcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet'])
    X = df_sub_scaled #['yearbuilt_scaled', 'taxvaluedollarcnt_scaled', 'bathroomcnt_scaled', 'calculatedfinishedsquarefeet_scaled']
    kmeans = KMeans(n_clusters=6, random_state=RAND_SEED)
    kmeans.fit(X)
    #add to the dataframe
    df['cluster'] = kmeans.predict(X)
    return df

def zillow_scale(df,
                column_names,
                scaler_in=MinMaxScaler(),
                return_scalers=False):
    '''
    Returns a dataframe of the scaled columns
    
    Args:
        df (DataFrame) : The dataframe with the columns to scale
        column_names (list) : The columns to scale
        scaler_in (sklearn.preprocessing) : scaler to use, default = MinMaxScaler()
        return_scalers (bool) : boolean to return a dictionary of the scalers used for 
            the columns, default = False
    Returns:
        df_scaled (DataFrame) : A dataframe containing the scaled columns
        scalers (dictionary) : a dictionary containing 'column' for the column name, 
            and 'scaler' for the scaler object used on that column
    '''
    #variables to hold the returns
    scalers = []
    df_scaled = df[column_names]
    for column_name in column_names:
        #determine the scaler
        scaler = scaler_in
        #fit the scaler
        scaler.fit(df[[column_name]])
        #transform the data
        scaled_col = scaler.transform(df[[column_name]])
        #store the column name and scaler
        scaler = {
            'column':column_name,
            'scaler':scaler
        }
        scalers.append(scaler)
        #store the transformed data
        df[f"{column_name}_scaled"] = scaled_col
    #determine the correct varibales to return
    if return_scalers:
        return df.drop(columns = column_names), scalers
    else:
        return df.drop(columns = column_names)

def make_X_and_y(df,
                target_column = 'logerror'):
    '''Makes a X and y sets based on the target column passed as a list'''
    #drop relevant columns
    X_train = df.drop(columns = [target_column])
    #make y_Train
    y_train = df[['parcelid', target_column]]
    return X_train, y_train