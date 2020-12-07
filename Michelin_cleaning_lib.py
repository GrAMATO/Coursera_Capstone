"""This library contain functions to clean the data"""

import numpy as np
import pandas as pd

def import_data():
    return [pd.read_csv("one-star.csv"), pd.read_csv("two-stars.csv"),
    pd.read_csv("three-stars.csv")]

def full_data():
    """Get the full dataframe"""
    one_star, two_stars, three_stars = import_data()
    one_star = add_star_col(one_star, 1)
    two_stars = add_star_col(two_stars, 2)
    three_stars = add_star_col(three_stars, 3)
    return pd.concat([one_star, two_stars, three_stars]).reset_index(drop = True)

def replace_dollar(data):
    """Replace $ by real categorical value"""
    data["price"] = data["price"].replace("$", "Cheap").replace("$$", "Mid-cheap").replace("$$$", "Medium").replace("$$$$", "Mid-expensive").replace("$$$$$", "Expensive")
    return data

def one_hot_data(data, variable):
    """Apply a one hot encoding on a variable from the dataframe, then merge the dummies df with the old df"""
    one_hot_cuisine_df = pd.get_dummies(data[variable])
    del(data[variable])
    return data.join(one_hot_cuisine_df)

def one_hot_data_pipe(data):
    """Apply a one hot encoding on a variable from the dataframe, then merge the dummies df with the old df, this is the pipe version"""
    variables = ["cuisine","price","region"]
    for variable in variables:
        one_hot_cuisine_df = pd.get_dummies(data[variable])
        del(data[variable])
        data = data.join(one_hot_cuisine_df)
    return data

def multiple_del(data, list_del):
    """Delete all variables in data, which are listed in list_del"""
    for i in list_del:
        del(data[i])
        
def multiple_del_pipe(data):
    """Delete all variables in data, which are listed in list_del, this is the pipe version"""
    list_del = ["year", "url"]
    for i in list_del:
        del(data[i])  
    return data
        
def add_star_col(data, nb_stars=1): 
    """Add a column to data named Star with nb_stars as int, the number of star of each restaurant"""
    x = np.array([nb_stars])
    Star = pd.DataFrame(np.repeat(x, len(data)), columns = ["Star"])
    return data.join(Star)

def clean_group_venues(venues):
    # one hot encoding
    onehot = pd.get_dummies(venues[['Venue Category']], prefix="", prefix_sep="")

    # add neighborhood column back to dataframe
    onehot['Neighborhood'] = venues['Neighborhood'] 

    # move neighborhood column to the first column
    fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
    onehot = onehot[fixed_columns]

    return onehot.groupby('Neighborhood').mean().reset_index()