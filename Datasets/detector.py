# this code will identify whether the dataset is for regression or classification 

import pandas as pd
import numpy as np

def detect_problem_type(y , threshold = 15):
    '''this function will help to classify the data as regression or classification
    
    Rules:
    --> if the dtype is object or category then its classification
    --> if unique values but small then again it would be classification
    --> else , it would be regression

    threshold : below which unique value is treated to be classification 
    
    '''

    # convert into pandas dataframe if not already there
    if not isinstance(y , pd.Series):
        y = pd.Series(y)

    # CASE:1 , check whether object or category
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        return 'classification'
    
    #CASE:2 , check bool
    if y.dtype == 'bool':
        return 'classification'
    
    #CASE:3 , check regression continuous target
    unique_values = y.nunique()
    if unique_values <= threshold:
        return 'classification'
    else:
        return 'regression'