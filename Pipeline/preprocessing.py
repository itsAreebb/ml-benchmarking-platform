import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def split_data(X , y , test_size = 0.2 , random_state = 42):
    '''this func will split the all input data into 80% training data and 20% test data'''
    return train_test_split(X , y  , test_size=test_size , random_state=random_state)

def build_preprocessor(X):
    '''this func acts as preprocessor who will process various data engineering
       like imputation , scaling etc on input features according to the needful
       features required to apply ML 
       Rules:
       Numerical : impute + scale
       categorical : encoding + impute'''
    
    #identify columns dtype like numerical or categorical
    numerical_features = X.select_dtypes(include = ['int64' , 'float64']).columns
    categorical_features = X.select_dtypes(exclude = ['int64' , 'float64']).columns
  

    #Numerical Pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer' ,SimpleImputer(strategy='mean')),
        ('scale' , StandardScaler())
    ])

    #Categorical Pipeline
    categorical_pipeline = Pipeline(steps=[ 
            ('imputer' , SimpleImputer(strategy='most_frequent')),
            ('encoder' , OneHotEncoder(handle_unknown='ignore'))
    ])

    #ColumnTransformer implementation
    preprocessor = ColumnTransformer(
        transformers=[
            ('num' , numerical_pipeline , numerical_features),
            ('cat' , categorical_pipeline , categorical_features)
        ]
    )
    return preprocessor