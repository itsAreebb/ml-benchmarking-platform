import pandas as pd
from sklearn.preprocessing import LabelEncoder

def normalize_labels(y):
    """
    Force labels into 0..K-1 format for classification.
    Works for strings, negatives, non-contiguous labels.
    """
    le = LabelEncoder()
    return le.fit_transform(y)


def load_dataset_from_registry(dataset_config):
    ''' this function will get the various datasets from list of 
        datasets and transforms them into pandas dataframe
        load and return X, y as pandas objects.
    '''

    source = dataset_config['source']

    if source == 'sklearn':
        loader = dataset_config['loader']
        data = loader(as_frame = True)
        X = data.data
        y = data.target

        return X , y
    
    elif source =='openml':
        loader = dataset_config['loader']
        name = dataset_config['name']
        


        data = loader(name = name , as_frame = True)
        X = data.data
        y  = data.target

        return X, y
    
    elif source == 'synthetic':
        generator = dataset_config['generator']
        params = dataset_config['params']

        X , y = generator(**params)
          # Convert numpy arrays to pandas
        X = pd.DataFrame(X)
        y = pd.Series(y, name="target")

        return X, y
    else:
        raise ValueError(f'unknown dataset source{source}')
    

         