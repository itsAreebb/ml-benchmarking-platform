import pandas as pd
from Datasets.loader import load_dataset_from_registry, normalize_labels
from Datasets.detector import detect_problem_type
from Pipeline.preprocessing import split_data
from Pipeline.trainer import train_models
from Pipeline.evaluator import evaluate_models
from tqdm import tqdm


def run_experiments(Datasets, tune=True, n_iter=10, cv=3):
    '''this is the final func that will take all 50 dataset and apply all the required ML
    pipeline and the results are evaluated sequentially in a Pandas DataFrame!
    and this concludes our ML project'''

    all_results = []
    failed_datasets = []

    for dataset in tqdm(Datasets , desc='Running ML Experiments'):
        dataset_name = dataset['name']
        print(f'\n Dataset: {dataset_name}')

        try:
            #Load Data
            X , y = load_dataset_from_registry(dataset)

            #detect task to perform
            problem_type = detect_problem_type(y)

            # Encode labels to 0..K-1 for classification only
            # (regression targets must keep their real numeric values)
            if problem_type == 'classification':
                y = normalize_labels(y)

            #split

            X_train , X_test , y_train , y_test , = split_data(X , y)

            #Initialize the training

            trained_models = train_models(X_train , y_train , problem_type,
                                          tune=tune , n_iter=n_iter , cv=cv)

            #Evaluate
            results = evaluate_models(trained_models , X_test , y_test , problem_type)
 
            if len(results) == 0:
                raise ValueError('No models we successfully evaluated')

            # Attach dataset info 
            for r in results:
                r['dataset'] = dataset_name
                r['problem_type'] = problem_type
                all_results.append(r)
        
        except Exception as e:
            print(f' Failed on dataset {dataset_name} : {e}')
            failed_datasets.append({
                'dataset': dataset_name,
                'error': str(e)
            })
            continue



    results_df =  pd.DataFrame(all_results)
    failed_datasets_df = pd.DataFrame(failed_datasets)

    return results_df , failed_datasets_df