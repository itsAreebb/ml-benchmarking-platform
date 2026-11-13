import pandas as pd



def rank_models_per_dataset(results_df: pd.DataFrame) -> pd.DataFrame:
    ''' Adds a rank column per dataset based on performance
        Classification : higher f1 score -> better
        Regression : lower RMSE -> better
        '''
    
    ranked_blocks = []

    for dataset_name in results_df['dataset'].unique():
        subset = results_df[results_df['dataset'] == dataset_name].copy()
        problem_type = subset['problem_type'].iloc[0]

        if problem_type == 'classification':
            subset = subset.sort_values(
                by = ['F1', 'Accuracy' , 'train_time'],
                ascending =[False , False, False]
            )
        else:
            subset = subset.sort_values(
                by=['RMSE' , 'R2' , 'train_time'],
                ascending = [True , False , True]
            )

        subset['rank'] = range(1 , len(subset) + 1)
        ranked_blocks.append(subset)

    ranked_df = pd.concat(ranked_blocks , ignore_index=True)
    return ranked_df



def global_model_ranking(ranked_df : pd.DataFrame) -> pd.DataFrame:
    '''
      Ranks models globally based on how often they win(rank == 1)


    '''

    winners = ranked_df[ranked_df['rank'] == 1]

    summary = (
        winners.groupby('model')
        .size()
        .reset_index(name='wins')
        .sort_values(by = 'wins' , ascending=False)

    )
    return summary