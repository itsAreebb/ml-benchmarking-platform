from Datasets.dataset_registry import get_all_datasets
from experiment_runner import run_experiments
from Pipeline.ranking import rank_models_per_dataset, global_model_ranking

datasets = get_all_datasets()

results_df, failed_df = run_experiments(datasets)

# Save raw results
if not results_df.empty:
    results_df.to_csv("results_all_datasets.csv", index=False)

# Save failures
if not failed_df.empty:
    failed_df.to_csv("failed_datasets.csv", index=False)

# -------- Ranking --------
ranked_df = rank_models_per_dataset(results_df)
ranked_df.to_csv("ranked_results_all_datasets.csv", index=False)

global_ranking = global_model_ranking(ranked_df)
global_ranking.to_csv("global_model_ranking.csv", index=False)

print("\nTop models across all datasets:")
print(global_ranking.head(10))
