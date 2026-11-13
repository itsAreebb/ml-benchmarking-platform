import pandas as pd
import matplotlib.pyplot as plt

ranked_df = pd.read_csv("ranked_results_all_datasets.csv")


# Speed vs Performance

df = ranked_df.copy()

# Unified performance:
# - Classification → F1 (higher better)
# - Regression → -RMSE (so higher is better)
df["performance"] = None
df.loc[df["problem_type"] == "classification", "performance"] = df["F1"]
df.loc[df["problem_type"] == "regression", "performance"] = -df["RMSE"]

plt.figure(figsize=(10,6))
plt.scatter(df["train_time"], df["performance"])
plt.title("Training Time vs Model Performance")
plt.xlabel("Training Time (seconds)")
plt.ylabel("Performance (Higher is Better)")
plt.tight_layout()
plt.show()

plt.savefig("Visuals/model_wins.png")
plt.close()
