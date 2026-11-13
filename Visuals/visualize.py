import pandas as pd
import matplotlib.pyplot as plt

ranked_df = pd.read_csv("Results/ranked_results_all_datasets.csv")

# ================================
# 1. Model Wins (Bar Chart)
# ================================
winners = ranked_df[ranked_df["rank"] == 1]
win_counts = winners["model"].value_counts()

plt.figure(figsize=(10,6))
win_counts.plot(kind="bar")
plt.title("Model Wins Across All Datasets")
plt.xlabel("Model")
plt.ylabel("Number of Wins")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("Visuals/model_wins.png")   # <-- Save image
plt.close()


# ================================
# 2. Speed vs Performance
# ================================
df = ranked_df.copy()
df["performance"] = None
df.loc[df["problem_type"] == "classification", "performance"] = df["F1"]
df.loc[df["problem_type"] == "regression", "performance"] = -df["RMSE"]

plt.figure(figsize=(10,6))
plt.scatter(df["train_time"], df["performance"])
plt.title("Training Time vs Model Performance")
plt.xlabel("Training Time (seconds)")
plt.ylabel("Performance (Higher is Better)")
plt.tight_layout()
plt.savefig("Visuals/speed_vs_performance.png")  # <-- Save image
plt.close()
