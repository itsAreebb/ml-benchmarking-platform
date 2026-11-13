import pandas as pd
import matplotlib.pyplot as plt

ranked_df = pd.read_csv("ranked_results_all_datasets.csv")

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
plt.show()

plt.savefig("Visuals/model_wins.png")
plt.close()
