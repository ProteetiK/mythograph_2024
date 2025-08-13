import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs("plots", exist_ok=True)
plt.style.use('ggplot')

df = pd.read_csv("similarity_scores_output.csv")
metrics = ["AnonSim", "SurfaceSim", "OppositionSim", "TextSim", "JaccardSim", "FinalScore"]
df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

pairs = [
    ("FinalScore", "JaccardSim"), 
    ("FinalScore", "OppositionSim"), 
    ("FinalScore", "AnonSim"), 
    ("FinalScore", "SurfaceSim")
]

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Pairwise Scatter Plots of Similarity Metrics", fontsize=16)

for i, (x, y) in enumerate(pairs):
    r = i // 2
    c = i % 2
    sns.scatterplot(data=df, x=x, y=y, ax=axs[r, c])
    sns.regplot(data=df, x=x, y=y, scatter=False, color="blue", ax=axs[r, c])
    axs[r, c].set_title(f"{x} vs {y}")
    axs[r, c].set_xlabel(x)
    axs[r, c].set_ylabel(y)
    axs[r, c].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plots/scatter_subplots.png")
plt.close()

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Histograms of Similarity Metrics", fontsize=16)

for i, metric in enumerate(metrics):
    r = i // 3
    c = i % 3
    axs[r, c].hist(df[metric], bins=15, color="blue", edgecolor="black")
    axs[r, c].set_title(metric)
    axs[r, c].set_xlabel(metric)
    axs[r, c].set_ylabel("Frequency")
    axs[r, c].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plots/histograms_metrics.png")
plt.close()

sorted_df = df.sort_values(by="FinalScore", ascending=False)
sorted_df = sorted_df[::-1]
plt.figure(figsize=(10, 12))
plt.barh(sorted_df["Title"], sorted_df["FinalScore"], color="mediumseagreen")
plt.title("Fables Ranked by FinalScore", fontsize=16)
plt.xlabel("FinalScore")
plt.ylabel("Fable Title")
plt.tight_layout()
plt.savefig("plots/rank_bar_finalscore.png")
plt.close()

print("All plots saved in 'plots' folder.")