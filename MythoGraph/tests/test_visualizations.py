import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clustering_threshold_panchatantra_hierarchical.csv")

plt.style.use('ggplot')

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Clustering Evaluation Metrics vs Threshold", fontsize=16)

axs[0, 0].plot(df["Threshold"], df["Clusters"], marker='o')
axs[0, 0].set_title("Number of Clusters")
axs[0, 0].set_xlabel("Threshold")
axs[0, 0].set_ylabel("Clusters")
axs[0, 0].grid(True)

axs[0, 1].plot(df["Threshold"], df["V-Measure"], marker='o', color='blue')
axs[0, 1].set_title("V-Measure")
axs[0, 1].set_xlabel("Threshold")
axs[0, 1].set_ylabel("V-Measure")
axs[0, 1].grid(True)

axs[1, 0].plot(df["Threshold"], df["ARI"], marker='o', color='green')
axs[1, 0].set_title("Adjusted Rand Index (ARI)")
axs[1, 0].set_xlabel("Threshold")
axs[1, 0].set_ylabel("ARI")
axs[1, 0].grid(True)

axs[1, 1].plot(df["Threshold"], df["Homogeneity"], label="Homogeneity", marker='o')
axs[1, 1].plot(df["Threshold"], df["Completeness"], label="Completeness", marker='o')
axs[1, 1].plot(df["Threshold"], df["V-Measure"], label="V-Measure", marker='o')
axs[1, 1].set_title("Homogeneity, Completeness, V-Measure")
axs[1, 1].set_xlabel("Threshold")
axs[1, 1].set_ylabel("Score")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("tests/plots/clustering_metrics_overview_panchatantra_hierarchical.png")
plt.close()

print("Saved figure: clustering_metrics_overview_panchatantra_hierarchical.png")
