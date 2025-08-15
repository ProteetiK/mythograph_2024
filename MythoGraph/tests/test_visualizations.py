import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize(file):
    df = pd.read_csv(file)
    plt.style.use('ggplot')

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Clustering Evaluation Metrics vs Threshold", fontsize=16)

    # Number of Clusters
    axs[0, 0].plot(df["Threshold"], df["Clusters"], marker='o')
    axs[0, 0].set_title("Number of Clusters")
    axs[0, 0].set_xlabel("Threshold")
    axs[0, 0].set_ylabel("Clusters")
    axs[0, 0].grid(True)

    # Adjusted Rand Index (ARI)
    axs[0, 1].plot(df["Threshold"], df["ARI"], marker='o', color='green')
    axs[0, 1].set_title("Adjusted Rand Index (ARI)")
    axs[0, 1].set_xlabel("Threshold")
    axs[0, 1].set_ylabel("ARI")
    axs[0, 1].grid(True)

    # Homogeneity, Completeness, V-Measure
    axs[1, 0].plot(df["Threshold"], df["Homogeneity"], label="Homogeneity", marker='o')
    axs[1, 0].plot(df["Threshold"], df["Completeness"], label="Completeness", marker='o')
    axs[1, 0].plot(df["Threshold"], df["V-Measure"], label="V-Measure", marker='o')
    axs[1, 0].set_title("Homogeneity, Completeness, V-Measure")
    axs[1, 0].set_xlabel("Threshold")
    axs[1, 0].set_ylabel("Score")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # New plot: NMI & FMS
    axs[1, 1].plot(df["Threshold"], df["NMI"], label="NMI", marker='o', color='purple')
    axs[1, 1].plot(df["Threshold"], df["FMS"], label="FMS", marker='o', color='orange')
    axs[1, 1].set_title("NMI & FMS vs Threshold")
    axs[1, 1].set_xlabel("Threshold")
    axs[1, 1].set_ylabel("Score")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Use the CSV filename to generate PNG filename
    basename = os.path.splitext(os.path.basename(file))[0]
    output_path = f"plots/clustering_metrics_overview_{basename}.png"

    plt.savefig(output_path)
    plt.close()

    print(f"Saved figure: {output_path}")


visualize("clustering_threshold_union_find.csv")

visualize("clustering_threshold_hierarchical.csv")

visualize("clustering_threshold_panchatantra_union_find.csv")

visualize("clustering_threshold_panchatantra_hierarchical.csv")

