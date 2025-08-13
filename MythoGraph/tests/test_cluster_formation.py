import os, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, homogeneity_score, completeness_score,
    v_measure_score, fowlkes_mallows_score, adjusted_mutual_info_score
)
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

MODE = "union_find"  # options: "union_find", "hierarchical"
MIN_CLUSTER_SIZE = 1
SIMILARITY_THRESHOLD_RANGE = (
    list(range(30, 40, 5)) +
    list(range(40, 70, 1)) +
    list(range(70, 101, 5))
)
LABEL_CSV_PATH = "D:/MythoGraph/MythoGraph/MythoGraph/LeviStrauss_Gold_JSON_Prep_Clusters.csv"
#LABEL_CSV_PATH = "D:/MythoGraph/MythoGraph/MythoGraph/tests/panchatantra_test_clustering.csv"
OUTPUT_CSV = f"clustering_threshold_panchatantra_{MODE}.csv"

current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from MythIsomorphism.MythIsomorphismUtil import (
    UF,
    extract_anonymized_triples,
    compute_similarity,
    normalize_title
)

def compare_with_ground_truth(files, computed_clusters, label_csv_path):
    try:
        df_labels = pd.read_csv(label_csv_path)
        title_to_index = {normalize_title(os.path.basename(f)): i for i, f in enumerate(files)}

        ground_truth_labels = []
        predicted_labels = []

        for _, row in df_labels.iterrows():
            csv_title = normalize_title(row['Title'])
            if csv_title in title_to_index:
                idx = title_to_index[csv_title]
                ground_truth_labels.append(row['Cluster'])
                predicted_labels.append(computed_clusters[idx])

        if not ground_truth_labels:
            print("No matching files found between clustering and ground truth.")
            return None

        le = LabelEncoder()
        gt_encoded = le.fit_transform(ground_truth_labels)
        pred_encoded = le.fit_transform(predicted_labels)

        return (
            homogeneity_score(ground_truth_labels, predicted_labels),
            completeness_score(ground_truth_labels, predicted_labels),
            v_measure_score(ground_truth_labels, predicted_labels),
            adjusted_rand_score(gt_encoded, pred_encoded),
            adjusted_mutual_info_score(gt_encoded, pred_encoded),
            fowlkes_mallows_score(gt_encoded, pred_encoded)
        )
    except Exception as e:
        print(f"Error comparing with ground truth: {e}")
        return None

folder = os.path.abspath("MythoGraphDB/LeviStrauss_Gold_JSON_Prep_Clusters")
files = sorted([
    os.path.join(root, f)
    for root, _, filenames in os.walk(folder)
    for f in filenames if f.endswith(".json")
])

graphs = [json.load(open(f, encoding="utf-8")) for f in files]
n = len(files)

print(f"\nCollected {n} files.")

triples_list = [extract_anonymized_triples(g) for g in graphs]
sims = {}
for i in range(n):
    for j in range(i + 1, n):
        sim = compute_similarity(triples_list[i], triples_list[j])
        sims[(i, j)] = sim

results = []

if MODE == "union_find":
    for thr in SIMILARITY_THRESHOLD_RANGE:
        print(f"Processing threshold: {thr}")
        uf = UF(n)
        for (i, j), sim in sims.items():
            if sim >= thr:
                uf.u(i, j)

        cluster_map = {}
        cluster_labels = []
        label_counter = 0
        for i in range(n):
            root = uf.f(i)
            if root not in cluster_map:
                cluster_map[root] = label_counter
                label_counter += 1
            cluster_labels.append(cluster_map[root])

        cluster_sizes = Counter(cluster_labels)
        filtered_labels = [
            lbl if cluster_sizes[lbl] >= MIN_CLUSTER_SIZE else -1
            for lbl in cluster_labels
        ]
        num_clusters = len(set(filtered_labels)) - (-1 in filtered_labels)
        singleton_count = list(filtered_labels).count(-1)

        intra_sims, inter_sims = [], []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sims[i, j]
                if filtered_labels[i] == filtered_labels[j] and filtered_labels[i] != -1:
                    intra_sims.append(sim)
                elif filtered_labels[i] != -1 and filtered_labels[j] != -1:
                    inter_sims.append(sim)

        avg_intra = np.mean(intra_sims) if intra_sims else 0
        avg_inter = np.mean(inter_sims) if inter_sims else 0

        metrics = compare_with_ground_truth(files, filtered_labels, LABEL_CSV_PATH)
        if metrics:
            h, c, v, ari, nmi, fms = metrics
            results.append({
                "Threshold": thr,
                "Clusters": num_clusters,
                "Singletons": singleton_count,
                "Intra-Cluster-Similarity": avg_intra,
                "Inter-Cluster-Similarity": avg_inter,
                "Homogeneity": h,
                "Completeness": c,
                "V-Measure": v,
                "ARI": ari,
                "NMI": nmi,
                "FMS": fms
            })

elif MODE == "hierarchical":
    print("Running Hierarchical Clustering...")

    triples_list = [extract_anonymized_triples(g) for g in graphs]
    sims = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(triples_list[i], triples_list[j])
            sims[(i, j)] = sim

    sim_matrix = np.ones((n, n)) * 100.0
    for (i, j), sim in sims.items():
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim
    np.fill_diagonal(sim_matrix, 100.0)
    dist_matrix = 100 - sim_matrix

    results = []

    for threshold in SIMILARITY_THRESHOLD_RANGE:
        print(f"Threshold: {threshold}")

        model = AgglomerativeClustering(
            affinity='precomputed',
            n_clusters=None,
            distance_threshold=100 - threshold,
            linkage='average'
        )
        cluster_labels = model.fit_predict(dist_matrix)

        cluster_sizes = Counter(cluster_labels)
        singleton_count = sum(1 for size in cluster_sizes.values() if size == 1)

        intra_sims = []
        inter_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sims[i, j]
                if cluster_labels[i] == cluster_labels[j]:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)

        avg_intra = np.mean(intra_sims) if intra_sims else 0
        avg_inter = np.mean(inter_sims) if inter_sims else 0

        metrics = compare_with_ground_truth(files, cluster_labels, LABEL_CSV_PATH)

        if metrics:
            h, c, v, ari, nmi, fms = metrics
            results.append({
                "Threshold": threshold,
                "Clusters": len(set(cluster_labels)),
                "Singletons": singleton_count,
                "Intra-Cluster-Similarity": avg_intra,
                "Inter-Cluster-Similarity": avg_inter,
                "Homogeneity": h,
                "Completeness": c,
                "V-Measure": v,
                "ARI": ari,
                "NMI": nmi,
                "FMS": fms
            })

if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(df_results)
else:
    print("No results. Check inputs or similarity values.")