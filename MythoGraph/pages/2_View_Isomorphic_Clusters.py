import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os, json, tempfile
from collections import defaultdict
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import streamlit.components.v1 as components
from pyvis.network import Network

from MythIsomorphism.MythIsomorphismUtil import (
    UF,
    extract_anonymized_triples,
    extract_original_triples,
    extract_characters,
    triple_to_sentence,
    characters_to_sentence,
    get_embeddings,
    compute_similarity,
    compare_with_ground_truth,
    cluster_color,
    load_graphs
)

st.set_page_config(layout="wide")
st.title("Myth Clusters Viewer")
label_csv_path = "D:/MythoGraph/MythoGraph/MythoGraph/LeviStrauss_Gold_JSON_Prep_Clusters.csv"

mode = st.radio("Choose Clustering Type", [
    "Structural Similarity (Anonymized)",
    "Structural Similarity (Original JSON)",
    "Character Similarity"
])

clustering_method = st.radio("Choose Clustering Algorithm", ["Union-Find", "Hierarchical"])

thr = st.slider("Similarity Threshold (%)", 30, 100, 70, 5)

if st.button("Generate Clusters"):
    folder = os.path.abspath("MythoGraphDB")
    files = sorted([
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder)
        for f in filenames if f.endswith(".json")
    ])
    n = len(files)

    if n == 0:
        st.warning("No myth JSON files found.")
    else:
        graphs = load_graphs(files)

        if mode == "Structural Similarity (Anonymized)":
            triples_list = [extract_anonymized_triples(g) for g in graphs]
        elif mode == "Structural Similarity (Original JSON)":
            triples_list = [extract_original_triples(g) for g in graphs]
        elif mode == "Character Similarity":
            triples_list = [[characters_to_sentence(extract_characters(g))] for g in graphs]

        sims = np.zeros((n, n))
        prog = st.progress(0.0)
        total = n * (n - 1) // 2
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_similarity(triples_list[i], triples_list[j])
                sims[i, j] = sim
                sims[j, i] = sim
                cnt += 1
                if cnt % 100 == 0 or cnt == total:
                    prog.progress(cnt / total)

        if clustering_method == "Union-Find":
            uf = UF(n)
            for i in range(n):
                for j in range(i + 1, n):
                    if sims[i, j] >= thr:
                        uf.u(i, j)
            cluster_map = defaultdict(list)
            for i in range(n):
                cluster_map[uf.f(i)].append(i)

        elif clustering_method == "Hierarchical":
            distance_matrix = 100 - sims
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity='precomputed',
                linkage='average',
                distance_threshold=100 - thr
            )
            labels = clustering.fit_predict(distance_matrix)
            cluster_map = defaultdict(list)
            for i, label in enumerate(labels):
                cluster_map[label].append(i)

        st.success(f"Processed {n} myths into {len(cluster_map)} clusters using {clustering_method}.")

        st.subheader("Cluster Network Graph")
        net = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff", font_color="#000000")
        net.force_atlas_2based()

        if clustering_method == "Union-Find":
            cid_map = {i: uf.f(i) for i in range(n)}
        elif clustering_method == "Hierarchical":
            cid_map = {i: labels[i] for i in range(n)}

        unique_clusters = sorted(set(cid_map.values()))
        colors = {cid: cluster_color(cid) for cid in unique_clusters}

        for i in range(n):
            cid = next((c for c, members in cluster_map.items() if i in members), None)
            file_name = os.path.basename(files[i])
            net.add_node(i, label=file_name, title=file_name, color=colors[cid], group=int(cid))

        for i in range(n):
            for j in range(i + 1, n):
                v = sims[i, j]
                if v >= thr:
                    net.add_edge(i, j, value=float(v), title=str(float(v)), label=str(round(float(v) / 100, 2)), color="#808080", font={"color": "#808080"})


        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as html_tmp:
            net.save_graph(html_tmp.name)
            html_content = open(html_tmp.name, encoding="utf-8").read()
            components.html(html_content, height=600, scrolling=True)

        st.subheader("Cluster Quality (Intra vs Inter Similarity)")
        intra_sims, inter_sims = [], []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sims[i, j]
                cid_i = next((c for c, members in cluster_map.items() if i in members), None)
                cid_j = next((c for c, members in cluster_map.items() if j in members), None)
                if cid_i == cid_j:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)

        avg_intra = np.mean(intra_sims) if intra_sims else 0
        avg_inter = np.mean(inter_sims) if inter_sims else 0

        st.metric("Avg Intra-cluster Similarity", f"{avg_intra:.2f}")
        st.metric("Avg Inter-cluster Similarity", f"{avg_inter:.2f}")

        st.subheader("Cluster Details")
        rows = []
        for root, members in cluster_map.items():
            if len(members) > 1:
                cluster_id = f"Cluster_{root}"
                graphs_cluster = [
                    json.load(open(files[i], encoding="utf-8"))
                    for i in members
                ]
                motif_counts = defaultdict(int)
                source_counts = defaultdict(int)
                target_counts = defaultdict(int)

                for g in graphs_cluster:
                    for link in g.get("links", []):
                        motif = link.get("motif") or link.get("label", "Unknown")
                        src = link.get("source", "Unknown")
                        tgt = link.get("target", "Unknown")
                        motif_counts[motif] += 1
                        source_counts[src] += 1
                        target_counts[tgt] += 1

                top_motifs = sorted(motif_counts.items(), key=lambda x: -x[1])[:5]
                top_sources = sorted(source_counts.items(), key=lambda x: -x[1])[:5]
                top_targets = sorted(target_counts.items(), key=lambda x: -x[1])[:5]

                top_label = f"Motif: {top_motifs[0][0]} | Source: {top_sources[0][0]} | Target: {top_targets[0][0]}"
                color_display = f'<span style="color:{colors[root]}; font-weight:bold;">{cluster_id}: {top_label}</span>'
                st.markdown(color_display, unsafe_allow_html=True)

                with st.expander("View Members"):
                    for i in members:
                        st.markdown(f"- {os.path.basename(files[i])}")
                        rows.append({"cluster_id": top_label, "file": os.path.basename(files[i])})

                with st.expander("View Top Similar Pairs"):
                    intra_pairs = []
                    for i1 in range(len(members)):
                        for j1 in range(i1 + 1, len(members)):
                            i_idx, j_idx = members[i1], members[j1]
                            sim = sims[i_idx, j_idx]
                            if sim:
                                intra_pairs.append((sim, i_idx, j_idx))

                    top_pairs = sorted(intra_pairs, reverse=True)[:5]
                    for sim, i_idx, j_idx in top_pairs:
                        st.markdown(f"- **{os.path.basename(files[i_idx])}** <-> **{os.path.basename(files[j_idx])}** : {sim:.2f}%")

                st.markdown("---")

        if rows:
            df = pd.DataFrame(rows)
            st.download_button(
                label="Download Cluster Info CSV",
                data=df.to_csv(index=False),
                file_name="myth_clusters.csv",
                mime="text/csv"
            )

        computed_cluster_ids = []
        for i in range(n):
            cid = next((c for c, members in cluster_map.items() if i in members), None)
            computed_cluster_ids.append(cid)

        if os.path.exists(label_csv_path):
            compare_with_ground_truth(files, computed_cluster_ids, label_csv_path)
        else:
            st.info("Provide a valid CSV path with 'Title' and 'Cluster' columns to evaluate clustering quality.")