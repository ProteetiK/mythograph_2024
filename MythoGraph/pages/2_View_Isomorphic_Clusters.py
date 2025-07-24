import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os, json, random, tempfile
from collections import defaultdict
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from MythIsomorphism.MythIsomorphismUtil import (
    UF,
    cosine_similarity,
    extract_anonymized_triples,
    triple_to_sentence,
    get_embeddings,
    sim_score,
    extract_original_triples,
    extract_characters,
    characters_to_sentence,
    generate_graph_from_links,
    get_all_cluster_labels,
    compare_clusterings,
    iso_algorithms,
)

st.title("Myth Clusters Viewer")

mode = st.radio("Choose Clustering Type", [
    "Structural Similarity (Anonymized)",
    "Character Similarity",
    "Structural Similarity (Original JSON)",
    "Isomorphism-Based Clustering"
])

thr = st.slider("Similarity Threshold (%)", 40, 100, 60, 5)

def cluster_color(cid):
    random.seed(cid)
    return f"hsl({random.randint(0, 360)}, 70%, 60%)"

def load_graphs(files):
    graphs = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            with open(path, encoding="utf-8") as file:
                #print(file)
                data = json.load(file)
                graphs.append(data)
        except json.JSONDecodeError as e:
            st.error(f"JSONDecodeError in file: `{f}`\n\n**Error:** {str(e)}")
            st.stop()  # stop further execution to focus on the error
        except Exception as e:
            st.error(f"Unexpected error in file: `{f}`\n\n**Error:** {str(e)}")
            st.stop()
    return graphs

if st.button("Generate Clusters"):
    folder = "MythoGraphDB"
    files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    n = len(files)

    if n == 0:
        st.warning("No myth JSON files found.")
    else:
        graphs = load_graphs(files)

        if mode == "Structural Similarity (Anonymized)":
            triples = [extract_anonymized_triples(g) for g in graphs]
            sentences = [[triple_to_sentence(t) for t in tr] for tr in triples]
            emb = get_embeddings(sentences)
            compare_fn = lambda i, j: sim_score(emb[i], emb[j])
            compare_fn_cosine = compare_fn

        elif mode == "Structural Similarity (Original JSON)":
            triples = [extract_original_triples(g) for g in graphs]
            sentences = [[triple_to_sentence(t) for t in tr] for tr in triples]
            emb = get_embeddings(sentences)
            compare_fn = lambda i, j: sim_score(emb[i], emb[j])
            compare_fn_cosine = compare_fn

        elif mode == "Character Similarity":
            characters = [extract_characters(g) for g in graphs]
            sentences = [[characters_to_sentence(c)] for c in characters]
            emb = get_embeddings(sentences)
            compare_fn = lambda i, j: sim_score(emb[i], emb[j])
            compare_fn_cosine = compare_fn

        elif mode == "Isomorphism-Based Clustering":
            graphs_nx = [generate_graph_from_links(g.get("links", [])) for g in graphs]

            def graph_to_vector(g: nx.Graph):
                return [
                    g.number_of_nodes(),
                    g.number_of_edges(),
                    nx.density(g),
                    nx.average_clustering(g) if g.number_of_nodes() > 1 else 0,
                ]

            graph_vectors = [graph_to_vector(g) for g in graphs_nx]
            similarity_matrix = cosine_similarity(graph_vectors)
            compare_fn_cosine = lambda i, j: similarity_matrix[i][j] * 100

            iso_fn = iso_algorithms["Weisfeiler-Lehman"]
            compare_fn = lambda i, j: 100.0 if iso_fn(graphs_nx[i], graphs_nx[j])[0] else 0.0

        uf = UF(n)
        sims, sims_measure = {}, {}
        prog = st.progress(0.0)
        total = n * (n - 1) // 2
        cnt = 0

        for i in range(n):
            for j in range(i + 1, n):
                sim = compare_fn(i, j)
                sim_measure = compare_fn_cosine(i, j) if mode == "Isomorphism-Based Clustering" else sim
                sims[(i, j)] = sim
                sims_measure[(i, j)] = sim_measure
                if sim >= thr:
                    uf.u(i, j)
                cnt += 1
                if cnt % 100 == 0 or cnt == total:
                    prog.progress(cnt / total)

        clust = defaultdict(list)
        for i in range(n):
            clust[uf.f(i)].append(i)

        st.success(f"Processed {n} myths into {len(clust)} clusters.")

        st.subheader("Cluster Network Graph")
        net = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff", font_color="#000000")
        net.force_atlas_2based()

        cid_map = {i: uf.f(i) for i in range(n)}
        unique_clusters = sorted(set(cid_map.values()))
        colors = {cid: cluster_color(cid) for cid in unique_clusters}

        for i in range(n):
            cid = cid_map[i]
            net.add_node(i, label=files[i], title=files[i], color=colors[cid], group=cid)

        for (i, j), v in sims.items():
            if v >= thr:
                net.add_edge(i, j, value=v)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as html_tmp:
            net.save_graph(html_tmp.name)
            html_content = open(html_tmp.name, encoding="utf-8").read()
            components.html(html_content, height=600, scrolling=True)

        st.subheader("Cluster Quality (Intra vs Inter Similarity)")
        intra_sims, inter_sims = [], []

        for i in range(n):
            for j in range(i + 1, n):
                sim_measure = sims_measure.get((i, j)) or sims_measure.get((j, i)) or 0
                if cid_map[i] == cid_map[j]:
                    intra_sims.append(sim_measure)
                else:
                    inter_sims.append(sim_measure)

        avg_intra = np.mean(intra_sims) if intra_sims else 0
        avg_inter = np.mean(inter_sims) if inter_sims else 0

        st.metric("Avg Intra-cluster Similarity", f"{avg_intra:.2f}")
        st.metric("Avg Inter-cluster Similarity", f"{avg_inter:.2f}")

        st.subheader("Cluster Details")
        rows = []

        for root, mem in clust.items():
            if len(mem) > 1:
                graphs_cluster = [json.load(open(os.path.join(folder, files[i]), encoding="utf-8")) for i in mem]
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

                if motif_counts:
                    top_motifs = sorted(motif_counts.items(), key=lambda x: -x[1])[:5]
                    top_sources = sorted(source_counts.items(), key=lambda x: -x[1])[:5]
                    top_targets = sorted(target_counts.items(), key=lambda x: -x[1])[:5]
                    cluster_name = f"{top_motifs[0][0]} | {top_sources[0][0]} | {top_targets[0][0]}"
                    color_display = f'<span style="color:{colors[root]}; font-weight:bold;">Cluster ID: {cluster_name}</span>'
                    st.markdown(color_display, unsafe_allow_html=True)

                for i in mem:
                    rows.append({"cluster_id": cluster_name, "file": files[i]})

                st.markdown("---")

        if rows:
            df = pd.DataFrame(rows)
            st.download_button(
                label="Download Cluster Info CSV",
                data=df.to_csv(index=False),
                file_name="myth_clusters.csv",
                mime="text/csv"
            )

elif st.button("Generate Comparative Report"):
    folder = "MythoGraphDB"
    files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    n = len(files)

    if n == 0:
        st.warning("No myth JSON files found.")
    else:
        graphs = load_graphs(files)

        graphs_nx = [generate_graph_from_links(g.get("links", [])) for g in graphs]

        labels_dict, timings = get_all_cluster_labels(graphs_nx)

        for method, t in timings.items():
            st.write(f"{method}: {t} seconds")

        df_similarity = compare_clusterings(labels_dict)
        st.subheader("Similarity of Clusters Between Isomorphism Methods")
        st.dataframe(df_similarity)