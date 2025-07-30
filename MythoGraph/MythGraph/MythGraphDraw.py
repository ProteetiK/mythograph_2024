import os
import json
import spacy
from urllib.parse import quote
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA
import streamlit as st
import math

nlp = spacy.load("en_core_web_sm")

GRAPH_MODEL = None
GRAPH_EMBEDDINGS = []
GRAPH_FILES = []
KMEANS = None
N_CLUSTERS = 5

def build_nx_graph(triples):
    G = nx.MultiDiGraph()
    for triple in triples:
        if len(triple) == 5:
            s, o, p, motif, weight = triple
        elif len(triple) == 4:
            s, o, p, motif = triple
            weight = 0.9
        elif len(triple) == 3:
                s, o, motif = triple
                p = "Unknown"
                weight = 0.9
        else:
            continue
        G.add_edge(s, o, label=p, motif=motif, weight=weight)
    return G

def save_graph_to_json(G, filepath, myth_text=""):
    data = nx.node_link_data(G)
    data["multigraph"] = True
    data["myth_text"] = myth_text
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# def draw_graph(G, title="Knowledge Graph"):
#     pos = nx.spring_layout(G, seed=123, k=2)
#     fig, ax = plt.subplots(figsize=(12, 8))
#     nx.draw_networkx_nodes(G, pos, node_color='#FFBF00', node_size=500, ax=ax, alpha=0.2)
#     nx.draw_networkx_labels(G, pos, font_size=15, ax=ax)
#     edge_count = defaultdict(int)
#     for u, v, k, data in G.edges(keys=True, data=True):
#         source = u
#         target = v
#         label = data.get('label', '')
#         motif = data.get('motif', '')
#         rel = label
#         if motif:
#             rel = f"{label} / {motif}"
#         edge_count[(u, v)] += 1
#         rad = 0.15 * edge_count[(u, v)]
#         nx.draw_networkx_edges(
#             G, pos,
#             edgelist=[(u, v)],
#             connectionstyle=f'arc3,rad={rad}',
#             arrowstyle='-|>',
#             arrowsize=20,
#             edge_color='gray',
#             ax=ax
#         )
#         mid_x = (pos[u][0] + pos[v][0]) / 2
#         mid_y = (pos[u][1] + pos[v][1]) / 2 + rad * 0.5
#         ax.text(mid_x, mid_y, rel, fontsize=12, color='red', ha='center',
#                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
#     ax.set_title(title)
#     ax.axis('off')
#     fig.tight_layout()
#     return fig

def draw_graph(G, title="Knowledge Graph"):
    pos = nx.spring_layout(G, seed=123, k=2)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='#FFBF00', node_size=500, ax=ax, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=15, ax=ax)

    merged_edges = defaultdict(lambda: {"labels": [], "weight": 0.9})
    motif_counts = defaultdict(int)

    for u, v, data in G.edges(data=True):
        label = data.get('label', '')
        motif = data.get('motif', '')
        key = (u, v, motif)
        merged_edges[key]["labels"].append(label)
        merged_edges[key]["weight"] += 0.2
        motif_counts[(u, v)] += 1

    offset_map = defaultdict(int)

    for (u, v, motif), info in merged_edges.items():
        labels = sorted(set(info["labels"]))
        label_text = " || ".join(labels)
        full_text = f"{label_text} / {motif}" if motif else label_text
        weight = info["weight"]
        offset_index = offset_map[(u, v)]
        offset_map[(u, v)] += 1
        rad = 0.15 * (offset_index + 1)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>',
            arrowsize=20 + 5 * len(labels),
            width=weight,
            edge_color='gray',
            ax=ax
        )
        angle = math.atan2(pos[v][1] - pos[u][1], pos[v][0] - pos[u][0])
        dx = 0.1 * math.cos(angle + math.pi / 2) * (offset_index + 1)
        dy = 0.1 * math.sin(angle + math.pi / 2) * (offset_index + 1)

        mid_x = (pos[u][0] + pos[v][0]) / 2 + dx
        mid_y = (pos[u][1] + pos[v][1]) / 2 + dy

        ax.text(mid_x, mid_y, full_text, fontsize=12, color='red', ha='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig

def load_graphs_from_folder(folder_path):
    graphs = []
    global GRAPH_FILES
    GRAPH_FILES = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".json"):
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                data = json.load(f)
            G = json_graph.node_link_graph(data)
            G = nx.convert_node_labels_to_integers(G)
            graphs.append(G)
            GRAPH_FILES.append(fname)
    return graphs

def visualize_clusters():
    if len(GRAPH_EMBEDDINGS) < 2:
        print("Not enough data.")
        return
    reduced = PCA(n_components=2).fit_transform(GRAPH_EMBEDDINGS)
    labels = KMEANS.labels_
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, label=f"Cluster {labels[i]}")
        plt.annotate(GRAPH_FILES[i], (x, y), fontsize=8)
    plt.title("Myth Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def export_graph_as_custom_json(G, myth_text):
    data = {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [{"id": str(n)} for n in G.nodes()],
        "links": [],
        "myth_text": myth_text.strip()
    }

    for u, v, data_dict in G.edges(data=True):
        label = data_dict.get("label", "")
        motif = data_dict.get("motif", "Unknown")
        weight = data_dict.get("weight", 0.9)

        data["links"].append({
            "source": str(u),
            "target": str(v),
            "label": label,
            "motif": motif,
            "weight": round(weight, 2)
        })

    return data
