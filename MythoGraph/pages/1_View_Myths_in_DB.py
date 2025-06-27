import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

W1 = 0.7

st.title("View Knowledge Graph from Database")

folder_path = "MythoGraphDB"

json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

if not json_files:
    st.warning("No JSON files found in the 'MythoGraphDB' folder.")
else:
    selected_file = st.selectbox("Select a file to view its knowledge graph", json_files)

    if selected_file:
        file_path = os.path.join(folder_path, selected_file)

        def load_graph_from_file(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                links = data.get("links", [])
                myth_text = data.get("myth_text", "No myth text found.")
                edges = []
                for link in links:
                    src = link.get("source")
                    tgt = link.get("target")
                    rel = link.get("label")
                    motif = link.get("motif", None)
                    if src and tgt and rel:
                        edges.append((src, tgt, {'weight': W1, 'label': rel, 'motif': motif}))
                return create_graph(edges), myth_text

        def create_graph(edge_list):
            G = nx.MultiDiGraph()
            for u, v, attr in edge_list:
                G.add_edge(u, v, **attr)
            return G

        def draw_graph(G, title="Knowledge Graph"):
            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(12, 8))

            nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=2000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

            edge_count = defaultdict(int)
            for u, v, k, data in G.edges(keys=True, data=True):
                rel = data.get('label', '')
                motif = data.get('motif', '')
                if motif:
                    rel = f"{rel} / {motif}"
                edge_count[(u, v)] += 1
                rad = 0.15 * edge_count[(u, v)]
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    connectionstyle=f'arc3,rad={rad}',
                    arrowstyle='-|>',
                    arrowsize=20,
                    edge_color='gray',
                    ax=ax
                )
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2 + rad * 0.5
                ax.text(mid_x, mid_y, rel, fontsize=9, color='red', ha='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            ax.set_title(title)
            ax.axis('off')
            return fig

        G, myth_text = load_graph_from_file(file_path)
        st.subheader("Knowledge Graph")
        fig = draw_graph(G, title=selected_file)
        st.pyplot(fig)

        st.subheader("Extracted Triples")
        st.write([(u, v, d['label'], d['motif']) for u, v, d in G.edges(data=True)])

        st.subheader("Myth Text")
        st.write(myth_text)