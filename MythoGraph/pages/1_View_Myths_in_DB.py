import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
from MythGraph.MythGraphDraw import draw_graph

W1 = 0.7
st.title("View Knowledge Graph from Database")
folder_path = "MythoGraphDB"
json_files = [
    os.path.join(root, f)
    for root, _, files in os.walk("MythoGraphDB")
    for f in files if f.endswith(".json")
]

if not json_files:
    st.warning("No JSON files found in the 'MythoGraphDB' folder.")
else:
    selected_file = st.selectbox("Select a file to view its knowledge graph", json_files)
    if selected_file:

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

        G, myth_text = load_graph_from_file(selected_file)
        st.subheader("Knowledge Graph")
        fig = draw_graph(G, title=selected_file)
        st.pyplot(fig)
        st.subheader("Extracted Triples")
        st.write([(u, v, d['label'], d['motif']) for u, v, d in G.edges(data=True)])
        st.subheader("Myth Text")
        st.write(myth_text)