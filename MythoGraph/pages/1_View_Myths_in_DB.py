import streamlit as st
import networkx as nx
import os
from pathlib import Path
import json

from MythGraph.MythGraphDraw import draw_graph, export_graph_as_custom_json
from MythExtraction.MythExtractIsomorphism import display_similarity
from MythIsomorphism.MythIsomorphismUtil import extract_oppositions

W1 = 0.7
DATA_FOLDER = "MythoGraphDB"

st.title("View Knowledge Graph from Database")

def get_all_json_files():
    DATA_FOLDER = "MythoGraphDB"
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(DATA_FOLDER)
        for f in files if f.endswith(".json")
    ]

json_files = get_all_json_files()

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
        save_dir = Path("saved_graphs")
        save_dir.mkdir(exist_ok=True)
        if st.button("Save Graph"):
            plot_path = save_dir / f"{Path(selected_file).stem}_plot.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        st.subheader("Extracted Triples")
        st.write([(u, v, d['label'], d.get('motif')) for u, v, d in G.edges(data=True)])

        st.subheader("Opposition Pairs in the Knowledge Graph")
        graph_json = export_graph_as_custom_json(G, myth_text)
        opposition_freq = extract_oppositions(graph_json["links"])
        if opposition_freq:
            for pair, freq in opposition_freq.items():
                st.write(f"**{pair}** -> {freq} occurrence(s)")
        else:
            st.info("No opposition pairs found in this myth.")

        display_similarity(graph_json)

        st.subheader("Myth Text")
        st.write(myth_text)