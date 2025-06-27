import os
import re
import json
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from MythoGraphUtil import (
    train_unsupervised_model,
    is_model_trained,
    export_graph_as_custom_json,
    extract_knowledge_graph,
    draw_graph
)

st.set_page_config(page_title="Myth Knowledge Graph", layout="wide")
st.title("Myth Knowledge Graph Visualizer")

if is_model_trained():
    st.success("Model is trained and ready.")
else:
    st.warning("Model is NOT trained. Please click 'Train Model' below.")
    if st.button("Train Model"):
        with st.spinner("Training the unsupervised model..."):
            train_unsupervised_model("MythoGraphDB", n_clusters=3)
        st.success("Training complete.")

DATA_FOLDER = "MythoGraphDB"
os.makedirs(DATA_FOLDER, exist_ok=True)

def load_motifs_from_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    motifs = [link.get("motif", "Unknown") for link in data.get("links", [])]
    return motifs

def find_isomorphic_myths(current_motifs, all_files, current_file):
    matches = []
    for f in all_files:
        if f == current_file:
            continue
        motifs = load_motifs_from_json(os.path.join(DATA_FOLDER, f))
        if motifs == current_motifs:
            matches.append(f)
    return matches

uploaded_file = st.file_uploader("Upload a myth text file (.txt) OR a CSV file (.csv) with 'Title' and 'Content' columns", type=["txt", "csv"])

if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "text/csv" or uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if not {"Title", "Content"}.issubset(df.columns):
            st.error("CSV must contain 'Title' and 'Content' columns.")
        else:
            st.subheader(f"Processing {len(df)} myths from CSV")
            if st.button("Process & Save All Myths"):
                with st.spinner("Processing all myths and saving JSONs..."):
                    for idx, row in df.iterrows():
                        title = str(row["Title"])
                        content = str(row["Content"])

                        G = extract_knowledge_graph(content)
                        graph_json = export_graph_as_custom_json(G, content)

                        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title.strip())
                        save_path = os.path.join(DATA_FOLDER, f"{safe_title}_knowledge_graph.json")
                        with open(save_path, "w", encoding="utf-8") as f_out:
                            json.dump(graph_json, f_out, indent=2)
                st.success(f"Processed and saved {len(df)} graphs to {DATA_FOLDER}")

            all_graph_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]

            for idx, row in df.iterrows():
                title = str(row["Title"])
                content = str(row["Content"])
                safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title.strip())
                json_filename = f"{safe_title}_knowledge_graph.json"
                json_path = os.path.join(DATA_FOLDER, json_filename)
                with st.expander(f"Myth: {title}", expanded=False):
                    st.text_area("Myth Content", content, height=200, key=f"content_{idx}")

                    if st.button(f"Visualize Knowledge Graph for '{title}'", key=f"btn_{idx}"):
                        with st.spinner("Extracting entities and relationships..."):
                            G = extract_knowledge_graph(content)

                        graph_json = export_graph_as_custom_json(G, content)

                        st.subheader("Extracted Triples (Links)")
                        st.json(graph_json["links"])

                        st.subheader("Knowledge Graph Visualization")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pos = nx.spring_layout(G, seed=42)
                        nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_color='orange', edge_color='gray')
                        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
                        st.pyplot(fig)
                    if os.path.exists(json_path):
                        if st.button(f"Check Isomorphic Myths for '{title}'", key=f"isom_{idx}"):
                            with st.spinner("Checking for myths with same motif sequence..."):
                                current_motifs = load_motifs_from_json(json_path)
                                matches = find_isomorphic_myths(current_motifs, all_graph_files, json_filename)

                                if matches:
                                    st.success(f"Myths with the same motif sequence as '{title}':")
                                    for m in matches:
                                        st.write(f"- {m.replace('_knowledge_graph.json','').replace('_', ' ')}")
                                else:
                                    st.info(f"No myths found with the same motif sequence as '{title}'.")

    elif file_type == "text/plain" or uploaded_file.name.endswith(".txt"):
        myth_text = uploaded_file.read().decode("utf-8")

        st.subheader("Input Myth Text")
        st.text_area("Myth Content", myth_text, height=300)

        if st.button("Generate Knowledge Graph"):
            with st.spinner("Extracting entities and relationships..."):
                G = extract_knowledge_graph(myth_text)

            graph_json = export_graph_as_custom_json(G, myth_text)

            st.subheader("Extracted Triples (Links)")
            st.json(graph_json["links"])

            st.subheader("Knowledge Graph Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_color='orange', edge_color='gray')
            edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
            st.pyplot(fig)

            st.subheader("Download Graph JSON")
            st.download_button(
                label="Download JSON",
                data=json.dumps(graph_json, indent=2),
                file_name="myth_knowledge_graph.json",
                mime="application/json"
            )
    else:
        st.error("Unsupported file type. Please upload a .txt or .csv file.")