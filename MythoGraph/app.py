import os
import re
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from MythModelTrain.MotifTrainer import train_unsupervised_model, is_model_trained, train_motif_model
from MythGraph.MythGraphDraw import export_graph_as_custom_json, draw_graph
from MythExtraction.MythExtractUtil import extract_knowledge_graph
from MythExtraction.MythExtractIsomorphism import find_any_isomorphic, load_motifs_from_dict

st.set_page_config(page_title="Myth Knowledge Graph", layout="wide")
st.title("Myth Knowledge Graph Visualizer")

if is_model_trained():
    st.success("Model is trained and ready.")
else:
    st.warning("Model is NOT trained. Please click 'Train Model' below.")
    if st.button("Train Model", key="train_model"):
        with st.spinner("Training the unsupervised model..."):
            train_motif_model()
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

uploaded_file = st.file_uploader(
    "Upload a myth text file (.txt) OR a CSV file (.csv) with 'Title' and 'Content' columns",
    type=["txt", "csv"]
)

if uploaded_file:
    file_type = uploaded_file.type
    all_graph_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]

    if file_type == "text/csv" or uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if not {"Title", "Content"}.issubset(df.columns):
            st.error("CSV must contain 'Title' and 'Content' columns.")
        else:
            st.subheader(f"Processing {len(df)} myths from CSV")
            if st.button("Process & Save All Myths", key="process_all"):
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
                        fig = draw_graph(G, title=f"Knowledge Graph for '{title}'")
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
        st.text_area("Myth Content", myth_text, height=300, key="myth_txt")

        if "graph_generated" not in st.session_state:
            st.session_state.graph_generated = False
        if "graph_json" not in st.session_state:
            st.session_state.graph_json = None
        if "G" not in st.session_state:
            st.session_state.G = None

        if st.button("Generate Knowledge Graph", key="generate_txt"):
            with st.spinner("Extracting entities and relationships..."):
                G = extract_knowledge_graph(myth_text)
            graph_json = export_graph_as_custom_json(G, myth_text)
            st.session_state.graph_generated = True
            st.session_state.graph_json = graph_json
            st.session_state.G = G

        if st.session_state.graph_generated and st.session_state.graph_json:
            graph_json = st.session_state.graph_json
            G = st.session_state.G

            st.subheader("Extracted Triples (Links)")
            st.json(graph_json["links"])
            st.subheader("Knowledge Graph Visualization")
            fig = draw_graph(G, title="Knowledge Graph")
            st.pyplot(fig)

            st.subheader("Download Graph JSON")
            st.download_button(
                label="Download JSON",
                data=json.dumps(graph_json, indent=2),
                file_name="myth_knowledge_graph.json",
                mime="application/json"
            )

            if st.button("Check for Isomorphic Myths", key="check_isomorphic_txt"):
                with st.spinner("Checking for myths with same motif sequence..."):
                    matches = find_any_isomorphic(graph_json, all_graph_files)
                    if matches:
                        st.success("Myths with the same motif sequence:")
                        for m in matches:
                            st.write(f"- {m['file'].replace('_knowledge_graph.json','').replace('_', ' ')} ({m['similarity']}% match)")
                    else:
                        st.info("No myths found with the same motif sequence.")

    else:
        st.error("Unsupported file type. Please upload a .txt or .csv file.")