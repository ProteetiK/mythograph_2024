import os
import re
import json
import pandas as pd
import streamlit as st
from MythModelTrain.MotifTrainer import train_unsupervised_model, is_model_trained, train_motif_model
from MythModelTrain.TripleTrainer import train_triple_extractor
from MythGraph.MythGraphDraw import export_graph_as_custom_json, draw_graph
from MythExtraction.MythExtractUtil import extract_knowledge_graph, extract_oppositions
from MythExtraction.MythExtractIsomorphism import find_any_isomorphic, find_any_similar

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("C:/Users/KIIT/all-MiniLM-L6-v2/")
#model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Myth Knowledge Graph", layout="wide")
st.title("Myth Knowledge Graph Visualizer")

def display_similarity(graph_json):
    all_graph_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(DATA_FOLDER)
        for file in files
        if file.endswith(".json")
    ]
    with st.spinner("Checking for myths with same motif sequence..."):
        matches = find_any_isomorphic(graph_json, all_graph_files)
    if matches:
        st.success(f"Isomorphic Myths:")
        for m in matches:
            st.write(f"- {m.replace('_knowledge_graph.json','').replace('_', ' ')}")
    else:
        st.info(f"No isomorphic myths.")

    with st.spinner("Checking for myths with similar motif sequence..."):
        matches = find_any_similar(graph_json, all_graph_files)
    if matches:
        st.success("Similar Myths:")
        for m in matches:
            st.write(f"- {m['file'].replace('_knowledge_graph.json','').replace('_', ' ')} ({m['similarity']}% match)")
    else:
        st.info("No similar myths.")

if is_model_trained():
    st.success("Model is trained and ready.")
else:
    st.warning("Model is NOT trained. Please click 'Train Model' below.")
    if st.button("Train Model", key="train_model"):
        with st.spinner("Training the unsupervised model..."):
            #train the model to extract triples from text
            train_triple_extractor()

            #train the model to extract motifs from text
            #train_motif_model()

            #train_unsupervised_model("MythoGraphDB", n_clusters=3)
        st.success("Training complete.")

DATA_FOLDER = "MythoGraphDB"
os.makedirs(DATA_FOLDER, exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload a myth text file (.txt) OR a CSV file (.csv)",
    type=["txt", "csv"]
)

if uploaded_file:
    file_type = uploaded_file.type
    
    if file_type == "text/csv" or uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if not {"Title", "Content"}.issubset(df.columns):
            st.error("CSV must contain 'Title' and 'Content' columns.")
        else:
            csv_file_name = os.path.splitext(uploaded_file.name)[0]
            output_folder = os.path.join(DATA_FOLDER, csv_file_name)
            os.makedirs(output_folder, exist_ok=True)
            st.subheader(f"Processing {len(df)} myths from CSV")
            if st.button("Process & Save All Myths", key="process_all"):
                with st.spinner("Processing all myths and saving JSONs..."):
                    progress_bar = st.progress(0)
                    for idx, row in df.iterrows():
                        title = str(row["Title"])
                        content = str(row["Content"])
                        G = extract_knowledge_graph(content)
                        graph_json = export_graph_as_custom_json(G, content)
                        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title.strip())
                        save_path = os.path.join(output_folder, f"{safe_title}_knowledge_graph.json")
                        with open(save_path, "w", encoding="utf-8") as f_out:
                            json.dump(graph_json, f_out, indent=2)
                        progress_bar.progress((idx + 1) / len(df))
                    progress_bar.empty()
                st.success(f"Processed and saved {len(df)} graphs to `{output_folder}`")

            for idx, row in df.iterrows():
                title = str(row["Title"])
                content = str(row["Content"])
                safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title.strip())
                json_filename = f"{safe_title}_knowledge_graph.json"
                json_path = os.path.join(output_folder, json_filename)

                with st.expander(f"Myth: {title}", expanded=False):
                    st.text_area("Myth Content", content, height=200, key=f"content_{idx}")

                    if st.button(f"Visualize Knowledge Graph", key=f"btn_{idx}"):
                        with st.spinner("Extracting entities and relationships..."):
                            G = extract_knowledge_graph(content)
                        graph_json = export_graph_as_custom_json(G, content)
                        st.subheader("Knowledge Graph Visualization")
                        fig = draw_graph(G, title=f"Knowledge Graph")
                        st.pyplot(fig)
                        st.subheader("Extracted Triples (Links)")
                        st.json(graph_json["links"])
                        st.subheader("Opposition Pairs in the Knowledge Graph")
                        opposition_freq = extract_oppositions(graph_json["links"])
                        if opposition_freq:
                            for pair, freq in opposition_freq.items():
                                st.write(f"**{pair}** -> {freq} occurrence(s)")
                        else:
                            st.info("No opposition pairs found in this myth.")

                    if os.path.exists(json_path):
                        if st.button(f"Check for Similar Myths", key=f"isom_{idx}"):
                            with st.spinner("Extracting entities and relationships..."):
                                G = extract_knowledge_graph(content)
                            graph_json = export_graph_as_custom_json(G, content)
                            display_similarity(graph_json)

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

            st.subheader("Knowledge Graph Visualization")
            fig = draw_graph(G, title="Knowledge Graph")
            st.pyplot(fig)
            st.subheader("Extracted Triples (Links)")
            st.json(graph_json["links"])

            links = graph_json.get("links", [])
            st.subheader("Opposition Pairs in the Knowledge Graph")
            opposition_freq = extract_oppositions(graph_json["links"])
            if opposition_freq:
                for pair, freq in opposition_freq.items():
                    st.write(f"**{pair}** -> {freq} occurrence(s)")
            else:
                st.info("No opposition pairs found in this myth.")

            st.subheader("Download Graph JSON")
            st.download_button(
                label="Download JSON",
                data=json.dumps(graph_json, indent=2),
                file_name=uploaded_file.name + "_JSON.json",
                mime="application/json"
            )

            if st.button("Check for Similar or Isomorphic Myths", key="check_isomorphic_txt"):
                display_similarity(graph_json)

    else:
        st.error("Unsupported file type. Please upload a .txt or .csv file.")