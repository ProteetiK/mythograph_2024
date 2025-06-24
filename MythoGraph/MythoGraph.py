import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from MythoGraphUtil import extract_knowledge_graph, draw_graph, is_isomorphic_with_db

st.set_page_config(page_title="Myth Knowledge Graph", layout="wide")

st.title("Myth Knowledge Graph Visualizer")

uploaded_file = st.file_uploader("Upload a myth text file (.txt)", type=["txt"])

if uploaded_file:
    myth_text = uploaded_file.read().decode("utf-8")
    st.subheader("Input Myth Text")
    st.text_area("Myth Content", myth_text, height=300)

    if st.button("Generate Knowledge Graph"):
        with st.spinner("Extracting entities and relationships..."):
            G = extract_knowledge_graph(myth_text)

        st.subheader("Extracted Triples")
        edges = [(u, v, d['label']) for u, v, d in G.edges(data=True)]
        st.write(edges)

        st.subheader("Knowledge Graph")
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_color='orange', edge_color='gray')
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        st.pyplot(fig)

        if st.button("Check for Isomorphic Match"):
            with st.spinner("Checking for isomorphic graphs in the database..."):
                match = is_isomorphic_with_db(G, db_folder="graph_db")

            if match:
                st.success(f"Isomorphic match found in database: {match}")
            else:
                st.info("No isomorphic graph found in the database.")
