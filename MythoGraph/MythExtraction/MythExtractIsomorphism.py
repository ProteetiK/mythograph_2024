import spacy
import json
import os
from sentence_transformers import SentenceTransformer
import networkx as nx
import streamlit as st
import re
from MythIsomorphism.MythIsomorphismUtil import isomorphism_wl_vf2, get_similarity_scores, compute_text_similarity

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("C:/Users/KIIT/all-MiniLM-L6-v2/")

def clean_file_name(filename):
    basename = os.path.basename(filename)
    basename = re.sub(r'_knowledge_graph\.json$', '', basename)
    basename = re.sub(r'\.json$', '', basename)
    basename = re.sub(r'\s+txt$', '', basename, flags=re.IGNORECASE)
    basename = re.sub(r'\s*\(.*\)$', '', basename)
    return basename

def find_any_isomorphic(current_graph_json, all_graph_files):
    matches = []
    current_graph = nx.node_link_graph(current_graph_json)
    i=0
    for graph_file in all_graph_files:
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                other_graph_json = json.load(f)
            other_graph = nx.node_link_graph(other_graph_json)
        except Exception as e:
            print(f"Skipping {graph_file} due to error: {e}")
            continue
        print (i)
        i = i+1
        if len(current_graph.nodes) != len(other_graph.nodes) and len(current_graph.edges) != len(other_graph.edges):
            continue

        is_iso, elapsed = isomorphism_wl_vf2(current_graph, other_graph)
        print(elapsed)
        if is_iso:
            matches.append(clean_file_name(graph_file))

    return matches


def find_any_similar(current_graph_json, all_graph_files,
                         weight_anonymized=0.6, weight_surface=0.05, 
                         weight_opposition=0.2, weight_text=0.1, 
                         weight_jaccard=0.05, similarity_threshold=20.0):

    similarity_results = []
    current_text = current_graph_json.get("myth_text", "")
    for filename in all_graph_files:
        try:
            with open(filename, encoding="utf-8") as f:
                other_graph_json = json.load(f)
                other_text = other_graph_json.get("myth_text", "")
                anon_sim, surf_sim, opposition_sim, jaccard_sim  = get_similarity_scores(current_graph_json, other_graph_json)
                text_sim = compute_text_similarity(current_text, other_text)
                if opposition_sim == -1:
                    weight_anonymized=0.7
                    weight_surface=0.05 
                    weight_opposition=0.0
                    weight_text=0.2
                    weight_jaccard=0.05
                    final_score = round(
                    weight_anonymized * anon_sim + 
                    weight_surface * surf_sim +
                    weight_text * text_sim +
                    weight_jaccard * jaccard_sim * 100, 2)
                else:
                    weight_anonymized = 0.6
                    final_score = round(
                        weight_anonymized * anon_sim + 
                        weight_surface * surf_sim + 
                        weight_text * text_sim +
                        weight_opposition * opposition_sim + 
                        weight_jaccard * jaccard_sim * 100, 2)
                st.write("Anon Sim:", anon_sim)
                st.write("Surface Sim:", surf_sim)
                st.write("Opposition Sim:", opposition_sim)
                st.write("Text Sim:", text_sim)
                st.write("Jaccard Sim:", jaccard_sim)
                st.write("Final Sim:", final_score)
                if final_score >= similarity_threshold:
                    similarity_results.append({
                        "file": clean_file_name(filename),
                        "similarity": final_score
                    })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return sorted(similarity_results, key=lambda x: x["similarity"], reverse=True)

def display_similarity(graph_json):
    DATA_FOLDER = "MythoGraphDB"
    all_graph_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(DATA_FOLDER)
        for file in files
        if file.endswith(".json")
    ]
    matches = find_any_similar(graph_json, all_graph_files)
    if matches:
        st.success("Similar Myths:")
        for m in matches:
            st.write(f"- {m['file'].replace('_knowledge_graph.json','').replace('_', ' ')} ({m['similarity']}% match)")
    else:
        st.info("No similar myths.")
