import spacy
import json
import os
from sentence_transformers import SentenceTransformer
import networkx as nx
from MythIsomorphism.MythIsomorphismUtil import isomorphism_wl, compute_similarity, extract_anonymized_triples, extract_surface_triples

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("C:/Users/KIIT/all-MiniLM-L6-v2/")

def find_any_isomorphic(current_graph_json, all_graph_files, data_folder="MythoGraphDB",
                         weight_anonymized=0.8, weight_surface=0.2, similarity_threshold=60.0):
    matches = []
    current_graph = nx.node_link_graph(current_graph_json)

    for graph_file in all_graph_files:
        graph_path = os.path.join(data_folder, graph_file)

        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                other_graph_json = json.load(f)
            other_graph = nx.node_link_graph(other_graph_json)
        except Exception as e:
            print(f"Skipping {graph_file} due to error: {e}")
            continue

        is_iso, elapsed = isomorphism_wl(current_graph, other_graph)
        #print(f"Checked {graph_file} in {elapsed:.2f} seconds: {'Isomorphic' if is_iso else 'Not isomorphic'}")

        if is_iso:
            matches.append(graph_file)

    return matches

def find_any_similar(current_graph_json, all_graph_files, data_folder="MythoGraphDB",
                         weight_anonymized=0.8, weight_surface=0.2, similarity_threshold=60.0):

    current_anonymized = extract_anonymized_triples(current_graph_json)
    current_surface = extract_surface_triples(current_graph_json)
    similarity_results = []

    for filename in all_graph_files:
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, encoding="utf-8") as f:
                other_graph = json.load(f)

                other_anonymized = extract_anonymized_triples(other_graph)
                other_surface = extract_surface_triples(other_graph)

                anon_sim = compute_similarity(current_anonymized, other_anonymized)
                surf_sim = compute_similarity(current_surface, other_surface)

                final_score = round(weight_anonymized * anon_sim + weight_surface * surf_sim, 2)

                #print(f"{filename}: Anon={anon_sim}, Surface={surf_sim}, Final={final_score}")
                if final_score >= similarity_threshold:
                    similarity_results.append({
                        "file": filename,
                        "similarity": final_score
                    })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return sorted(similarity_results, key=lambda x: x["similarity"], reverse=True)