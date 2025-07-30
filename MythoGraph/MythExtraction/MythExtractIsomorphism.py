import spacy
import json
import os
from sentence_transformers import SentenceTransformer
import networkx as nx
import re
from MythIsomorphism.MythIsomorphismUtil import isomorphism_wl_vf2, extract_anonymized_triples, extract_surface_triples, get_similarity_scores

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

    for graph_file in all_graph_files:
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                other_graph_json = json.load(f)
            other_graph = nx.node_link_graph(other_graph_json)
        except Exception as e:
            print(f"Skipping {graph_file} due to error: {e}")
            continue

        is_iso, elapsed = isomorphism_wl_vf2(current_graph, other_graph)

        if is_iso:
            matches.append(clean_file_name(graph_file))

    return matches

def find_any_similar(current_graph_json, all_graph_files,
                         weight_anonymized=0.7, weight_surface=0.1, weight_jaccard=0.1, similarity_threshold=75.0):

    current_anonymized = extract_anonymized_triples(current_graph_json)
    current_surface = extract_surface_triples(current_graph_json)
    similarity_results = []

    for filename in all_graph_files:
        try:
            with open(filename, encoding="utf-8") as f:
                other_graph = json.load(f)
                anon_sim, surf_sim, jaccard_sim = get_similarity_scores(other_graph, current_anonymized, current_surface)
                final_score = round(weight_anonymized * anon_sim + weight_surface * surf_sim + weight_jaccard * jaccard_sim * 100, 3)

                if final_score >= similarity_threshold:
                    similarity_results.append({
                        "file": clean_file_name(filename),
                        "similarity": final_score
                    })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return sorted(similarity_results, key=lambda x: x["similarity"], reverse=True)