import numpy as np
import spacy
import json
import os
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import networkx as nx
from MythIsomorphism.MythIsomorphismUtil import isomorphism_wl

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("C:/Users/KIIT/all-MiniLM-L6-v2/")

def load_motifs_from_dict(data):
    return [link.get("motif", "Unknown") for link in data.get("links", [])]

def classify_entity(label):
    label = label.lower().strip()
    if not label:
        return "Unknown"
    doc = nlp(label)
    for ent in doc.ents:
        if ent.label_ in {"PERSON"}:
            return "Human"
        elif ent.label_ in {"ORG", "GPE"}:
            return "Concept"
        elif ent.label_ == "ANIMAL":
            return "Animal"
        elif ent.label_ in {"LOC", "FAC"}:
            return "Thing"

    synsets = wn.synsets(label)
    if synsets:
        top_syn = synsets[0]
        lexname = top_syn.lexname()
        if "animal" in lexname:
            return "Animal"
        elif "person" in lexname:
            return "Human"
        elif "artifact" in lexname or "object" in lexname:
            return "Thing"
        elif "noun.abstract" in lexname:
            return "Concept"

    return "Unknown"

def extract_anonymized_triples(graph_json):
    triples = []
    for link in graph_json.get("links", []):
        source = link.get("source", "")
        src_type = classify_entity(source)
        target = link.get("target", "")
        tgt_type = classify_entity(target)
        motif = link.get("motif", "Unknown")
        triples.append((src_type, tgt_type, motif))
    return triples

def extract_surface_triples(graph_json):
    triples = []
    for link in graph_json.get("links", []):
        source = link.get("source", "").lower().strip()
        target = link.get("target", "").lower().strip()
        motif = link.get("motif", "Unknown").lower().strip()
        triples.append((source, target, motif))
    return triples

def triple_to_sentence(triple):
    subj, obj, rel = triple[0], triple[1], triple[2]
    return f"{subj} {rel} {obj}"

def compute_similarity(triples1, triples2):
    if not triples1 or not triples2:
        return 0.0

    sents1 = [triple_to_sentence(t) for t in triples1]
    sents2 = [triple_to_sentence(t) for t in triples2]

    emb1 = sbert_model.encode(sents1)
    emb2 = sbert_model.encode(sents2)

    sim_matrix = cosine_similarity(emb1, emb2)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    sim_scores = sim_matrix[row_ind, col_ind]

    average_similarity = np.mean(sim_scores)
    return round(average_similarity * 100, 2)

def find_any_isomorphic(current_graph_json, all_graph_files, data_folder="MythoGraphDB",
                         weight_anonymized=0.8, weight_surface=0.2, similarity_threshold=60.0):
    import networkx as nx
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

def build_graph(triples):
    G = nx.DiGraph()
    for src, tgt, rel in triples:
        G.add_node(src)
        G.add_node(tgt)
        G.add_edge(src, tgt, label=rel)
    return G

def are_isomorphic(g1_triples, g2_triples):
    G1 = build_graph(g1_triples)
    G2 = build_graph(g2_triples)

    def node_match(n1, n2):
        return n1 == n2

    def edge_match(e1, e2):
        return e1["label"] == e2["label"]

    return nx.is_isomorphic(G1, G2,
                            node_match=node_match,
                            edge_match=edge_match)