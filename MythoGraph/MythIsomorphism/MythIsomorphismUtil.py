import streamlit as st, time
import pandas as pd
import numpy as np
import spacy
import os
import networkx as nx
from nltk.corpus import wordnet as wn
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from networkx.algorithms.isomorphism import isomorphvf2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
import time
from networkx.algorithms import isomorphism as iso
from collections import Counter
from MythGraph.MythGraphDraw import build_nx_graph

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# def triple_to_sentence(triple):
#     subj, obj, rel = triple[0], triple[1], triple[2]
#     return f"{subj} {rel} {obj}"

# def load_motifs_from_dict(data):
#     return [link.get("motif", "Unknown") for link in data.get("links", [])]


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

def compute_similarity(triples1, triples2):
    if triples1 is None or len(triples1) == 0 or triples2 is None or len(triples2) == 0:
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

@lru_cache(maxsize=10000)
def classify_entity(label):
    label = label.lower().strip()
    if not label:
        return "Unknown"
    doc = nlp(label)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return "Human"
        if ent.label_ in {"ORG", "GPE"}:
            return "Concept"
        if ent.label_ == "ANIMAL":
            return "Animal"
        if ent.label_ in {"LOC", "FAC"}:
            return "Thing"
    syn = wn.synsets(label)
    if syn:
        lex = syn[0].lexname()
        if "animal" in lex:
            return "Animal"
        if "person" in lex:
            return "Human"
        if "artifact" in lex or "object" in lex:
            return "Thing"
        if "noun.abstract" in lex:
            return "Concept"
    return "Unknown"

def extract_anonymized_triples(g):
    return [(classify_entity(l.get("source", "")), classify_entity(l.get("target", "")), l.get("motif", "Unknown")) for l in g.get("links", [])]

def extract_original_triples(g):
    return [(l.get("source", ""), l.get("target", ""), l.get("motif", "Unknown")) for l in g.get("links", [])]

def extract_characters(g):
    chars = set()
    for l in g.get("links", []):
        chars.add(classify_entity(l.get("source", "")))
        chars.add(classify_entity(l.get("target", "")))
    return list(chars)

def triple_to_sentence(t):
    return f"{t[0]} {t[2]} {t[1]}"

def characters_to_sentence(char_list):
    return " ".join(sorted(set(char_list)))

def get_embeddings(sent_lists, batch=64):
    flat = [s for sub in sent_lists for s in sub]
    emb_flat = sbert_model.encode(flat, batch_size=batch, convert_to_numpy=True, show_progress_bar=False)
    offsets, res = 0, []
    for sl in sent_lists:
        cnt = len(sl)
        res.append(emb_flat[offsets: offsets+cnt])
        offsets += cnt
    return res

def sim_score(e1, e2):
    if e1.size == 0 or e2.size == 0:
        return 0.0
    m = cosine_similarity(e1, e2)
    r, c = linear_sum_assignment(-m)
    return round(np.mean(m[r, c]) * 100, 2)

class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def u(self, a, b):
        pa, pb = self.f(a), self.f(b)
        if pa != pb:
            self.p[pa] = pb

def generate_graph_from_links(links):
    G = nx.Graph()
    for link in links:
        source = link.get("source", "")
        target = link.get("target", "")
        G.add_edge(source, target)
    return G

# def isomorphism_babai(G1, G2):
#     start = time.time()
#     result = nx.is_isomorphic(G1, G2)
#     return result, time.time() - start

# def isomorphism_wl(G1, G2):
#     start = time.time()
#     gm = isomorphvf2.GraphMatcher(G1, G2)
#     return gm.is_isomorphic(), time.time() - start

def WL_refinement(G, max_iter=10):
    labels = {v: str(G.degree[v]) for v in G.nodes()}    
    for _ in range(max_iter):
        new_labels = {}
        for v in G.nodes():
            neighbor_labels = sorted(labels[n] for n in G.neighbors(v))
            new_labels[v] = str(hash(labels[v] + "_" + "_".join(neighbor_labels)))
        if new_labels == labels:
            break
        labels = new_labels
    return labels

def isomorphism_wl_vf2(G1, G2):
    start = time.time()
    labels1 = WL_refinement(G1)
    labels2 = WL_refinement(G2)
    hist1 = Counter(labels1.values())
    hist2 = Counter(labels2.values())
    if hist1 != hist2:
        return False, time.time() - start
    gm = iso.GraphMatcher(G1, G2)
    result = gm.is_isomorphic()
    return result, time.time() - start

def isomorphism_spectral(G1, G2, tol=1e-6):
    start = time.time()
    s1 = np.sort(nx.adjacency_spectrum(G1))
    s2 = np.sort(nx.adjacency_spectrum(G2))
    if len(s1) != len(s2):
        return False, time.time() - start
    result = np.allclose(s1, s2, atol=tol)
    elapsed = time.time() - start
    return result, elapsed

def isomorphism_degree(G1, G2):
    start = time.time()
    d1 = sorted(dict(G1.degree()).values())
    d2 = sorted(dict(G2.degree()).values())
    result = (d1 == d2)
    elapsed = time.time() - start
    return result, elapsed

def jaccard_similarity_wl(G1, G2, max_iter=10):
    def WL_labels_multiset(G, max_iter=10):
        labels = {v: str(G.degree[v]) for v in G.nodes()}
        for _ in range(max_iter):
            new_labels = {}
            for v in G.nodes():
                neighbor_labels = sorted(labels[n] for n in G.neighbors(v))
                new_labels[v] = labels[v] + "_" + "_".join(neighbor_labels)
            if new_labels == labels:
                break
            labels = new_labels
        return Counter(labels.values())

    c1 = WL_labels_multiset(G1, max_iter)
    c2 = WL_labels_multiset(G2, max_iter)

    intersection = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return intersection / union if union > 0 else 1.0

iso_algorithms = {
    "Weisfeiler-Lehman": isomorphism_wl_vf2,
    "Spectral (Eigenvalue)": isomorphism_spectral,
    "Degree Sequence": isomorphism_degree
}

def cluster_graphs(graphs_nx, iso_fn):
    n = len(graphs_nx)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                is_iso, _ = iso_fn(graphs_nx[i], graphs_nx[j])
            except Exception:
                is_iso = False
            if is_iso:
                uf.u(i, j)
    return [uf.f(i) for i in range(n)]

def get_all_cluster_labels(graphs_nx):
    methods = {
        #"Babai": isomorphism_babai,
        "WL_Hash": isomorphism_wl_vf2,
        "Degree": isomorphism_degree,
        "Spectral": isomorphism_spectral,
    }

    labels_dict = {}
    timings_dict = {}

    for name, fn in methods.items():
        st.info(f"Clustering using {name}")
        start = time.time()
        labels = cluster_graphs(graphs_nx, fn)
        end = time.time()
        labels_dict[name] = labels
        timings_dict[name] = round(end - start, 4)

    return labels_dict, timings_dict

def compare_clusterings(labels_dict):
    methods = list(labels_dict.keys())
    results = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            l1, l2 = labels_dict[m1], labels_dict[m2]
            ari = adjusted_rand_score(l1, l2)
            nmi = normalized_mutual_info_score(l1, l2)
            results.append({
                "Method 1": m1,
                "Method 2": m2,
                "ARI": round(ari, 4),
                "NMI": round(nmi, 4),
            })
    return pd.DataFrame(results)

def get_similarity_scores(other_graph, current_anonymized, current_surface):
    
    other_anonymized = extract_anonymized_triples(other_graph)
    other_surface = extract_surface_triples(other_graph)
    # print()
    # print(current_surface)
    # print()
    # print(other_surface)
    predicted_graph_anon = build_nx_graph(current_anonymized)
    ref_graph_anon = build_nx_graph(other_anonymized)
    predicted_graph_surface = build_nx_graph(current_anonymized)
    ref_graph_surface = build_nx_graph(other_anonymized)
    
    anon_sim = compute_similarity(current_anonymized, other_anonymized)
    surf_sim = compute_similarity(current_surface, other_surface)
    jaccard_sim_anon = jaccard_similarity_wl(predicted_graph_anon, ref_graph_anon)
    jaccard_sim_surface = jaccard_similarity_wl(predicted_graph_surface, ref_graph_surface)
    jaccard_sim = (jaccard_sim_anon+jaccard_sim_surface)/2.0
    return anon_sim, surf_sim, jaccard_sim_anon

def evaluate_clusters_vs_reference(files, cid_map):
    st.subheader("Evaluate Clustering Against Reference")

    reference_csv_path = "D:/MythoGraph/MythoGraph/MythoGraph/LeviStrauss_Gold_JSON_Prep_Clusters.csv"
    
    if reference_csv_path:
        try:
            ref_df = pd.read_csv(reference_csv_path)
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return

        if not {"Title", "Cluster"}.issubset(ref_df.columns):
            st.error("CSV must contain both 'Title' and 'Cluster' columns.")
            return
        title_to_true_cluster = {
            os.path.splitext(str(row["Title"]).replace(".txt", "").strip())[0]: str(row["Cluster"]).strip()
            for _, row in ref_df.iterrows()
        }

        predicted_labels = []
        true_labels = []

        for idx, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            file_base = os.path.splitext(file_name)[0]
            file_base_clean = file_base.replace("_txt_knowledge_graph", "").strip()

            predicted = cid_map[idx]
            true = title_to_true_cluster.get(file_base_clean)

            if true is not None:
                predicted_labels.append(predicted)
                true_labels.append(true)

        if not predicted_labels:
            st.warning("No title matches found between JSONs and the reference CSV.")
            return

        # Compute clustering evaluation metrics
        homogeneity = homogeneity_score(true_labels, predicted_labels)
        completeness = completeness_score(true_labels, predicted_labels)
        v_measure = v_measure_score(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)

        st.subheader("Cluster Evaluation Metrics")
        st.metric("Homogeneity", f"{homogeneity:.3f}")
        st.metric("Completeness", f"{completeness:.3f}")
        st.metric("V-Measure", f"{v_measure:.3f}")
        st.metric("Adjusted Rand Index (ARI)", f"{ari:.3f}")