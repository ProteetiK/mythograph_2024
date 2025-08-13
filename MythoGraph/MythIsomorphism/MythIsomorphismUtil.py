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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, fowlkes_mallows_score, adjusted_mutual_info_score
import time
from networkx.algorithms import isomorphism as iso
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from MythGraph.MythGraphDraw import build_nx_graph
import re
import random
import json

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

folder = os.path.abspath("MythoGraphDB")

def cluster_color(cid):
    random.seed(cid)
    return f"hsl({random.randint(0, 360)}, 70%, 60%)"

def load_graphs(files):
    graphs = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
                graphs.append(data)
        except json.JSONDecodeError as e:
            st.error(f"JSONDecodeError in file: `{f}`\n\n**Error:** {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error in file: `{f}`\n\n**Error:** {str(e)}")
            st.stop()
    return graphs



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

def compute_similarity(triples1, triples2, weight_hungarian=0.15, weight_ordered=0.85):
    if not triples1 or not triples2:
        return 0.0
    sents1 = [triple_to_sentence(t) for t in triples1]
    sents2 = [triple_to_sentence(t) for t in triples2]
    emb1 = sbert_model.encode(sents1)
    emb2 = sbert_model.encode(sents2)
    sim_matrix = cosine_similarity(emb1, emb2)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    hungarian_scores = sim_matrix[row_ind, col_ind]
    hungarian_avg = np.mean(hungarian_scores)
    min_len = min(len(emb1), len(emb2))
    ordered_scores = [
        cosine_similarity([emb1[i]], [emb2[i]])[0][0]
        for i in range(min_len)
    ]
    ordered_avg = np.mean(ordered_scores)
    final_score = (weight_hungarian * hungarian_avg + weight_ordered * ordered_avg)

    return round(final_score * 100, 2)

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

def extract_oppositions(links):
    oppositions = [
        ("Victory", "Defeat"),
        ("Defeat", "Victory"),
        ("Conflict", "Defeat"),
        ("Conflict", "Victory"),
        ("Guidance", "Trickery"),
        ("Appeal", "Conflict"),
        ("Appeal", "Guidance"),
        ("Appeal", "Trickery"),
        ("Quest", "Victory"),
        ("Quest", "Defeat")
    ]

    motifs_sequence = []
    seen = set()
    for link in links:
        motif = link.get("motif")
        if motif and motif not in seen:
            motifs_sequence.append(motif)
            seen.add(motif)

    motif_counts = Counter(link.get("motif") for link in links if link.get("motif"))

    opposition_freq = {}
    counted_pairs = set()

    for a, b in oppositions:
        if a in motifs_sequence and b in motifs_sequence:
            key = f"{a} vs. {b}"
            if key not in counted_pairs and f"{b} vs. {a}" not in counted_pairs:
                total = motif_counts[a] + motif_counts[b]
                opposition_freq[key] = total
                counted_pairs.add(key)

    return opposition_freq

def compute_opposition_similarity(current_graph_json, other_graph_json):
    opp_current = set(extract_oppositions(current_graph_json.get("links", [])).keys())
    opp_other = set(extract_oppositions(other_graph_json.get("links", [])).keys())

    if not opp_current or not opp_other:
        return -1.0

    intersection = opp_current & opp_other
    union = opp_current | opp_other

    similarity = len(intersection) / len(union)
    return round(similarity * 100, 2)

def compute_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    
    emb1 = sbert_model.encode([text1])[0]
    emb2 = sbert_model.encode([text2])[0]

    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return round(similarity * 100, 2)

def get_similarity_scores(current_graph_json, other_graph_json):
    
    current_anonymized = extract_anonymized_triples(current_graph_json)
    current_surface = extract_surface_triples(current_graph_json)
    other_anonymized = extract_anonymized_triples(other_graph_json)
    other_surface = extract_surface_triples(other_graph_json)

    predicted_graph_anon = build_nx_graph(current_anonymized)
    ref_graph_anon = build_nx_graph(other_anonymized)

    predicted_graph_surface = build_nx_graph(current_anonymized)
    ref_graph_surface = build_nx_graph(other_anonymized)
    
    anon_sim = compute_similarity(current_anonymized, other_anonymized)
    surf_sim = compute_similarity(current_surface, other_surface)

    opposition_sim = compute_opposition_similarity(current_graph_json, other_graph_json)

    jaccard_sim_anon = jaccard_similarity_wl(predicted_graph_anon, ref_graph_anon)
    jaccard_sim_surface = jaccard_similarity_wl(predicted_graph_surface, ref_graph_surface)

    jaccard_sim = (jaccard_sim_anon+jaccard_sim_surface)/2.0
    return anon_sim, surf_sim, opposition_sim, jaccard_sim

def strip_extension(filename):
    return os.path.splitext(filename.strip())[0].lower()

def normalize_title(title):
    title = os.path.splitext(title)[0]
    title = title.lower()
    title = re.sub(r'_knowledge_graph$', '', title)
    title = title.replace("_", " ").replace("-", " ")
    title = title.replace(" txt", "").strip()
    title = re.sub(r'\s+', ' ', title)

    return title
    
def compare_with_ground_truth(files, computed_clusters, label_csv_path):
    try:
        df_labels = pd.read_csv(label_csv_path)

        title_to_index = {
            normalize_title(os.path.basename(f)): i
            for i, f in enumerate(files)
        }

        ground_truth_labels = []
        predicted_labels = []
        not_found = []

        for _, row in df_labels.iterrows():
            csv_title = normalize_title(row['Title'])
            cluster = row['Cluster']
            if csv_title in title_to_index:
                idx = title_to_index[csv_title]
                ground_truth_labels.append(cluster)
                predicted_labels.append(computed_clusters[idx])
            else:
                not_found.append(row['Title'])

        if not ground_truth_labels:
            st.warning("No matching files between clustering and ground truth.")
            return

        if not_found:
            st.warning(f"Titles not matched from ground truth CSV:\n{not_found}")

        le = LabelEncoder()
        gt_encoded = le.fit_transform(ground_truth_labels)
        pred_encoded = le.fit_transform(predicted_labels)

        homogeneity = homogeneity_score(ground_truth_labels, predicted_labels)
        completeness = completeness_score(ground_truth_labels, predicted_labels)
        v_measure = v_measure_score(ground_truth_labels, predicted_labels)
        ari = adjusted_rand_score(gt_encoded, pred_encoded)
        nmi = adjusted_mutual_info_score(gt_encoded, pred_encoded)
        fms = fowlkes_mallows_score(gt_encoded, pred_encoded)

        st.subheader("Cluster Comparison with Ground Truth")
        st.metric("Homogeneity", f"{homogeneity:.3f}")
        st.metric("Completeness", f"{completeness:.3f}")
        st.metric("V-Measure", f"{v_measure:.3f}")
        st.metric("Adjusted Rand Index (ARI)", f"{ari:.3f}")
        st.metric("Normalized Mutual Info (NMI)", f"{nmi:.3f}")
        st.metric("Fowlkes-Mallows Score", f"{fms:.3f}")

        return homogeneity, completeness, v_measure, ari, nmi, fms

    except Exception as e:
        st.error(f"Error comparing with ground truth: {e}")