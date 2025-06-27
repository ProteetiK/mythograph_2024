import os
import json
import numpy as np
import spacy
from rdflib import Graph, URIRef
from urllib.parse import quote
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from karateclub import Graph2Vec
from networkx.readwrite import json_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from MythoMapping.Mapping import MOTIF_DICT, CLUSTER_LABELS

nlp = spacy.load("en_core_web_sm")

GRAPH_MODEL = None
GRAPH_EMBEDDINGS = []
GRAPH_FILES = []
KMEANS = None
N_CLUSTERS = 5

POSITIVE_VERBS = {
    "win", "help", "support", "save", "protect", "love", "heal", "give", "guide", "rescue",
    "accept", "conquer", "build", "find", "create", "gain"
}
NEGATIVE_VERBS = {
    "fight", "attack", "lose", "betray", "destroy", "kill", "steal", "hurt", "defeat", "fail",
    "reject", "escape", "break", "fear"
}

def get_verb_sentiment(verb):
    verb = verb.lower()
    if verb in POSITIVE_VERBS:
        return "positive"
    elif verb in NEGATIVE_VERBS:
        return "negative"
    else:
        return "neutral"

def sentiment_to_cluster_label(sentiment):
    if sentiment == "positive":
        return "Victory"
    elif sentiment == "negative":
        return "Conflict"
    else:
        return "Quest"

def classify_motif(verb, current_motif):
    verb_lower = verb.lower()
    base_motif = MOTIF_DICT.get(verb_lower, None)

    if base_motif and base_motif not in ["Unknown", "0"]:
        return base_motif
    else:
        sentiment = get_verb_sentiment(verb_lower)
        return sentiment_to_cluster_label(sentiment)

def extract_triples_with_nlp(text, window=5):
    doc = nlp(text)
    triples = []
    
    for sent in doc.sents:
        sent_tokens = list(sent)
        for i, token in enumerate(sent_tokens):
            if token.pos_ == "VERB":
                verb = token.lemma_.lower()
                subj = None
                obj = None
                left_tokens = sent_tokens[max(i - window, 0):i]
                right_tokens = sent_tokens[i+1 : i+1+window]
                for t in reversed(left_tokens):
                    if t.pos_ in ("NOUN", "PROPN") and t.tag_ not in ("PRP", "PRP$"):
                        subj = t.text
                        break
                for t in right_tokens:
                    if t.pos_ in ("NOUN", "PROPN") and t.tag_ not in ("PRP", "PRP$"):
                        obj = t.text
                        break
                if subj and obj:
                    triples.append((subj, obj, verb, "TEMP"))
    return triples

def build_rdf_graph(triples, myth_text=None):
    g = Graph()
    for s, o, p, motif in triples:
        if all(isinstance(x, str) and x.strip() for x in (s, o, p, motif)):
            subj = URIRef(quote(s.strip().replace(" ", "_")))
            obj = URIRef(quote(o.strip().replace(" ", "_")))
            pred = URIRef(quote(f"{p}_motif_{motif}".strip().replace(" ", "_")))
            g.add((subj, pred, obj))
    return g

def rdf_to_nx(rdf_graph):
    G = nx.MultiDiGraph()
    for s, p, o in rdf_graph:
        G.add_edge(str(s), str(o), label=str(p), weight=0.9)
    return G

def save_graph_to_json(G, filepath, myth_text=""):
    data = nx.node_link_data(G)
    data["multigraph"] = True
    data["myth_text"] = myth_text
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def draw_graph(G, title="Knowledge Graph"):
    pos = nx.spring_layout(G, seed=123, k=2)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12)

    edge_count = defaultdict(int)
    for u, v, k, data in G.edges(keys=True, data=True):
        rel = data.get('label', '')
        edge_count[(u, v)] += 1
        rad = 0.15 * edge_count[(u, v)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>',
            arrowsize=20,
            edge_color='gray',
        )
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2 + rad * 0.5
        plt.text(mid_x, mid_y, rel, fontsize=9, color='red', ha='center',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_graphs_from_folder(folder_path):
    graphs = []
    global GRAPH_FILES
    GRAPH_FILES = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".json"):
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                data = json.load(f)
            G = json_graph.node_link_graph(data)
            G = nx.convert_node_labels_to_integers(G)
            graphs.append(G)
            GRAPH_FILES.append(fname)
    return graphs

def train_unsupervised_model(graph_folder, n_clusters=N_CLUSTERS):
    global GRAPH_MODEL, GRAPH_EMBEDDINGS, KMEANS
    graphs = load_graphs_from_folder(graph_folder)

    GRAPH_MODEL = Graph2Vec(dimensions=64, workers=2, min_count=1)
    GRAPH_MODEL.fit(graphs)

    GRAPH_EMBEDDINGS = GRAPH_MODEL.get_embedding().astype(np.float64)
    KMEANS = KMeans(n_clusters=n_clusters, random_state=123)
    KMEANS.fit(GRAPH_EMBEDDINGS)

    print(f"Trained on {len(graphs)} myth graphs into {n_clusters} clusters.")

def predict_graph_cluster(G):
    if GRAPH_MODEL is None or KMEANS is None:
        raise ValueError("Model not trained. Run `train_unsupervised_model()` first.")

    all_graphs = load_graphs_from_folder("MythoGraphDB")

    all_graphs.append(G)
    temp_model = Graph2Vec(dimensions=64, workers=2, min_count=1)
    temp_model.fit(all_graphs)

    embeddings = temp_model.get_embedding().astype(np.float64)
    kmeans = KMeans(n_clusters=KMEANS.n_clusters, random_state=123)
    kmeans.fit(embeddings[:-1])
    cluster = kmeans.predict(embeddings[-1].reshape(1, -1))[0]
    return CLUSTER_LABELS.get(cluster, "Unknown")

def visualize_clusters():
    if len(GRAPH_EMBEDDINGS) < 2:
        print("Not enough data.")
        return
    reduced = PCA(n_components=2).fit_transform(GRAPH_EMBEDDINGS)
    labels = KMEANS.labels_
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, label=f"Cluster {labels[i]}")
        plt.annotate(GRAPH_FILES[i], (x, y), fontsize=8)
    plt.title("Myth Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def is_model_trained():
    return GRAPH_MODEL is not None and KMEANS is not None and len(GRAPH_EMBEDDINGS) > 0

def extract_knowledge_graph(myth_text):
    raw_triples = extract_triples_with_nlp(myth_text)
    print("Extracted raw triples:")
    for t in raw_triples:
        print(t)

    temp_rdf = build_rdf_graph([(s, o, p, "TEMP") for s, o, p, _ in raw_triples])
    temp_nx = rdf_to_nx(temp_rdf)
    temp_nx = nx.convert_node_labels_to_integers(temp_nx)

    if is_model_trained():
        predicted_motif = predict_graph_cluster(temp_nx)
        print(f"Predicted motif: {predicted_motif}")
    else:
        predicted_motif = "Unknown"
        print("Model not trained. Using default motif.")
    final_triples = []
    for s, o, p, _ in raw_triples:
        motif = classify_motif(p, predicted_motif)
        final_triples.append((s, o, p, motif))
    final_rdf = build_rdf_graph(final_triples, myth_text)
    final_nx = rdf_to_nx(final_rdf)
    return final_nx


def export_graph_as_custom_json(G, myth_text):
    data = {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [{"id": str(n)} for n in G.nodes()],
        "links": [],
        "myth_text": myth_text.strip()
    }

    for u, v, data_dict in G.edges(data=True):
        full_label = data_dict.get("label", "")
        weight = data_dict.get("weight", 0.7)

        if "_motif_" in full_label:
            parts = full_label.split("_motif_")
            label = parts[0]
            motif = parts[1]
        else:
            label = full_label
            motif = "Unknown"

        data["links"].append({
            "source": str(u),
            "target": str(v),
            "label": label,
            "motif": motif,
            "weight": round(weight, 2)
        })

    return data