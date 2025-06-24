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
from Mapping.Mapping import MOTIF_DICT

nlp = spacy.load("en_core_web_sm")
def classify_motif(verb_lemma):
    return MOTIF_DICT.get(verb_lemma.lower(), "General")

def extract_triples_with_nlp(text):
    doc = nlp(text)
    triples = []
    characters = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG"} and ent.text not in characters:
            characters.append(ent.text)
    for token in doc:
        if token.pos_ == "PROPN" and token.text not in characters:
            characters.append(token.text)
    last_subject = None
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB":
                verb = token.lemma_.lower()
                subj = None
                obj = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass", "agent") and child.pos_ in ("NOUN", "PROPN", "PRON"):
                        if child.pos_ == "PRON" and last_subject:
                            subj = last_subject
                        elif child.text in characters:
                            subj = child.text
                            last_subject = subj
                        else:
                            subj = child.text
                for child in token.child:
                    if child.dep_ in ("dobj", "pobj", "dative", "attr", "xcomp", "ccomp"):
                        if child.pos_ == "PRON" and last_subject:
                            obj = last_subject
                        elif child.text in characters:
                            obj = child.text
                        else:
                            inner_noun = next((w.text for w in child.subtree if w.pos_ in ("NOUN", "PROPN", "PRON")), None)
                            if inner_noun:
                                obj = inner_noun
                            else:
                                obj = child.text
                if subj and obj:
                    motif = classify_motif(verb)
                    triples.append((subj, obj, verb, motif))
    return triples

def build_rdf_graph(triples, myth_text=None):
    g = Graph()
    for s, o, p, motif in triples:
        if all(isinstance(x, str) and x.strip() for x in (s, o, p, motif)):
            subj = URIRef(quote(s.strip().replace(" ", "_")))
            obj = URIRef(quote(o.strip().replace(" ", "_")))
            combined_pred = URIRef(quote(f"{p}_motif_{motif}".strip().replace(" ", "_")))
            g.add((subj, combined_pred, obj))
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

def encode_graph(G):
    model = Graph2Vec(dimensions=64, workers=2, min_count=1)
    model.fit([G])
    return model.get_embedding()[0]

def is_isomorphic_with_db(G_input, db_folder, threshold=0.8):
    vec_input = encode_graph(G_input)
    best_score, best_match = 0, None
    for fname in os.listdir(db_folder):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(db_folder, fname)) as f:
            data = json.load(f)
        G_db = json_graph.node_link_graph(data)
        vec_db = encode_graph(G_db)
        score = np.dot(vec_input, vec_db) / (np.linalg.norm(vec_input) * np.linalg.norm(vec_db))
        if score > best_score and score >= threshold:
            best_score, best_match = score, fname.replace(".json", "")
    return best_match

def extract_knowledge_graph(myth_text):
    triples = extract_triples_with_nlp(myth_text)
    print("Extracted Triples and Motifs:")
    for t in triples:
        print(t)
    rdf = build_rdf_graph(triples, myth_text)
    G = rdf_to_nx(rdf)
    return G