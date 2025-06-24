import networkx as nx
import matplotlib.pyplot as plt
import re
import ast
import os
import json
from collections import defaultdict
import ollama
from networkx.readwrite import json_graph

W1 = 0.7

def llm_extract_with_ollama(chunk):
    prompt = f"""
You are a network graph maker who extracts named entities and semantic relationships.
Extract only explicitly mentioned proper nouns (persons, animals) and their direct relations.
Respond ONLY with a Python list of (subject, object, relation) triples.
Example format: [("Wolf", "Lamb", "met"), ("Wolf", "Lamb", "accused")]

Text:
\"\"\"{chunk}\"\"\"
"""
    response = ollama.chat(model='mistral', messages=[
        {"role": "user", "content": prompt}
    ])
    content = response['message']['content']
    try:
        match = re.search(r'\[\s*\(.*?\)\s*\]', content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
        else:
            print("No valid list found in LLM output.")
            print("Raw output:", content)
            return []
    except Exception as e:
        print("LLM output parsing failed:", e)
        print("Raw output:", content)
        return []

def extract_knowledge_graph(text):
    words = text.split()
    chunks = [" ".join(words[i:i+100]) for i in range(0, len(words), 100)]
    edges = []
    for chunk in chunks:
        triples = llm_extract_with_ollama(chunk)
        for triple in triples:
            if (
                isinstance(triple, (list, tuple)) and
                len(triple) == 3 and
                all(isinstance(t, str) and t.strip() for t in triple)
            ):
                c1, c2, rel = triple
                edges.append((c1.strip(), c2.strip(), {'weight': W1, 'label': rel.strip()}))
            else:
                print("Invalid triple skipped:", triple)
    return create_graph(edges)

def create_graph(edge_list):
    G = nx.MultiDiGraph()
    for c1, c2, data in edge_list:
        if c1 is not None and c2 is not None:
            G.add_edge(c1, c2, **data)
    return G

def draw_graph(G, title="Knowledge Graph"):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=2000)
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

def save_graph_to_json(G, filename):
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    data = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def is_isomorphic_with_db(G_input, db_folder):
    from networkx.algorithms import isomorphism
    GM_input = nx.convert_node_labels_to_integers(G_input, label_attribute="name")
    for filename in os.listdir(db_folder):
        if filename.endswith(".json"):
            with open(os.path.join(db_folder, filename), "r") as f:
                data = json.load(f)
                G_db = nx.node_link_graph(data)
                GM_db = nx.convert_node_labels_to_integers(G_db, label_attribute="name")
                matcher = isomorphism.GraphMatcher(
                    GM_input, GM_db,
                    edge_match=lambda e1, e2: e1['label'] == e2['label']
                )
                if matcher.is_isomorphic():
                    return filename.replace(".json", "")
    return None
