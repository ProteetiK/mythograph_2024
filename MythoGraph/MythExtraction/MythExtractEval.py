import os
import json
from difflib import SequenceMatcher
from collections import Counter

def load_mythographs(db_folder="MythoGraphDB"):
    mythographs = []
    for root, _, files in os.walk(db_folder):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        mythographs.append(data)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    return mythographs

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_matching_mythograph(text, mythographs, threshold=0.85):
    for myth in mythographs:
        myth_text = myth.get("myth_text", "")
        sim = similar(text.strip().lower(), myth_text.strip().lower())
        if sim >= threshold:
            return myth
    return None

def triples_to_set(triples_list):
    normalized = set()
    for t in triples_list:
        if isinstance(t, dict):
            subj = t.get("source", "").lower()
            obj = t.get("target", "").lower()
            label = t.get("label", "").lower()
        else:
            subj, obj, label = t[0].lower(), t[1].lower(), t[2].lower()
        normalized.add((subj.strip(), obj.strip(), label.strip()))
    return normalized

def evaluate_extraction_accuracy(text, extract_function, db_folder="MythoGraphDB"):
    mythographs = load_mythographs(db_folder)
    matched = find_matching_mythograph(text, mythographs)
    if not matched:
        print("No matching mythograph found in DB.")
        return None
    #print(f"Matching myth found: {matched.get('myth_text')[:60]}...")
    extracted_triples = extract_function(text)
    ref_triples = matched.get("links", [])
    def dict_triple_to_tuple(d):
        return (d.get("source"), d.get("target"), d.get("label"), d.get("motif"))
    ref_triples_tuples = [dict_triple_to_tuple(triple) for triple in ref_triples]
    extracted_triples_tuples = [
        tuple(triple) if not isinstance(triple, dict) else dict_triple_to_tuple(triple)
        for triple in extracted_triples
    ]
    extracted_counter = Counter(extracted_triples_tuples)
    reference_counter = Counter(ref_triples_tuples)
    true_positives_counter = extracted_counter & reference_counter
    false_positives_counter = extracted_counter - reference_counter
    false_negatives_counter = reference_counter - extracted_counter
    tp = sum(true_positives_counter.values())
    fp = sum(false_positives_counter.values())
    fn = sum(false_negatives_counter.values())
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives_counter,
        "false_positives": false_positives_counter,
        "false_negatives": false_negatives_counter,
    }