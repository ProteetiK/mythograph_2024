import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from MythoMapping.Mapping import MOTIF_DICT
from MythModelTrain.MotifTrainer import load_motif_model, classify_motif_with_model, load_motif_model
from MythExtraction.MythExtract import extract_triples_combined
from MythExtraction.MythExtractEval import evaluate_extraction_accuracy
from MythGraph.MythGraphDraw import build_nx_graph
from MythoMapping.Mapping import MOTIF_DICT
import torch
from transformers import BertTokenizer
from collections import Counter

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("C:/Users/KIIT/all-MiniLM-L6-v2/")

POSITIVE_VERBS = {
    "win", "help", "support", "save", "protect", "love", "heal", "give", "guide", "rescue",
    "accept", "conquer", "build", "find", "create", "gain"
}
NEGATIVE_VERBS = {
    "fight", "attack", "lose", "betray", "destroy", "kill", "steal", "hurt", "defeat", "fail",
    "reject", "escape", "break", "fear", "hostile"
}

def generate_cluster_seeds(motif_dict):
    clusters = defaultdict(set)
    for verb, cluster in motif_dict.items():
        clusters[cluster].add(verb)
    return {label: list(verbs) for label, verbs in clusters.items()}

CLUSTER_SEEDS = generate_cluster_seeds(MOTIF_DICT)

def build_cluster_embeddings():
    cluster_vecs = {}
    for label, verbs in CLUSTER_SEEDS.items():
        vecs = sbert_model.encode(verbs, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        cluster_vecs[label] = np.mean(vecs, axis=0)
    return cluster_vecs

CLUSTER_EMBEDDINGS = build_cluster_embeddings()

verb_embedding_cache = {}

def get_verb_embedding(verb):
    verb = verb.lower()
    if verb not in verb_embedding_cache:
        emb = sbert_model.encode([verb], convert_to_numpy=True, normalize_embeddings=True)[0]
        verb_embedding_cache[verb] = emb
    return verb_embedding_cache[verb]

def get_verb_sentiment(verb):
    verb = verb.lower()
    if verb in POSITIVE_VERBS:
        return "positive"
    elif verb in NEGATIVE_VERBS:
        return "negative"
    else:
        return "neutral"

def sentiment_to_motif_label(verb):
    sentiment = get_verb_sentiment(verb)
    verb_lower = verb.lower()

    if sentiment == "positive":
        if verb_lower in {"guide", "help", "support"}:
            return "Guidance"
        elif verb_lower in {"communicate", "speak", "share"}:
            return "Communication"
        elif verb_lower in {"win", "conquer", "save"}:
            return "Victory"
        elif verb_lower in {"ask", "beg", "appeal"}:
            return "Appeal"
        else:
            return "Quest"

    elif sentiment == "negative":
        if verb_lower in {"lose", "fail", "retreat"}:
            return "Defeat"
        elif verb_lower in {"betray", "deceive", "trick"}:
            return "Trickery"
        elif verb_lower in {"compete", "rival"}:
            return "Competition"
        else:
            return "Conflict"

    return "General"

motif_model = None
label_encoder = None
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def classify_motif(verb, predicted_cluster_motif=None, subject=None, obj=None):
    global motif_model, label_encoder
    verb_lower = verb.lower()

    if subject and obj:
        if motif_model is None or label_encoder is None:
            try:
                motif_model, label_encoder = load_motif_model()
            except Exception as e:
                print(f"Error loading motif model: {e}")
                motif_model = None
        if motif_model:
            text = f"{subject} {verb} {obj}"
            encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
            device = next(motif_model.parameters()).device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.no_grad():
                logits = motif_model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                top_prob, pred_label = torch.max(probs, dim=1)
            if top_prob.item() >= 0.5:
                return label_encoder.inverse_transform([pred_label.item()])[0]

    verb_emb = get_verb_embedding(verb_lower)
    sims = {label: cosine_similarity([verb_emb], [vec])[0][0] for label, vec in CLUSTER_EMBEDDINGS.items()}
    best_label = max(sims, key=sims.get)
    if sims[best_label] >= 0.4:
        return best_label

    sentiment_motif = sentiment_to_motif_label(verb_lower)
    if sentiment_motif != "General":
        return sentiment_motif

    # base_motif = MOTIF_DICT.get(verb_lower)
    # if base_motif and base_motif not in ["Unknown", "0"]:
    #     return base_motif

    return "General"

KMEANS = None

def extract_knowledge_graph(myth_text_raw):
    myth_text = myth_text_raw.title()
    raw_triples = extract_triples_combined(myth_text)
    evaluate_extraction_accuracy(myth_text, extract_triples_combined, db_folder="MythoGraphDB")
    temp_triples = [(s, o, p, "TEMP", 0.9) for s, o, p, *rest in raw_triples]
    temp_nx = build_nx_graph(temp_triples)
    model, label_encoder = load_motif_model()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predicted_cluster = None
    final_triples = []
    for s, o, p, *rest in raw_triples:
        if model is not None and label_encoder is not None:
            motif = classify_motif_with_model(model, label_encoder, tokenizer, s, p, o, device=device)
        else:
            motif = classify_motif(p, predicted_cluster, subject=s, obj=o)
        if MOTIF_DICT.get(p) != None:
            motif = MOTIF_DICT.get(p)

        final_triples.append((s, o, p, motif, 0.9))

    final_nx = build_nx_graph(final_triples)
    return final_nx

def extract_oppositions(links):
    oppositions = [
        ("Victory", "Defeat"),
        ("Guidance", "Trickery"),
        ("Appeal", "Conflict"),
        ("Appeal", "Competition")
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