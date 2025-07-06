import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from MythoMapping.Mapping import MOTIF_DICT
from MythModelTrain.MotifTrainer import load_motif_classifier, is_model_trained, predict_graph_cluster
from MythExtraction.MythExtract import extract_triples_with_nlp
from MythExtraction.MythExtractEval import evaluate_extraction_accuracy
from MythGraph.MythGraphDraw import load_graphs_from_folder, build_rdf_graph, rdf_to_nx, build_nx_graph
import torch
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

POSITIVE_VERBS = {
    "win", "help", "support", "save", "protect", "love", "heal", "give", "guide", "rescue",
    "accept", "conquer", "build", "find", "create", "gain"
}
NEGATIVE_VERBS = {
    "fight", "attack", "lose", "betray", "destroy", "kill", "steal", "hurt", "defeat", "fail",
    "reject", "escape", "break", "fear", "hostile"
}

def get_primary_verb_hypernym(verb_lemma):
    synsets = wn.synsets(verb_lemma, pos=wn.VERB)
    if not synsets:
        return verb_lemma
    primary_syn = synsets[0]
    hypernyms = primary_syn.hypernyms()
    if hypernyms:
        return hypernyms[0].lemma_names()[0]
    return primary_syn.lemma_names()[0]


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

def sentiment_to_cluster_label(sentiment):
    if sentiment == "positive":
        return "Victory"
    elif sentiment == "negative":
        return "Conflict"
    else:
        return "Quest"

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

def classify_motif(verb, current_motif=None):
    verb_lower = verb.lower()
    base_motif = MOTIF_DICT.get(verb_lower)

    if base_motif and base_motif not in ["Unknown", "0"]:
        return base_motif

    verb_emb = get_verb_embedding(verb_lower)
    sims = {label: cosine_similarity([verb_emb], [vec])[0][0] for label, vec in CLUSTER_EMBEDDINGS.items()}
    best_label = max(sims, key=sims.get)

    if sims[best_label] < 0.4:
        return sentiment_to_motif_label(verb_lower)

    return best_label

motif_model = None
label_encoder = None
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def classify_motif(verb, predicted_cluster_motif=None, subject=None, obj=None):
    global motif_model, label_encoder
    verb_lower = verb.lower()

    # 1. Rule-based dictionary lookup
    #base_motif = MOTIF_DICT.get(verb_lower)
    #if base_motif and base_motif not in ["Unknown", "0"]:
     #   return base_motif

    # 2. Try fine-tuned BERT classifier on triplet
    if subject and obj:
        if motif_model is None or label_encoder is None:
            try:
                motif_model, label_encoder = load_motif_classifier()
            except Exception as e:
                print(f"Error loading motif model: {e}")
                motif_model = None  # Fallback protection
        if motif_model:
            text = f"{subject} {verb} {obj}"
            encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
            input_ids = encoding["input_ids"].to(motif_model.device)
            attention_mask = encoding["attention_mask"].to(motif_model.device)
            with torch.no_grad():
                logits = motif_model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                top_prob, pred_label = torch.max(probs, dim=1)
            if top_prob.item() >= 0.6:
                return label_encoder.inverse_transform([pred_label.item()])[0]

    # 3. SBERT similarity to motif clusters
    verb_emb = get_verb_embedding(verb_lower)
    sims = {label: cosine_similarity([verb_emb], [vec])[0][0] for label, vec in CLUSTER_EMBEDDINGS.items()}
    best_label = max(sims, key=sims.get)
    if sims[best_label] >= 0.4:
        return best_label

    # 4. Sentiment fallback
    return sentiment_to_motif_label(verb_lower)

def extract_knowledge_graph(myth_text):
    raw_triples = extract_triples_with_nlp(myth_text)
    evaluate_extraction_accuracy(myth_text, extract_triples_with_nlp, db_folder="MythoGraphDB")
    # temp_rdf = build_rdf_graph([(s, o, p, "TEMP") for s, o, p, *rest in raw_triples])
    # temp_nx = rdf_to_nx(temp_rdf)
    temp_triples = [(s, o, p, "TEMP", 0.9) for s, o, p, *rest in raw_triples]
    temp_nx = build_nx_graph(temp_triples)
    if is_model_trained():
        predicted_motif = predict_graph_cluster(temp_nx)
        print(f"Predicted motif cluster: {predicted_motif}")
    else:
        predicted_motif = "Unknown"
        print("Model not trained. Using default motif.")
    final_triples = []
    for s, o, p, *rest in raw_triples:
        motif = classify_motif(p, predicted_motif, subject=s, obj=o)
        final_triples.append((s, o, p, motif, 0.9))
    # final_rdf = build_rdf_graph(final_triples, myth_text)
    # final_nx = rdf_to_nx(final_rdf)
    
    final_nx = build_nx_graph(final_triples)
    return final_nx
