import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm
import os
import numpy as np
import spacy
import networkx as nx
from sklearn.cluster import KMeans
from MythoMapping.Mapping import CLUSTER_LABELS
from MythGraph.MythGraphDraw import load_graphs_from_folder
import glob
import pickle
from sklearn.cluster import KMeans
import networkx as nx

nlp = spacy.load("en_core_web_sm")
GRAPH_MODEL = None
GRAPH_EMBEDDINGS = []
GRAPH_FILES = []
KMEANS = None
N_CLUSTERS = 5

model_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/motif_classifier.pt"
unsupervised_model_path = "D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/graph_model.pkl"

class MotifTripletDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['source']} {item['label']} {item['target']}"
        label = self.label_encoder.transform([item['motif']])[0]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

class MotifClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)

def load_all_json_from_folder(folder_path):
    data = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            try:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON in {file}: {e}")
    return data

def train_motif_model(dataset_dir="D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB/",
                      save_model_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/motif_classifier.pt",
                      batch_size=5, epochs=15, lr=2e-5):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    raw_data = load_all_json_from_folder(dataset_dir)
    print(raw_data)
    motifs = []
    all_links = []    

    for item in raw_data:
        links = item.get("links", [])
        all_links.extend(links)
        for link in links:
            if "motif" in link:
                motifs.append(link["motif"])
    print(motifs)
    unique_motifs = sorted(set(motifs))
    print(unique_motifs)
    label_encoder = LabelEncoder()
    label_encoder.fit(motifs)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    label_encoder_path = os.path.join(os.path.dirname(save_model_path), "label_encoder.json")
    with open(label_encoder_path, "w") as f:
        label_dict = {cls: int(idx) for cls, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
        json.dump(label_dict, f)

    dataset = MotifTripletDataset(all_links, tokenizer, label_encoder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if os.path.exists(save_model_path):
        print(f"Loading existing model from {save_model_path}...")
        checkpoint = torch.load(save_model_path, map_location=device)
        saved_num_classes = checkpoint['classifier.weight'].shape[0]
        current_num_classes = len(set(motifs))
        if saved_num_classes == current_num_classes:
            print("Model class count matches. Loading model weights.")
            model = MotifClassifier(num_classes=current_num_classes).to(device)
            model.load_state_dict(checkpoint)
        else:
            print(f"Mismatch in class count (saved: {saved_num_classes}, current: {current_num_classes}).")
            print("Reinitializing model from scratch.")
            model = MotifClassifier(num_classes=current_num_classes).to(device)
    else:
        print("No existing model found. Initializing new model...")
        model = MotifClassifier(num_classes=len(set(motifs))).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

def load_motif_classifier(model_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/motif_classifier.pt",
                          encoder_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/label_encoder.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(encoder_path, "r") as f:
        label_map = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(label_map.keys()))
    model = MotifClassifier(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, label_encoder

def classify_motif_with_model(model, label_encoder, tokenizer, subject, predicate, obj, device="cpu"):
    text = f"{subject} {predicate} {obj}"
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    motif = label_encoder.inverse_transform([predicted_idx])[0]
    return motif

def save_unsupervised_model():
    with open(unsupervised_model_path, "wb") as f:
        pickle.dump({
            "graph_model": GRAPH_MODEL,
            "kmeans": KMEANS,
            "embeddings": GRAPH_EMBEDDINGS
        }, f)

def extract_graph_features(graph):
    simple_graph = nx.Graph(graph)  # Convert multigraph to simple graph
    return [
        graph.number_of_nodes(),
        graph.number_of_edges(),
        nx.density(graph),
        nx.average_clustering(simple_graph),  # Use simple graph here
    ]

def train_unsupervised_model(graph_folder, n_clusters=N_CLUSTERS):
    global GRAPH_MODEL, GRAPH_EMBEDDINGS, KMEANS
    graphs = load_graphs_from_folder(graph_folder)
    GRAPH_EMBEDDINGS = [extract_graph_features(g) for g in graphs]
    KMEANS = KMeans(n_clusters=n_clusters, random_state=123)
    KMEANS.fit(GRAPH_EMBEDDINGS)
    print(f"Trained on {len(graphs)} myth graphs into {n_clusters} clusters.")
    save_unsupervised_model()

def load_unsupervised_model(model_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/graph_model.pkl"):
    global GRAPH_MODEL, KMEANS, GRAPH_EMBEDDINGS

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Unsupervised model not found at: {model_path}")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    GRAPH_MODEL = data.get("graph_model")
    KMEANS = data.get("kmeans")
    GRAPH_EMBEDDINGS = data.get("embeddings")

    if GRAPH_MODEL is None or KMEANS is None or GRAPH_EMBEDDINGS is None:
        raise ValueError("Model file missing components.")

def cluster_graphs(graphs, n_clusters=3):
    feature_vectors = [extract_graph_features(g) for g in graphs]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_vectors)
    return labels, kmeans

def predict_graph_cluster(G, kmeans_model):
    features = [extract_graph_features(G)]
    return kmeans_model.predict(features)[0]

def is_model_trained():
    return GRAPH_MODEL is not None and KMEANS is not None and len(GRAPH_EMBEDDINGS) > 0

def is_model_saved():
    return os.path.exists(model_path)