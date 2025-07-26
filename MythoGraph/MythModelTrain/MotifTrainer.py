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
import glob
import pickle
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from MythoMapping.Mapping import CLUSTER_LABELS
from MythGraph.MythGraphDraw import load_graphs_from_folder

nlp = spacy.load("en_core_web_sm")

model_path = "D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/motif_classifier.pt"
unsupervised_model_path = "D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/graph_model.pkl"

GRAPH_MODEL = None
GRAPH_EMBEDDINGS = []
KMEANS = None
N_CLUSTERS = 5

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

def extract_graph_features(graph):
    simple_graph = nx.Graph(graph)
    return [
        graph.number_of_nodes(),
        graph.number_of_edges(),
        nx.density(graph),
        nx.average_clustering(simple_graph),
    ]

def train_unsupervised_model(graph_folder, n_clusters=N_CLUSTERS):
    global GRAPH_MODEL, GRAPH_EMBEDDINGS, KMEANS
    graphs = load_graphs_from_folder(graph_folder)
    GRAPH_EMBEDDINGS = [extract_graph_features(g) for g in graphs]
    KMEANS = KMeans(n_clusters=n_clusters, random_state=123)
    KMEANS.fit(GRAPH_EMBEDDINGS)
    print(f"Trained on {len(graphs)} myth graphs into {n_clusters} clusters.")
    with open(unsupervised_model_path, "wb") as f:
        pickle.dump({
            "kmeans": KMEANS,
            "embeddings": GRAPH_EMBEDDINGS
        }, f)

def load_unsupervised_model(model_path=unsupervised_model_path):
    global KMEANS, GRAPH_EMBEDDINGS
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Unsupervised model not found at: {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    KMEANS = data.get("kmeans")
    GRAPH_EMBEDDINGS = data.get("embeddings")
    if KMEANS is None or GRAPH_EMBEDDINGS is None:
        raise ValueError("Model file missing components")

def predict_graph_cluster(G, kmeans_model):
    features = [extract_graph_features(G)]
    return kmeans_model.predict(features)[0]

def evaluate_motif_model(model, dataloader, device, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    unique_labels = np.unique(all_labels + all_preds).tolist()

    present_class_names = [label_encoder.classes_[i] for i in unique_labels]

    precision = precision_score(all_labels, all_preds, labels=unique_labels, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, labels=unique_labels, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, labels=unique_labels, average='weighted', zero_division=0)
    
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=present_class_names,
        zero_division=0
    )

    print("Evaluation metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:\n", report)

from sklearn.utils.class_weight import compute_class_weight

def train_motif_model(dataset_dir="D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB",
                      val_dataset_dir="D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB/ValidationDataSet",
                      save_model_path=model_path,
                      batch_size=10,
                      epochs=51,
                      lr=2e-5):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    raw_data = load_all_json_from_folder(dataset_dir)
    motifs = []
    all_links = []

    for item in raw_data:
        links = item.get("links", [])
        all_links.extend(links)
        for link in links:
            if "motif" in link:
                motifs.append(link["motif"])

    label_encoder = LabelEncoder()
    label_encoder.fit(motifs)
    
    num_classes = len(label_encoder.classes_)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=label_encoder.transform(motifs)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    label_encoder_path = os.path.join(os.path.dirname(save_model_path), "label_encoder.json")
    with open(label_encoder_path, "w") as f:
        label_dict = {cls: int(idx) for cls, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
        json.dump(label_dict, f)

    dataset = MotifTripletDataset(all_links, tokenizer, label_encoder)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if val_dataset_dir is not None:
        val_raw_data = load_all_json_from_folder(val_dataset_dir)
        val_links = []
        for item in val_raw_data:
            val_links.extend(item.get("links", []))
        val_dataset = MotifTripletDataset(val_links, tokenizer, label_encoder)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    if os.path.exists(save_model_path):
        print(f"Loading existing model from {save_model_path}...")
        checkpoint = torch.load(save_model_path, map_location=device)
        saved_num_classes = checkpoint['classifier.weight'].shape[0]
        if saved_num_classes == num_classes:
            model = MotifClassifier(num_classes=num_classes).to(device)
            model.load_state_dict(checkpoint)
            print("Loaded existing model weights.")
        else:
            print(f"Class count mismatch (saved: {saved_num_classes}, current: {num_classes}), initializing new model.")
            model = MotifClassifier(num_classes=num_classes).to(device)
    else:
        print("No existing model found. Initializing new model...")
        model = MotifClassifier(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

        if val_loader is not None and epoch % 10 == 0:
            print(f"Epoch {epoch+1} Validation:")
            evaluate_motif_model(model, val_loader, device, label_encoder)
            scheduler.step(avg_loss)

    torch.save(model.state_dict(), save_model_path)
    print(f"Motif classifier model saved to {save_model_path}")

def load_motif_model(model_path=model_path,
                          encoder_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if encoder_path is None:
        encoder_path = os.path.join(os.path.dirname(model_path), "label_encoder.json")
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

def is_model_trained():
    return (os.path.exists(model_path) and os.path.exists(unsupervised_model_path))

def train_all_models(json_dataset_dir, graph_folder,
                     motif_save_path=model_path,
                     graph_save_path=unsupervised_model_path,
                     n_clusters=N_CLUSTERS):
    print("Training motif classifier...")
    train_motif_model(json_dataset_dir, save_model_path=motif_save_path)

    print("Training unsupervised graph clustering model...")
    train_unsupervised_model(graph_folder, n_clusters=n_clusters)

def is_model_saved():
    return os.path.exists(model_path)