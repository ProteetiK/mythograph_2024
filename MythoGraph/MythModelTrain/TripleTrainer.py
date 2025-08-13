import os, json, torch, string
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import (
    BertTokenizerFast, BertForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import string

LABELS = ["O", "B-source", "I-source", "B-PRED", "I-PRED", "B-OBJ", "I-OBJ"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def load_data_from_json_folder(folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                try:
                    file_data = json.load(f)
                    data.extend(file_data if isinstance(file_data, list) else [file_data])
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
    return data

def find_token_spans(text, phrase, offset_mapping):
    phrase = phrase.strip().lower()
    start = text.lower().find(phrase)
    if start == -1:
        return []

    end = start + len(phrase)
    span = []
    for i, (token_start, token_end) in enumerate(offset_mapping):
        if token_end == 0 and token_start == 0:
            continue
        if token_start >= end:
            break
        if token_end > start:
            span.append(i)
    return span


def preprocess_for_training(raw_data):
    processed = []
    for item in raw_data:
        text   = item.get("myth_text", "")
        links  = item.get("links", [])

        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=64)
        ids = enc["input_ids"]
        off = enc["offset_mapping"]
        ids  = ids[1:-1]
        off  = off[1:-1]
        toks = tokenizer.convert_ids_to_tokens(ids)
        labs = ["O"] * len(toks)

        for i, tok in enumerate(toks):
            if not is_valid_token(tok):
                labs[i] = None

        for link in links:
            for phrase, pref in [(link["source"], "source"),
                                 (link["label"],  "PRED"),
                                 (link["target"], "OBJ")]:
                span = find_token_spans(text, phrase, off)
                for j, idx in enumerate(span):
                    if labs[idx] is None or labs[idx] != "O":
                        continue
                    labs[idx] = f"B-{pref}" if j == 0 else f"I-{pref}"

        filtered_toks_labels = [(tok, lab) for tok, lab in zip(toks, labs) if lab is not None]
        toks, labs = zip(*filtered_toks_labels) if filtered_toks_labels else ([], [])

        processed.append({"tokens": toks, "labels": labs})
    return processed

class TripleExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"][:self.max_len - 2]
        labels = self.data[idx]["labels"][:self.max_len - 2]

        input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        label_ids = [-100] + [LABEL2ID[tag] for tag in labels] + [-100]

        pad_len = self.max_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        label_ids += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(label_ids),
        }

def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }

class BertTokenClassifier(nn.Module):
    def __init__(self, num_labels=len(LABELS)):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def train_triple_extractor(
    data_folder="D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB/",
    save_model_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/triple_classifier.pt",
    val_data_folder="D:/MythoGraph/MythoGraph/MythoGraph/MythoGraphDB/ValidationDataSet/",
    batch_size=10,
    epochs=51,
    max_len=64,
    lr=2e-5,
    device=None,
    freeze_epochs=0,
):
    weights = torch.ones(len(LABELS))
    weights[LABEL2ID["O"]] = 0.05

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_train = load_data_from_json_folder(data_folder)
    raw_val = load_data_from_json_folder(val_data_folder)

    train_processed = preprocess_for_training(raw_train)
    val_processed = preprocess_for_training(raw_val)

    train_ds = TripleExtractionDataset(train_processed, tokenizer, max_len)
    val_ds = TripleExtractionDataset(val_processed, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BertTokenClassifier().to(device)

    for p in model.bert.bert.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_loader) * epochs),
        num_training_steps=len(train_loader) * epochs,
    )

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()

        if epoch == freeze_epochs:
            for p in model.bert.bert.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * len(train_loader) * (epochs - freeze_epochs)),
                num_training_steps=len(train_loader) * (epochs - freeze_epochs),
            )

        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits

            loss_fct = CrossEntropyLoss(weight=weights.to(device), ignore_index=-100)
            loss = loss_fct(logits.view(-1, len(LABELS)), batch["labels"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Training Loss: {total_loss / len(train_loader):.4f}")

        if val_loader is not None and epoch % 10 == 0:
            print(f"Epoch {epoch+1} Validation:")
            evaluate_motif_model(model, val_loader, device)

        torch.save(model.state_dict(), save_model_path)
        print(f"Triple classifier model saved to {save_model_path}")

def evaluate_motif_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    loss_fct = CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
            total_loss += loss.item()

            for true_seq, pred_seq in zip(labels, preds):
                for true_label, pred_label in zip(true_seq, pred_seq):
                    if true_label != -100:
                        all_labels.append(true_label.item())
                        all_preds.append(pred_label.item())

    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    avg_loss  = total_loss / len(dataloader)

    print("Evaluation metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Val Loss:  {avg_loss:.4f}")
    print("\nClassification Report:\n", classification_report(
        all_labels, all_preds,
        labels=list(ID2LABEL.keys()),
        target_names=LABELS,
        zero_division=0
    ))

    return avg_loss, precision, recall, f1

def load_triple_extractor(model_path="D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/triple_classifier.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertTokenClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def is_valid_token(token):
    return token not in string.punctuation and not token.startswith("##")

def decode_triples(tokens, label_ids):
    triples = []
    subj, pred, obj = [], [], []
    current = None

    def add_token(token_list, token):
        if not is_valid_token(token):
            return
        if token.startswith("##") and token_list:
            token_list[-1] += token[2:]
        else:
            token_list.append(token)

    for token, label_id in zip(tokens, label_ids):
        label = ID2LABEL.get(label_id, "O")

        if label == "B-source":
            if subj and pred and obj:
                triples.append((" ".join(subj), " ".join(pred), " ".join(obj)))
                subj, pred, obj = [], [], []
            subj = []
            add_token(subj, token)
            current = "subj"

        elif label == "I-source" and current == "subj":
            add_token(subj, token)

        elif label == "B-PRED":
            if subj and pred and obj:
                triples.append((" ".join(subj), " ".join(pred), " ".join(obj)))
                subj, pred, obj = subj, [], []
            pred = []
            add_token(pred, token)
            current = "pred"

        elif label == "I-PRED" and current == "pred":
            add_token(pred, token)

        elif label == "B-OBJ":
            if subj and pred and obj:
                triples.append((" ".join(subj), " ".join(pred), " ".join(obj)))
                subj, pred, obj = subj, pred, []
            obj = []
            add_token(obj, token)
            current = "obj"

        elif label == "I-OBJ" and current == "obj":
            add_token(obj, token)

        elif label == "O":
            current = None

    if subj and pred and obj:
        triples.append((" ".join(subj), " ".join(pred), " ".join(obj)))

    return triples

def extract_triples_from_text(model, text, max_len=256, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_labels = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    filtered_tokens, filtered_labels = [], []
    for tok, lab in zip(tokens, pred_labels):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        filtered_tokens.append(tok)
        filtered_labels.append(lab)
    return decode_triples(filtered_tokens, filtered_labels)

nlp = spacy.load("en_core_web_sm")