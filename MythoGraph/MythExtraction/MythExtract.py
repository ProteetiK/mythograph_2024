from itertools import combinations_with_replacement
import spacy
from transformers import BertTokenizer
import torch
from MythModelTrain.MotifTrainer import load_motif_model
from MythModelTrain.TripleTrainer import load_triple_extractor, extract_triples_from_text

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model_path = "D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/motif_classifier.pt"
encoder_path = "D:/MythoGraph/MythoGraph/MythoGraph/MythModelTrain/model/label_encoder.json"

def get_core_noun(token):
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            nouns = [t.text.lower() for t in chunk if t.pos_ in ("NOUN", "PROPN")]
            if nouns:
                return " ".join(nouns)
    nouns = [t.text.lower() for t in token.subtree if t.pos_ in ("NOUN", "PROPN")]
    return " ".join(nouns) if nouns else token.text.lower()

def extract_characters(text):
    doc = nlp(text)
    characters = set()
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG"):
            characters.add(ent.text.title())
    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            characters.add(token.text.title())
    return characters

def build_coref_cache(doc):
    cache = {
        "he": None, "him": None, "his": None,
        "she": None, "her": None, "hers": None,
        "they": None, "them": None, "their": None
    }
    character_mentions = []

    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                character_mentions.append(ent.text)

        for token in sent:
            if token.pos_ == "PROPN" and not token.is_stop:
                character_mentions.append(token.text)

        for pronoun in cache:
            if character_mentions:
                cache[pronoun] = character_mentions[-1]

    return cache

def resolve_pronoun(token_text, coref_cache):
    token_text = token_text.lower()
    return coref_cache.get(token_text, token_text)


def extract_triples_with_nlp(text):
    doc = nlp(text.title())
    triples = []

    subject_deps = ("nsubj", "nsubjpass", "csubj", "agent", "expl")
    object_deps = ("dobj", "dative", "attr", "oprd", "xcomp", "ccomp", "acomp", "advcl", "relcl")

    characters = extract_characters(text)
    characters_lower = {c.lower() for c in characters}
    pronouns = {"he", "she", "they", "him", "her", "them", "his", "their", "i", "you"}

    coref_cache = build_coref_cache(doc)

    last_character_subject = None
    last_character_object = None

    def resolve_noun(token):
        noun = get_core_noun(token)
        if noun.lower() in pronouns:
            return resolve_pronoun(noun, coref_cache)
        return noun

    for sent in doc.sents:
        verbs = [token for token in sent if token.pos_ == "VERB"]
        for verb in verbs:
            subj = None
            objects = []

            for child in verb.children:
                if child.dep_ in subject_deps:
                    subj = resolve_noun(child)

            if not subj and verb.tag_ == "VBN":
                for child in verb.children:
                    if child.dep_ == "agent":
                        subj = resolve_noun(child)

            for child in verb.children:
                if child.dep_ in object_deps and child.pos_ != "VERB":
                    obj = resolve_noun(child)
                    if obj:
                        objects.append(obj)
                        for conj in child.conjuncts:
                            conj_obj = resolve_noun(conj)
                            if conj_obj:
                                objects.append(conj_obj)

                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj = resolve_noun(pobj)
                            if obj:
                                objects.append(obj)

            verb_lemma = verb.lemma_.lower()
            for child in verb.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    verb_lemma += f"_{child.lemma_.lower()}"

            if verb_lemma == "have" and any(o.lower() in {"hostility", "anger", "fear"} for o in objects):
                verb_lemma = "express"
            elif verb_lemma == "say":
                verb_lemma = "speak"

            if subj and not objects and verb_lemma in {"refuse", "reject", "deny"}:
                if last_character_subject and last_character_subject != subj:
                    objects = [last_character_subject]
                elif last_character_object and last_character_object != subj:
                    objects = [last_character_object]

            if subj and subj.lower() in characters_lower:
                last_character_subject = subj
            for o in objects:
                if o.lower() in characters_lower:
                    last_character_object = o

            if subj and objects:
                for obj in objects:
                    triples.append((subj, obj, verb_lemma, "TEMP"))

    filtered_triples = [
        (s, o, v, l) for s, o, v, l in triples
        if s.lower() in characters_lower or o.lower() in characters_lower
    ]

    return filtered_triples

def extract_triples_with_model(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, label_encoder = load_motif_model(model_path, encoder_path)
    model.to(device)
    model = load_triple_extractor()
    triples = extract_triples_from_text(model, text)
    return triples

def extract_triples_combined(text):
    model_triples = []
    nlp_triples_raw = []
    try:
        model_triples_raw = extract_triples_with_model(text)
    except Exception as e:
        print(f"[Model extractor error] {e}")
        model_triples_raw = []
    for triple in model_triples_raw:
        if len(triple) == 3:
            subj, pred, obj = triple
            model_triples.append((subj.strip(), obj.strip(), pred.strip()))
    try:
        nlp_triples_raw = extract_triples_with_nlp(text)
    except Exception as e:
        print(f"[NLP extractor error] {e}")
        nlp_triples_raw = []

    combined = model_triples + nlp_triples_raw
    return combined