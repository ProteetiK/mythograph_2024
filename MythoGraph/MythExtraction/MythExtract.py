import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_core_noun(token):
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            nouns = [t.text.lower() for t in chunk if t.pos_ in ("NOUN", "PROPN")]
            if nouns:
                return " ".join(nouns)
    nouns = [t.text.lower() for t in token.subtree if t.pos_ in ("NOUN", "PROPN")]
    if nouns:
        return " ".join(nouns)
    return token.text.lower()

def extract_characters(text):
    doc = nlp(text)
    characters = set()
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG"):
            characters.add(ent.text.title())
    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            if token.text.title() not in characters:
                characters.add(token.text.title())
    return characters

def build_coref_cache(doc):
    cache = {
        "he": None, "him": None, "his": None,
        "she": None, "her": None, "hers": None,
        "they": None, "them": None, "their": None
    }
    gender_map = {
        "he": "male", "him": "male", "his": "male",
        "she": "female", "her": "female", "hers": "female",
        "they": "neutral", "them": "neutral", "their": "neutral"
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
    doc = nlp(text.lower())
    triples = []

    subject_deps = ("nsubj", "nsubjpass", "csubj", "agent", "expl")
    object_deps = ("dobj", "dative", "attr", "oprd", "xcomp", "ccomp", "acomp", "advcl", "relcl")

    characters = extract_characters(text)
    characters_lower = {c.lower() for c in characters}
    pronouns = {"he", "she", "they", "him", "her", "them", "his", "their", "i", "you"}

    coref_cache = build_coref_cache(doc)

    for sent in doc.sents:
        verbs = [token for token in sent if token.pos_ == "VERB"]
        for verb in verbs:
            subj = None
            objects = []

            for child in verb.children:
                if child.dep_ in subject_deps:
                    subj_candidate = get_core_noun(child)
                    if subj_candidate.lower() in pronouns:
                        subj_candidate = resolve_pronoun(subj_candidate, coref_cache)
                    subj = subj_candidate

            if not subj and verb.tag_ == "VBN":
                for child in verb.children:
                    if child.dep_ == "agent":
                        subj = get_core_noun(child)

            for child in verb.children:
                if child.dep_ in object_deps:
                    if child.pos_ == "VERB":
                        obj_candidate = child.lemma_
                    else:
                        obj_candidate = get_core_noun(child)
                    if obj_candidate.lower() in pronouns:
                        obj_candidate = resolve_pronoun(obj_candidate, coref_cache)
                    if obj_candidate:
                        objects.append(obj_candidate)

            verb_lemma = verb.lemma_.lower()
            xcomp_verb = None
            for child in verb.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    xcomp_verb = child.lemma_.lower()

            if verb_lemma in {"have", "be", "do"} and xcomp_verb:
                verb_lemma = xcomp_verb

            verb_semantic = verb_lemma

            if verb_semantic == "have" and any(o.lower() in {"hostility", "anger", "fear"} for o in objects):
                verb_semantic = "express"
            elif verb_semantic == "say":
                verb_semantic = "speak"

            if subj and objects:
                for obj in objects:
                    triples.append((subj, obj, verb_semantic, "TEMP"))

    filtered_triples = []
    for (subj, obj, verb_semantic, label) in triples:
        if subj.lower() in characters_lower or obj.lower() in characters_lower:
            filtered_triples.append((subj, obj, verb_semantic, label))

    return filtered_triples