"""
finnegans_transgender_search_final.py

Usage:
    python finnegans_transgender_search_final.py data/finnegans_wake.txt --both

Requirements:
    pip install sentence-transformers scikit-learn nltk pandas tqdm fuzzywuzzy python-levenshtein transformers torch
"""

import sys, re, argparse
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm

# Embeddings
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from sklearn.cluster import AgglomerativeClustering
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Optional zero-shot
# from transformers import pipeline

# ----------------------------
# Seed categories
# ----------------------------

SEED_PHRASES_PHYSICAL = [
    "sex change", "changing sex", "change of sex",
    "transsexual", "gender reassignment",
    "become a woman", "become a man",
    "born a woman", "born a man",
    "hermaphrodite", "intersex",
    "manhood", "womanhood", "both sexes"
]
SEED_SENTENCES_PHYSICAL = [
    "a person changing from male to female",
    "a person changing from female to male",
    "surgery to alter sex",
    "living as a woman though born male",
    "living as a man though born female",
    "a body that is both male and female"
]

SEED_PHRASES_ROLE = [
    "cross-dress", "cross dressing", "dressed as a man", "dressed as a woman",
    "ladyman", "mannish", "woman-man", "man-woman", "boy-girl", "she-he",
    "two-spirit", "androgyny", "androgynous",
    "genderless", "sexless", "ambiguous gender",
    "switching roles", "wearing the other's clothes", "gender identity"
]
SEED_SENTENCES_ROLE = [
    "dressing as the opposite sex",
    "taking on the gender role of the other sex",
    "ambiguous gender presentation",
    "a person living as another gender",
    "someone genderless or without sex",
    "crossing social gender boundaries"
]

REGEX_PATTERNS = [
    re.compile(r"\b(androgyn|androgyne|androgynous)\b", re.I),
    re.compile(r"\b(herma?phrodit(e|ic)?)\b", re.I),
    re.compile(r"\b(trans(?:-?sexual|(?:-?gender)?))\b", re.I),
    re.compile(r"\b(cross[\-\s]?dress(?:ing)?)\b", re.I),
    re.compile(r"\b(manhood|womanhood|masculin|feminin)\b", re.I),
    re.compile(r"\b(shemale|maleish|femaleish|bothish)\b", re.I),
    re.compile(r"\b(?:man(?:[-\s]?woman|woman)|woman(?:[-\s]?man|man))\b", re.I),
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_THRESHOLD = 0.56
FUZZY_THRESHOLD = 85
CLUSTER_NUM = 12
CONTEXT_WINDOW = 1

# ----------------------------
# Helpers
# ----------------------------

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def heuristic_sentence_split(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sents = []
    for p in paragraphs:
        for sent in sent_tokenize(p):
            pieces = re.split(r'(?<=[;——:])\s+(?=[A-Z0-9“"\'\(\[])', sent)
            for piece in pieces:
                piece = piece.strip()
                if piece:
                    sents.append(piece)
    return [s for s in sents if len(re.sub(r'\s+','',s)) > 0]

def build_embeddings(sentences: List[str]):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
    return model, embeddings

def cosine_sim(a: np.ndarray, b: np.ndarray):
    return util.cos_sim(a, b).cpu().numpy()

def fuzzy_keyword_matches(sentences, keywords, threshold=FUZZY_THRESHOLD):
    results = []
    for i, s in enumerate(sentences):
        for kw in keywords:
            score = fuzz.partial_ratio(kw.lower(), s.lower())
            if score >= threshold:
                results.append((i, s, score, kw))
    return results

def regex_matches(sentences, patterns):
    results = []
    for i, s in enumerate(sentences):
        for pat in patterns:
            m = pat.search(s)
            if m:
                results.append((i, s, m.group(0)))
    return results

def semantic_search(sentences, embeddings, seed_sentences, model, threshold=EMBED_THRESHOLD):
    seed_emb = model.encode(seed_sentences, convert_to_numpy=True)
    sims = cosine_sim(seed_emb, embeddings)
    max_sim_per_sentence = sims.max(axis=0)
    hits = []
    for idx, score in enumerate(max_sim_per_sentence):
        if score >= threshold:
            seed_idx = sims[:, idx].argmax()
            hits.append({"idx": idx, "sentence": sentences[idx], "score": float(score), "best_seed": seed_sentences[seed_idx]})
    return sorted(hits, key=lambda x: x["score"], reverse=True)

def cluster_candidates(embeddings, candidate_indices, num_clusters=CLUSTER_NUM):
    if not candidate_indices: return {}
    cand_emb = embeddings[candidate_indices]
    k = min(num_clusters, len(candidate_indices))
    clustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(cand_emb)
    return dict(zip(candidate_indices, labels))

def extract_context(sentences, idx, window=CONTEXT_WINDOW):
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])

def highlight_text(sentence: str, matches: List[str]) -> str:
    highlighted = sentence
    for m in matches:
        phrase = re.sub(r"\(.*?\)", "", m).strip()
        if not phrase: continue
        highlighted = re.sub(
            rf"({re.escape(phrase)})",
            r"<mark>\1</mark>",
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted

def highlight_category(val: str) -> str:
    if "physical" in val and "role" in val:
        return "background-color: #d8b4fe; color: black;"  # purple
    elif "physical" in val:
        return "background-color: #93c5fd; color: black;"  # blue
    elif "role" in val:
        return "background-color: #86efac; color: black;"  # green
    return ""

# ----------------------------
# Main analysis
# ----------------------------
def analyze_text_file(path: str, out_csv: str, run_zero_shot=False,
                      seed_phrases_physical=None, seed_sentences_physical=None,
                      seed_phrases_role=None, seed_sentences_role=None,
                      both_mode=False, category_label="unspecified"):

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    raw = load_text(path)
    sentences = heuristic_sentence_split(raw)
    print(f"[INFO] Loaded {len(sentences)} sentence-like fragments.")

    model, embeddings = build_embeddings(sentences)
    candidate_info = {}

    def add_hit(idx, match_type, match_value, category):
        if idx not in candidate_info:
            candidate_info[idx] = {"regex": [], "fuzzy": [], "semantic": [], "categories": set()}
        candidate_info[idx][match_type].append(match_value)
        candidate_info[idx]["categories"].add(category)

    # Regex matches
    regex_hits = regex_matches(sentences, REGEX_PATTERNS)
    for i, s, matched in regex_hits:
        add_hit(i, "regex", matched, category_label if not both_mode else "general")

    # Physical
    if seed_phrases_physical and seed_sentences_physical:
        for i, s, score, kw in fuzzy_keyword_matches(sentences, seed_phrases_physical):
            add_hit(i, "fuzzy", f"{kw}({score})", "physical")
        for h in semantic_search(sentences, embeddings, seed_sentences_physical, model):
            add_hit(h["idx"], "semantic", f"{h['best_seed']}({h['score']:.3f})", "physical")

    # Role
    if seed_phrases_role and seed_sentences_role:
        for i, s, score, kw in fuzzy_keyword_matches(sentences, seed_phrases_role):
            add_hit(i, "fuzzy", f"{kw}({score})", "role")
        for h in semantic_search(sentences, embeddings, seed_sentences_role, model):
            add_hit(h["idx"], "semantic", f"{h['best_seed']}({h['score']:.3f})", "role")

    candidate_indices = sorted(candidate_info.keys())
    print(f"[INFO] Found {len(candidate_indices)} candidate sentences.")

    clustering_map = cluster_candidates(embeddings, candidate_indices)

    # Optional zero-shot can be added here (omitted for simplicity)

    rows = []
    for idx in candidate_indices:
        entry = candidate_info[idx]
        categories = ", ".join(sorted(entry["categories"]))
        matches_all = entry["regex"] + entry["fuzzy"] + entry["semantic"]
        highlighted_sentence = highlight_text(sentences[idx], matches_all)
        rows.append({
            "index": idx,
            "sentence": highlighted_sentence,
            "context": extract_context(sentences, idx, CONTEXT_WINDOW),
            "regex_matches": "; ".join(entry["regex"]),
            "fuzzy_matches": "; ".join(entry["fuzzy"]),
            "semantic_matches": "; ".join(entry["semantic"]),
            "cluster_label": clustering_map.get(idx, ""),
            "zero_shot_raw": "",
            "match_category": categories
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # HTML export
    styled = df.style.applymap(highlight_category, subset=["match_category"])
    out_html = out_csv.replace(".csv", ".html")
    styled.to_html(out_html, escape=False)
    print(f"[RESULT] Wrote {len(df)} rows to {out_csv} and {out_html}")
    return df

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Finnegans Wake for transgender-related allusions.")
    parser.add_argument("infile", help="Finnegans Wake text file")
    parser.add_argument("--physical", action="store_true", help="Search physical sex-change imagery")
    parser.add_argument("--role", action="store_true", help="Search identity/role-blending imagery")
    parser.add_argument("--both", action="store_true", help="Search both categories")
    args = parser.parse_args()

    if args.physical:
        category = "physical"
        analyze_text_file(args.infile, out_csv=f"jw_transgender_candidates_{category}.csv",
                          seed_phrases_physical=SEED_PHRASES_PHYSICAL,
                          seed_sentences_physical=SEED_SENTENCES_PHYSICAL,
                          category_label=category)
    elif args.role:
        category = "role"
        analyze_text_file(args.infile, out_csv=f"jw_transgender_candidates_{category}.csv",
                          seed_phrases_role=SEED_PHRASES_ROLE,
                          seed_sentences_role=SEED_SENTENCES_ROLE,
                          category_label=category)
    elif args.both:
        category = "both"
        analyze_text_file(args.infile, out_csv=f"jw_transgender_candidates_{category}.csv",
                          seed_phrases_physical=SEED_PHRASES_PHYSICAL,
                          seed_sentences_physical=SEED_SENTENCES_PHYSICAL,
                          seed_phrases_role=SEED_PHRASES_ROLE,
                          seed_sentences_role=SEED_SENTENCES_ROLE,
                          both_mode=True,
                          category_label=category)
    else:
        category = "physical"
        analyze_text_file(args.infile, out_csv=f"jw_transgender_candidates_{category}.csv",
                          seed_phrases_physical=SEED_PHRASES_PHYSICAL,
                          seed_sentences_physical=SEED_SENTENCES_PHYSICAL,
                          category_label=category)
