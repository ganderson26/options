"""
finnegans_transgender_search.py

Usage:
    python finnegans_transgender_search.py data/finnegans_wake.txt

What it does:
 - Loads a plaintext version of the book (user provides path).
 - Splits into sentences (with fallback heuristics for Joyce's unusual punctuation).
 - Performs:
    1) Lexical + regex + fuzzy searches for seed terms and morphological variants.
    2) Semantic search using sentence-transformers embeddings (cosine similarity to seed sentences).
    3) Agglomerative clustering (to group candidate sentences).
    4) Optional zero-shot classification with a NLI model (commented as optional: requires transformers + torch).
 - Produces a CSV of candidate sentences with scores and context.

Requirements (install via pip):
    pip install sentence-transformers scikit-learn nltk pandas tqdm python-levenshtein fuzzywuzzy transformers torch

Notes:
 - Sentence-splitting for Joyce is imperfect; you may want to tune the regex/split heuristics.
 - Adjust thresholds for precision/recall to taste.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

# Embeddings
from sentence_transformers import SentenceTransformer, util

# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Fuzzy matching
from fuzzywuzzy import fuzz

# Sentence splitting (nltk fallback)
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Optional zero-shot classification (uncomment to use)
# from transformers import pipeline

# ----------------------------
# Configuration / seeds
# ----------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small & fast; change if you need higher quality
EMBED_THRESHOLD = 0.56  # cosine similarity threshold for candidate detection (tune)
FUZZY_THRESHOLD = 85     # fuzzy ratio threshold for lexical fuzzy matches
CLUSTER_NUM = 12         # how many clusters to group candidate sentences into
CONTEXT_WINDOW = 1       # how many sentences of context to include on each side

# Seed phrases (mix modern terminology and synonyms / paraphrases)
SEED_PHRASES = [
    # explicit modern-ish terms (may not appear verbatim in Joyce)
    "transgender", "transsexual", "transvestite", "sex change", "gender identity",
    "cross-dress", "cross dressing", "become a woman", "become a man", "born a woman",
    "born a man", "manhood", "womanhood", "androgyne", "androgyny", "hermaphrodite",
    "two-spirit", "changing sex", "gendered", "genderless", "sexless",
    # Joyce-y / archaic possibilities
    "ladyman", "mannish", "woman-man", "women men", "man-woman", "boy-girl", "she-he",
    # conceptual paraphrases that could suggest boundary-crossing
    "dressed as a man", "dressed as a woman", "disguise", "changed sex", "become other",
    "switching roles", "wearing the other's clothes"
]

# Short seed sentences for semantic similarity (phrases capturing the phenomenon)
SEED_SENTENCES = [
    "a person living as a gender different from their assigned sex",
    "dressing as the opposite sex",
    "changing one's sex",
    "living as a woman though born male",
    "living as a man though born female",
    "ambiguous gender or both sexes",
    "taking on a different gender role",
    "crossing gender boundaries"
]

# Regex patterns to find morphological variants and Joyce-like spellings
REGEX_PATTERNS = [
    re.compile(r"\b(androgyn|androgyne|androgynous)\b", re.I),
    re.compile(r"\b(herma?phrodit(e|ic)?)\b", re.I),
    re.compile(r"\b(trans(?:-?sexual|(?:-?gender)?))\b", re.I),
    re.compile(r"\b(cross[\-\s]?dress(?:ing)?)\b", re.I),
    re.compile(r"\b(manhood|womanhood|masculin|feminin)\b", re.I),
    re.compile(r"\b(shemale|maleish|femaleish|bothish)\b", re.I),
    # catch Joyce's compounding tendencies: e.g. 'womanman', 'manwoman', etc.
    re.compile(r"\b(?:man(?:[-\s]?woman|woman)|woman(?:[-\s]?man|man))\b", re.I),
]

# ----------------------------
# Helpers
# ----------------------------
def load_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt

def heuristic_sentence_split(text: str) -> List[str]:
    """
    Use NLTK sentence tokenizer as baseline and then apply additional splitting
    for Joyce's long run-ons (split on multiple newlines, long em dashes, '—', etc.).
    """
    # First normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Break on double newlines (Joyce often uses long paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sents = []
    for p in paragraphs:
        # NLTK first
        for sent in sent_tokenize(p):
            # further split on long em-dash sequences or semicolons followed by capitalization
            pieces = re.split(r'(?<=[;——:])\s+(?=[A-Z0-9“"\'\(\[])', sent)
            for piece in pieces:
                piece = piece.strip()
                if piece:
                    sents.append(piece)
    # Final cleanup: collapse many short fragments that are punctuation-only
    cleaned = [s for s in sents if len(re.sub(r'\s+','',s)) > 0]
    return cleaned

def build_embeddings(sentences: List[str], model_name: str = EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
    return model, embeddings

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (n, d) b: (m, d)
    return util.cos_sim(a, b).cpu().numpy()

def fuzzy_keyword_matches(sentences: List[str], keywords: List[str], threshold: int = FUZZY_THRESHOLD) -> List[Tuple[int, str, int, str]]:
    """Return list of (index, sentence, score, matching_keyword) for fuzzy matches above threshold."""
    results = []
    for i, s in enumerate(sentences):
        for kw in keywords:
            score = fuzz.partial_ratio(kw.lower(), s.lower())
            if score >= threshold:
                results.append((i, s, score, kw))
    return results

def regex_matches(sentences: List[str], patterns: List[re.Pattern]) -> List[Tuple[int, str, str]]:
    results = []
    for i, s in enumerate(sentences):
        for pat in patterns:
            m = pat.search(s)
            if m:
                results.append((i, s, m.group(0)))
    return results

def semantic_search(sentences: List[str], embeddings: np.ndarray, seed_sentences: List[str], model: SentenceTransformer, threshold: float = EMBED_THRESHOLD) -> List[Dict]:
    seed_emb = model.encode(seed_sentences, convert_to_numpy=True)
    sims = cosine_sim(seed_emb, embeddings)  # shape (num_seeds, num_sentences)
    # For each sentence, take max similarity across seeds
    max_sim_per_sentence = sims.max(axis=0)
    hits = []
    for idx, score in enumerate(max_sim_per_sentence):
        if score >= threshold:
            # find the best seed
            seed_idx = sims[:, idx].argmax()
            hits.append({"idx": idx, "sentence": sentences[idx], "score": float(score), "best_seed": seed_sentences[seed_idx]})
    # sort by score desc
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    return hits

def cluster_candidates(embeddings: np.ndarray, candidate_indices: List[int], num_clusters: int = CLUSTER_NUM):
    if not candidate_indices:
        return {}
    cand_emb = embeddings[candidate_indices]
    # if there are fewer candidates than clusters, reduce k
    k = min(num_clusters, len(candidate_indices))
    clustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(cand_emb)
    return dict(zip(candidate_indices, labels))

def extract_context(sentences: List[str], idx: int, window: int = CONTEXT_WINDOW):
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])

# Optional: zero-shot classifier using NLI (uncomment to use)
def zero_shot_labeling(sentences: List[str], candidate_indices: List[int], hypothesis_template: str = "This sentence refers to someone changing or living in a gender different from their assigned sex."):
    """
    This uses an NLI model (e.g. facebook/bart-large-mnli) to score entailment between
    the sentence (premise) and a hypothesis about transgender allusion.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = {}
    for idx in candidate_indices:
        premise = sentences[idx]
        # We convert NLI to a label-ish by asking entailment probability
        out = classifier(premise, candidate_labels=["entailment", "neutral", "contradiction"], hypothesis_template=hypothesis_template)
        # Note: transformers zero-shot returns classification in a slightly different format if using alternative pipelines; adapt as needed.
        # For simplicity, we keep the highest class and score for "entailment"
        # This block may need tweaks depending on pipeline version.
        results[idx] = out
    return results

# ----------------------------
# Main driver
# ----------------------------
def analyze_text_file(path: str, out_csv: str = "jw_transgender_candidates.csv", run_zero_shot: bool = False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Provide a plaintext UTF-8 file.")
    raw = load_text(path)
    sentences = heuristic_sentence_split(raw)
    print(f"[INFO] Loaded {len(sentences)} sentence-like fragments from {path.name}.")

    # Build embeddings
    print("[INFO] Building embeddings (this may take a moment)...")
    model, embeddings = build_embeddings(sentences, model_name=EMBEDDING_MODEL)

    # 1) regex matches
    print("[INFO] Running regex searches...")
    regex_hits = regex_matches(sentences, REGEX_PATTERNS)

    # 2) fuzzy keyword matches
    print("[INFO] Running fuzzy keyword searches...")
    fuzzy_hits = fuzzy_keyword_matches(sentences, SEED_PHRASES, threshold=FUZZY_THRESHOLD)

    # 3) semantic search
    print("[INFO] Running semantic similarity search...")
    sem_hits = semantic_search(sentences, embeddings, SEED_SENTENCES, model, threshold=EMBED_THRESHOLD)

    # Gather candidate indices
    candidate_indices = set([h[0] for h in regex_hits] + [h[0] for h in fuzzy_hits] + [h["idx"] for h in sem_hits])
    candidate_indices = sorted(candidate_indices)
    print(f"[INFO] Found {len(candidate_indices)} candidate sentences (combined methods).")

    # 4) cluster candidates to group related hits
    clustering_map = cluster_candidates(embeddings, candidate_indices, num_clusters=CLUSTER_NUM)

    # 5) optional zero-shot
    zero_shot_results = {}
    if run_zero_shot and candidate_indices:
        print("[INFO] Running optional zero-shot NLI classifier (requires internet first run to download model)...")
        zero_shot_results = zero_shot_labeling(sentences, candidate_indices)

    # Compose rows for CSV
    rows = []
    # Build quick lookup maps for match metadata
    regex_map = {i:[] for i in candidate_indices}
    for i, s, matched in regex_hits:
        if i in regex_map: regex_map[i].append(matched)

    fuzzy_map = {i:[] for i in candidate_indices}
    for i, s, score, kw in fuzzy_hits:
        if i in fuzzy_map: fuzzy_map[i].append(f"{kw}({score})")

    sem_map = {i: None for i in candidate_indices}
    for h in sem_hits:
        if h["idx"] in sem_map:
            sem_map[h["idx"]] = (h["score"], h["best_seed"])

    for idx in candidate_indices:
        sent = sentences[idx]
        rows.append({
            "index": idx,
            "sentence": sent,
            "context": extract_context(sentences, idx, window=CONTEXT_WINDOW),
            "regex_matches": "; ".join(regex_map.get(idx, [])) or "",
            "fuzzy_matches": "; ".join(fuzzy_map.get(idx, [])) or "",
            "semantic_score": round(sem_map[idx][0], 4) if sem_map.get(idx) else "",
            "semantic_best_seed": sem_map[idx][1] if sem_map.get(idx) else "",
            "cluster_label": clustering_map.get(idx, ""),
            "zero_shot_raw": str(zero_shot_results.get(idx, "")) if zero_shot_results else ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[RESULT] Wrote {len(df)} candidate rows to {out_csv}. Inspect manually for literary interpretation.")
    return df

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python finnegans_transgender_search.py data/finnegans_wake_extract.txt")
        sys.exit(1)
    infile = sys.argv[1]
    # By default don't run zero-shot (heavy). If you want to enable, set second arg 'nli'
    run_nli = (len(sys.argv) >= 3 and sys.argv[2].lower() in ("nli", "zero-shot", "nli-on"))
    df = analyze_text_file(infile, out_csv="jw_transgender_candidates.csv", run_zero_shot=run_nli)
    print(df.head(20).to_string(index=False))
