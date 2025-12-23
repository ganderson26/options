# phonetic_allision_mining.py

# --- 1. Setup ---
# pip install metaphone rapidfuzz pandas networkx matplotlib nltk tqdm

import re
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from metaphone import doublemetaphone
from rapidfuzz.distance import Levenshtein
import nltk
nltk.download('punkt')

# --- 2. Helper functions ---
def tokenize(text):
    """Word tokenizer: alphabetic only."""
    return re.findall(r"[a-zA-Z]+", text.lower())

def phonetic_keys(word):
    """Double Metaphone keys (primary + secondary)."""
    return [k for k in doublemetaphone(word) if k]

def build_ref_index(ref_corpus):
    """
    ref_corpus = {source_name: full_text}
    Returns dict {phonetic_key -> list of (token, source)}.
    """
    index = {}
    for source, text in ref_corpus.items():
        for token in tokenize(text):
            for key in phonetic_keys(token):
                if key:
                    index.setdefault(key, []).append((token, source))
    return index

def fw_sliding_windows(text, win_size=5, step=2):
    """Generate sliding windows of tokens from FW text."""
    tokens = tokenize(text)
    for i in range(0, len(tokens) - win_size + 1, step):
        yield tokens[i:i+win_size], i

def phonetic_candidates(tokens, ref_index, threshold=0.35):
    """
    Compare FW token group vs. reference index.
    Returns candidate matches with phonetic similarity scores.
    """
    results = []
    for token in tokens:
        for fw_key in phonetic_keys(token):
            for ref_key, ref_items in ref_index.items():
                dist = Levenshtein.distance(fw_key, ref_key)
                score = 1 - dist / max(len(fw_key), len(ref_key))
                if score >= threshold:
                    for ref_token, source in ref_items:
                        results.append({
                            "fw_token": token,
                            "fw_key": fw_key,
                            "ref_token": ref_token,
                            "ref_key": ref_key,
                            "score": round(score, 3),
                            "source": source
                        })
    return results

# Main

# Example corpus (replace with full Gutenberg loads)
#ref_corpus = {
#    "Bible": open("data/bible_kjv.txt").read(),
#    "Shakespeare": open("data/shakespeare.txt").read(),
#    "Dante": open("data/dante_inferno.txt").read()
#}

ref_corpus = {
    "Dante": open("data/dante_extract.txt").read()
}

# Build phonetic index once
ref_index = build_ref_index(ref_corpus)
print("Loaded Reference Data")

# Finnegans Wake text (Project Gutenberg or custom edition)
fw_text = open("data/finnegans_wake_extract.txt").read()
print("Loaded FW Data")

all_results = []

for window, pos in tqdm(list(fw_sliding_windows(fw_text, win_size=5, step=3))):
    matches = phonetic_candidates(window, ref_index, threshold=0.4)
    for m in matches:
        m["fw_window"] = " ".join(window)
        m["position"] = pos
        all_results.append(m)

df = pd.DataFrame(all_results)
df.to_csv("fw_phonetic_matches.csv", index=False)

# Top candidate matches
df.sort_values("score", ascending=False).head(20)
print(df)

# Build bipartite graph FW <-> Source tokens
'''
G = nx.Graph()

for _, row in df[df.score > 0.5].iterrows():
    fw_node = f"FW:{row.fw_token}"
    src_node = f"{row.source}:{row.ref_token}"
    G.add_edge(fw_node, src_node, weight=row.score)

plt.figure(figsize=(10,8))
pos = nx.spring_layout(G, k=0.4)
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Phonetic Allusion Graph (FW → Sources)")
plt.axis("off")
plt.show()
'''

