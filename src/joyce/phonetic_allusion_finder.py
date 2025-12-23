# --- 1. Setup ---
# pip install metaphone rapidfuzz

from metaphone import doublemetaphone
from rapidfuzz.distance import Levenshtein
import re
import pandas as pd

# --- 2. Helper functions ---
def tokenize(text):
    """Basic tokenizer: words only, lowercase."""
    return re.findall(r"[a-zA-Z]+", text.lower())

def phonetic_keys(word):
    """Return list of non-empty Double Metaphone keys."""
    return [k for k in doublemetaphone(word) if k]

def build_ref_index(ref_corpus):
    """
    Build a phonetic index: key -> list of (token, source).
    ref_corpus = {source_name: text}
    """
    index = {}
    for source, text in ref_corpus.items():
        for token in tokenize(text):
            for key in phonetic_keys(token):
                index.setdefault(key, []).append((token, source))
    return index

def phonetic_candidates(fw_text, ref_index):
    """
    Find phonetic matches between FW text and reference index.
    Returns DataFrame of candidates with scores.
    """
    results = []
    for token in tokenize(fw_text):
        
        for fw_key in phonetic_keys(token):
            
            for ref_key, ref_items in ref_index.items():
                
                dist = Levenshtein.distance(fw_key, ref_key)
                score = 1 - dist / max(len(fw_key), len(ref_key))
                
                if score > 0.0:  # filter weak matches
                    
                    for ref_token, source in ref_items:
                        
                        results.append({
                            "fw_token": token,
                            "fw_key": fw_key,
                            "ref_token": ref_token,
                            "ref_key": ref_key,
                            "score": round(score, 3),
                            "source": source
                        })
                   
    return pd.DataFrame(results).sort_values("score", ascending=False)


# Main
# Reference corpus: toy sample (could load whole Bible, Shakespeare, etc.)
ref_corpus = {
    "Bible": "In the beginning God created the heaven and the earth. "
             "Babylon fell. Tribulation and thunder rolled.",
    "Shakespeare": "Full fathom five thy father lies; "
                   "Bronze and Barabbas echo through the ages."
}

# Build reference index
ref_index = build_ref_index(ref_corpus)
#print(ref_index)

# FW snippet: the thunderword
fw_snippet = ("bababadalgharaghtakamminarronnkonnbronntonnerronntuonnthunntrovarrhounawnskawn"
              "toohoohoordenenthurnuk")

# Run phonetic match
candidates = phonetic_candidates(fw_snippet, ref_index)
print(candidates.head(15))

