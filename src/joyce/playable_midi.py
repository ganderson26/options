# Phonetic Mapping of Joycean Texts to Musical Notes using Machine Learning

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pronouncing  # Library for phonetic analysis
from music21 import stream, note

# Sample Joycean text (excerpt from Finnegans Wake)
text = "riverrun, past Eve and Adam's"

# Step 1: Extract phonemes
words = text.split()
phonemes = []
for word in words:
    phones_list = pronouncing.phones_for_word(word)
    if phones_list:
        phonemes.append(phones_list[0].split())  # Take the first pronunciation

# Flatten the list of phonemes
phonemes = [p for sublist in phonemes for p in sublist]

# Step 2: Encode phonemes as numerical features
le = LabelEncoder()
phoneme_encoded = le.fit_transform(phonemes).reshape(-1, 1)

# Step 3: Example target mapping to MIDI notes (C4=60 to C6=84)
# Here, we use a simple regressor to "learn" a mapping pattern
# In practice, you could train on a larger corpus of text+music
np.random.seed(42)
target_midi = np.random.randint(60, 84, size=len(phoneme_encoded))

# Train a Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(phoneme_encoded, target_midi)

# Predict MIDI notes for phonemes
predicted_midi = model.predict(phoneme_encoded)

# Step 4: Generate a simple musical stream
melody = stream.Stream()
for midi_val in predicted_midi:
    n = note.Note(int(midi_val))
    n.quarterLength = 0.5
    melody.append(n)

# Save to MIDI file
melody.write('midi', fp='joyce_phonetic_mapping.mid')

print("MIDI file 'joyce_phonetic_mapping.mid' generated from phonetic mapping.")
